# ===========================================================
#  主題 : DDPG for simplified 5-axis machining parameter optimization
#  目標 : 使學員了解  DDPG方法 & 五軸加工機控制系統於深度強化學習應用
#  作者 : 國立雲林科技大學電機系 林家仁
# ============================================================

# !pip install torch matplotlib
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import matplotlib.pyplot as plt
import time

# ----------------------------
# 1) Environment / Simulation (function style)
# ----------------------------
# Ranges for machining parameters (action mapping)
SPINDLE_MIN, SPINDLE_MAX = 1000.0, 6000.0      # rpm
FEED_MIN, FEED_MAX = 50.0, 800.0              # mm/min
DEPTH_MIN, DEPTH_MAX = 0.05, 2.0              # mm
ANGLE_MIN, ANGLE_MAX = -30.0, 30.0            # degrees

# Episode / segment setup
SEGMENTS = 20              # number of sub-steps per workpiece
SEGMENT_LENGTH = 10.0      # arbitrary unit length per segment (affects time calc)

# Wear & failure thresholds
TOOL_WEAR_THRESHOLD = 1.5  # normalized units (when exceed --> fail)
SURFACE_ERROR_LIMIT = 10.0 # microns threshold for "bad" surface

# Reward weights (tuneable)
W_TIME = 0.4
W_ERROR = 0.4
W_WEAR = 0.2

# Random seed for reproducibility
R_SEED = 0
random.seed(R_SEED)
np.random.seed(R_SEED)
torch.manual_seed(R_SEED)

def action_to_params(a):
    """Map action in [-1,1]^4 to physical parameters."""
    a = np.clip(a, -1, 1)
    spindle = SPINDLE_MIN + (a[0] + 1) * 0.5 * (SPINDLE_MAX - SPINDLE_MIN)
    feed    = FEED_MIN    + (a[1] + 1) * 0.5 * (FEED_MAX - FEED_MIN)
    depth   = DEPTH_MIN   + (a[2] + 1) * 0.5 * (DEPTH_MAX - DEPTH_MIN)
    angle   = ANGLE_MIN   + (a[3] + 1) * 0.5 * (ANGLE_MAX - ANGLE_MIN)
    return spindle, feed, depth, angle

def normalize_state(params, tool_wear, surface_error):
    """Return normalized observation in roughly [-1,1]."""
    spindle, feed, depth, angle = params
    s_spindle = (spindle - SPINDLE_MIN) / (SPINDLE_MAX - SPINDLE_MIN) * 2 - 1
    s_feed    = (feed - FEED_MIN) / (FEED_MAX - FEED_MIN) * 2 - 1
    s_depth   = (depth - DEPTH_MIN) / (DEPTH_MAX - DEPTH_MIN) * 2 - 1
    s_angle   = (angle - ANGLE_MIN) / (ANGLE_MAX - ANGLE_MIN) * 2 - 1
    s_wear    = np.tanh(tool_wear)   # bounded
    s_err     = np.tanh(surface_error / 50.0)  # scale
    return np.array([s_spindle, s_feed, s_depth, s_angle, s_wear, s_err], dtype=np.float32)

def simulate_segment(prev_params, action_params, prev_wear):
    """
    Simulate one machining segment given previous params and chosen params.
    Returns:
      - seg_time (sec-like arbitrary unit)
      - surface_error (microns)
      - new_tool_wear (accumulated)
      - success_flag (bool) -> whether this segment causes immediate failure (very high wear or cut)
    """
    # Unpack
    spindle, feed, depth, angle = action_params

    # --- Simplified physics relationships (toy models) ---
    # Cutting intensity: higher feed & depth => higher cutting force; spindle mitigates
    cutting_force = 1.0 * (feed/100.0) * (depth / 0.1) / max(spindle/1000.0, 0.1)
    # Angle penalty: extreme tilt increases surface error slightly
    angle_penalty = 1.0 + 0.01 * (abs(angle) / 10.0)
    # Vibration effect (noise)
    noise = np.random.normal(0.0, 0.02)

    # Tool wear increment (cumulative); proportional to cutting_force
    wear_increment = 0.0008 * cutting_force * (1.0 + 0.2*np.random.rand())

    new_wear = prev_wear + wear_increment

    # Surface error model (microns): base error + components
    base_error = 1.0  # baseline
    error_from_feed = 0.02 * (feed / 100.0)
    error_from_depth = 0.5 * depth
    error_from_wear = 3.0 * new_wear
    error_from_spindle = -0.0005 * spindle   # higher spindle reduces error slightly
    surface_error = base_error + error_from_feed + error_from_depth + error_from_wear + error_from_spindle
    surface_error *= angle_penalty
    surface_error *= (1.0 + noise)
    surface_error = max(0.0, surface_error)

    # Time model for the segment (arbitrary): longer if feed small
    # Convert feed mm/min to mm/sec: feed / 60
    feed_mm_per_sec = max(feed / 60.0, 1e-3)
    seg_time = SEGMENT_LENGTH / feed_mm_per_sec  # seconds-ish

    # Immediate failure if wear too high or ridiculous cutting force
    fail = False
    if new_wear > TOOL_WEAR_THRESHOLD * 3:  # catastrophic
        fail = True

    return seg_time, surface_error, new_wear, fail

# ----------------------------
# 2) DDPG Agent (PyTorch)
# ----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Actor: maps state (6-dim) -> action (4-dim in [-1,1])
class ActorNet(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, action_dim), nn.Tanh()
        )
    def forward(self, s):
        return self.net(s)

# Critic: Q(s,a)
class CriticNet(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 1)
        )
    def forward(self, s, a):
        x = torch.cat([s, a], dim=1)
        return self.net(x)

state_dim = 6   # [spindle_norm, feed_norm, depth_norm, angle_norm, wear, err]
action_dim = 4

actor = ActorNet(state_dim, action_dim).to(device)
actor_target = ActorNet(state_dim, action_dim).to(device)
actor_target.load_state_dict(actor.state_dict())
actor_opt = optim.Adam(actor.parameters(), lr=1e-3)

critic = CriticNet(state_dim, action_dim).to(device)
critic_target = CriticNet(state_dim, action_dim).to(device)
critic_target.load_state_dict(critic.state_dict())
critic_opt = optim.Adam(critic.parameters(), lr=1e-3)

replay = deque(maxlen=50000)
BATCH = 64
GAMMA = 0.99
TAU = 0.005
mse_loss = nn.MSELoss()

def select_action(state_np, noise_scale=0.2):
    s = torch.FloatTensor(state_np.reshape(1,-1)).to(device)
    actor.eval()
    with torch.no_grad():
        a = actor(s).cpu().numpy().flatten()
    actor.train()
    a = a + noise_scale * np.random.randn(action_dim)
    return np.clip(a, -1.0, 1.0)

def soft_update(net, net_target, tau):
    for p, pt in zip(net.parameters(), net_target.parameters()):
        pt.data.copy_(tau * p.data + (1 - tau) * pt.data)

def train_from_replay():
    if len(replay) < BATCH:
        return
    batch = random.sample(replay, BATCH)
    s_b = torch.FloatTensor(np.array([b[0] for b in batch])).to(device)
    a_b = torch.FloatTensor(np.array([b[1] for b in batch])).to(device)
    r_b = torch.FloatTensor(np.array([b[2] for b in batch])).unsqueeze(1).to(device)
    s2_b = torch.FloatTensor(np.array([b[3] for b in batch])).to(device)
    done_b = torch.FloatTensor(np.array([b[4] for b in batch])).unsqueeze(1).to(device)

    with torch.no_grad():
        a2 = actor_target(s2_b)
        q2 = critic_target(s2_b, a2)
        y = r_b + GAMMA * (1 - done_b) * q2

    q = critic(s_b, a_b)
    critic_loss = mse_loss(q, y)
    critic_opt.zero_grad()
    critic_loss.backward()
    critic_opt.step()

    a_pred = actor(s_b)
    actor_loss = -critic(s_b, a_pred).mean()
    actor_opt.zero_grad()
    actor_loss.backward()
    actor_opt.step()

    soft_update(critic, critic_target, TAU)
    soft_update(actor, actor_target, TAU)

# ----------------------------
# 3) Training loop
# ----------------------------
EPISODES = 800    # can increase for better performance
REPORT_EVERY = 50

episode_rewards = []
episode_infos = []  # store per-episode traces for plotting later

start_time = time.time()
for ep in range(EPISODES):
    # initialize episode
    # initial param guess (mid-range)
    init_params = ((SPINDLE_MIN+SPINDLE_MAX)/2,
                   (FEED_MIN+FEED_MAX)/2,
                   (DEPTH_MIN+DEPTH_MAX)/2,
                   0.0)
    tool_wear = 0.0
    last_surface = 0.0
    params = init_params

    state_obs = normalize_state(params, tool_wear, last_surface)
    total_reward = 0.0

    segment_records = []  # record (params, surface_error, wear, time) per segment

    noise_scale = max(0.4 * (1 - ep / EPISODES), 0.05)

    for seg in range(SEGMENTS):
        # actor chooses action based on current state
        a = select_action(state_obs, noise_scale)
        new_params = action_to_params(a)
        seg_time, surface_err, new_wear, fail = simulate_segment(params, new_params, tool_wear)

        # Normalize metrics for reward composition
        # Simple normalizations (choose scales)
        norm_time = seg_time / 10.0      # divide by 10 sec scale
        norm_err = surface_err / 50.0    # microns / 50
        norm_wear = new_wear / TOOL_WEAR_THRESHOLD

        # instantaneous reward: small negative cost (we want minimize time/error/wear)
        inst_reward = - (W_TIME * norm_time + W_ERROR * norm_err + W_WEAR * norm_wear)

        # big penalty if fail
        done = False
        if fail:
            inst_reward -= 5.0
            done = True

        # if surface_err too big, mark done and penalty
        if surface_err > SURFACE_ERROR_LIMIT:
            inst_reward -= 3.0
            done = True

        # accumulate
        total_reward += inst_reward

        # push to replay buffer (state, action, reward, next_state, done)
        next_state_obs = normalize_state(new_params, new_wear, surface_err)
        replay.append((state_obs.copy(), a.copy(), inst_reward, next_state_obs.copy(), float(done)))

        # train
        train_from_replay()

        # record
        segment_records.append({
            'params': new_params,
            'surface_error': surface_err,
            'wear': new_wear,
            'time': seg_time
        })

        # update for next step
        params = new_params
        tool_wear = new_wear
        last_surface = surface_err
        state_obs = next_state_obs

        if done:
            break

    # episode-level bonus if finished all segments and final surface acceptable
    final_ok = (not fail) and (last_surface <= SURFACE_ERROR_LIMIT)
    if final_ok:
        total_reward += 5.0  # bonus for successful completion

    episode_rewards.append(total_reward)
    episode_infos.append({
        'records': segment_records,
        'total_reward': total_reward,
        'final_surface': last_surface,
        'final_wear': tool_wear
    })

    if (ep + 1) % REPORT_EVERY == 0 or ep == 0:
        avg_recent = np.mean(episode_rewards[-REPORT_EVERY:])
        print(f"Ep {ep+1}/{EPISODES}  tot_reward={total_reward:.2f}  avg_recent={avg_recent:.2f}  noise={noise_scale:.3f}")

elapsed = time.time() - start_time
print(f"Training finished in {elapsed:.1f}s")

# ----------------------------
# 4) Plot training rewards and analyze best episode
# ----------------------------
plt.figure(figsize=(10,4))
plt.plot(episode_rewards, label='Episode reward')
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('Training Rewards')
plt.grid(True)
plt.legend()
plt.show()

# find best episode
best_idx = int(np.argmax(episode_rewards))
print(f"Best episode index: {best_idx}, reward={episode_rewards[best_idx]:.2f}")
best_info = episode_infos[best_idx]
records = best_info['records']

# Plot surface error over segments (terminal curve)
surf_errors = [r['surface_error'] for r in records]
wears = [r['wear'] for r in records]
times = [r['time'] for r in records]
params_seq = [r['params'] for r in records]

plt.figure(figsize=(10,4))
plt.plot(surf_errors, marker='o', label='Surface error (microns)')
plt.axhline(SURFACE_ERROR_LIMIT, color='r', linestyle='--', label='Error limit')
plt.xlabel('Segment')
plt.ylabel('Surface error (microns)')
plt.title('Best Episode: Surface Error per Segment (Terminal Curve)')
plt.legend()
plt.grid(True)
plt.show()

# Plot tool wear evolution
plt.figure(figsize=(10,3))
plt.plot(wears, marker='s', label='Tool wear')
plt.xlabel('Segment')
plt.ylabel('Tool wear (arb)')
plt.title('Tool Wear Evolution (Best Episode)')
plt.grid(True)
plt.legend()
plt.show()

# Print best episode parameter sequence
print("Best episode parameter sequence (spindle, feed, depth, angle) per segment:")
for i, p in enumerate(params_seq):
    s,f,d,ang = p
    print(f"seg {i:02d}: spindle={s:.0f} rpm, feed={f:.1f} mm/min, depth={d:.3f} mm, angle={ang:.1f} deg")

# Show final metrics
print(f"\nBest episode final surface error: {best_info['final_surface']:.2f} microns")
print(f"Best episode final tool wear: {best_info['final_wear']:.4f} (arb)")

# Optional: visualize a 2D "policy rollout" path of chosen feed/depth across segments for best episode
feeds = [p[1] for p in params_seq]
depths = [p[2] for p in params_seq]
plt.figure(figsize=(6,4))
plt.plot(feeds, label='feed (mm/min)')
plt.plot(depths, label='depth (mm)')
plt.xlabel('Segment')
plt.legend()
plt.title('Feed and Depth over segments (Best Episode)')
plt.grid(True)
plt.show()
