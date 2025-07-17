# ===========================================================
#  主題 : 工廠工件排成最佳化設計 以DQN 估測
#  目標 : 使學員了解以DQN 方法與概念
#  產生  20 各工件 四台機器 每台預設加工時間 進行Q_Learning 估測  找到最佳解
#  作者 : 國立雲林科技大學電機系 林家仁
# ============================================================


import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import matplotlib.pyplot as plt

# 固定隨機種子，確保可重現
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# 問題設定：20工件，4台機器，工件加工時間（jobs x machines）
job_times = [
    [3, 5, 4, 6],  [2, 4, 3, 7],  [4, 2, 5, 3],  [3, 3, 6, 4],  [5, 4, 3, 2],
    [6, 5, 4, 3],  [7, 3, 5, 6],  [2, 6, 4, 5],  [4, 3, 7, 2],  [5, 4, 6, 3],
    [6, 2, 3, 5],  [4, 7, 2, 4],  [3, 5, 6, 3],  [5, 6, 4, 2],  [7, 4, 5, 3],
    [3, 3, 4, 6],  [4, 5, 3, 7],  [6, 2, 7, 4],  [5, 4, 6, 3],  [4, 3, 5, 6],
]

NUM_JOBS = len(job_times)
NUM_MACHINES = len(job_times[0])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 計算 makespan：最大機器累積時間
def makespan(schedule):
    times = [0] * NUM_MACHINES
    for job_idx, machine_id in enumerate(schedule):
        times[machine_id] += job_times[job_idx][machine_id]
    return max(times)

# DQN 神經網路
class DQNNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQNNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Replay Buffer
class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (torch.stack(states), torch.tensor(actions),
                torch.tensor(rewards), torch.stack(next_states), torch.tensor(dones))

    def __len__(self):
        return len(self.buffer)

# 狀態維度 = 工件 one-hot (20) + 機器負載時間 (4)
STATE_DIM = NUM_JOBS + NUM_MACHINES
ACTION_DIM = NUM_MACHINES

policy_net = DQNNet(STATE_DIM, ACTION_DIM).to(device)
target_net = DQNNet(STATE_DIM, ACTION_DIM).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters(), lr=1e-3)
criterion = nn.MSELoss()
buffer = ReplayBuffer(10000)

BATCH_SIZE = 64
GAMMA = 0.99
EPS_START = 1.0
EPS_END = 0.1
EPS_DECAY = 0.995
TARGET_UPDATE = 10
NUM_EPISODES = 700

def state_to_tensor(step, machine_loads):
    vec = torch.zeros(STATE_DIM, dtype=torch.float32)
    vec[step] = 1.0  # 工件one-hot
    vec[NUM_JOBS:NUM_JOBS+NUM_MACHINES] = torch.tensor(machine_loads, dtype=torch.float32)
    return vec.to(device)

def select_action(state, epsilon):
    if random.random() < epsilon:
        return random.randint(0, ACTION_DIM - 1)
    else:
        with torch.no_grad():
            q_values = policy_net(state.unsqueeze(0))
            return q_values.argmax().item()

def optimize_model():
    if len(buffer) < BATCH_SIZE:
        return
    states, actions, rewards, next_states, dones = buffer.sample(BATCH_SIZE)
    states = states.to(device)
    actions = actions.to(device)
    rewards = rewards.to(device)
    next_states = next_states.to(device)
    dones = dones.to(device)

    q_values = policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
    next_q_values = target_net(next_states).max(1)[0]
    expected_q_values = rewards + GAMMA * next_q_values * (1 - dones)

    loss = criterion(q_values, expected_q_values.detach())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# 訓練主迴圈
epsilon = EPS_START
episode_rewards = []

for episode in range(NUM_EPISODES):
    machine_loads = [0] * NUM_MACHINES
    schedule = []
    total_reward = 0
    for step in range(NUM_JOBS):
        state = state_to_tensor(step, machine_loads)
        action = select_action(state, epsilon)
        schedule.append(action)

        duration = job_times[step][action]
        machine_loads[action] += duration
        cost = max(machine_loads)
        reward = -cost
        total_reward += reward

        next_state = state_to_tensor(step + 1 if step + 1 < NUM_JOBS else step, machine_loads)
        done = 1 if step == NUM_JOBS - 1 else 0

        buffer.push(state, action, reward, next_state, done)
        optimize_model()

    epsilon = max(EPS_END, epsilon * EPS_DECAY)
    episode_rewards.append(total_reward)

    if episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())

    if (episode + 1) % 50 == 0:
        print(f"Episode {episode + 1}, total reward: {total_reward:.2f}, epsilon: {epsilon:.2f}")

# 測試學習結果
machine_loads = [0] * NUM_MACHINES
schedule = []
for step in range(NUM_JOBS):
    state = state_to_tensor(step, machine_loads)
    with torch.no_grad():
        action = policy_net(state.unsqueeze(0)).argmax().item()
    schedule.append(action)
    machine_loads[action] += job_times[step][action]

print("\nBest schedule (Job -> Machine):")
for job_idx, machine_id in enumerate(schedule):
    print(f"Job {job_idx} -> Machine {machine_id + 1}")

print(f"Total makespan: {makespan(schedule)}")

# 繪製甘特圖
machine_times = [[] for _ in range(NUM_MACHINES)]
current_time = [0] * NUM_MACHINES

for job_idx, machine_id in enumerate(schedule):
    start = current_time[machine_id]
    duration = job_times[job_idx][machine_id]
    end = start + duration
    machine_times[machine_id].append((start, end, job_idx))
    current_time[machine_id] = end

colors = plt.cm.tab20.colors
fig, ax = plt.subplots(figsize=(12, 6))

for m_id, jobs in enumerate(machine_times):
    for start, end, job_idx in jobs:
        ax.barh(y=m_id, width=end - start, left=start, height=0.5, color=colors[job_idx % 20])
        ax.text(start + (end - start) / 2, m_id, f"Job {job_idx}", va='center', ha='center', color='white', fontsize=8)

ax.set_yticks(range(NUM_MACHINES))
ax.set_yticklabels([f"Machine {i+1}" for i in range(NUM_MACHINES)])
ax.set_xlabel("Time")
ax.set_title("Factory Scheduling Gantt Chart (DQN with machine load state)")

plt.tight_layout()
plt.show()
