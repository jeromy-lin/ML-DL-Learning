# ===========================================================
#  主題 : DDPG 控制系統範例 : MountainCarContinuous
#   特點: 5k~10k steps + Self Defined Reward Function + 小型化DEMO網路
#  小車要靠來回加速，最後爬上右側山頂
#  目標 : 使學員了解  DDPG方法與概念
#  作者 : 國立雲林科技大學電機系 林家仁
# ============================================================

!pip install --upgrade gymnasium stable-baselines3[extra] --quiet

import warnings
warnings.filterwarnings('ignore')

import gymnasium as gym
import numpy as np
from stable_baselines3 import DDPG
from stable_baselines3.common.noise import NormalActionNoise
from gymnasium.wrappers import RecordVideo
import glob
from IPython.display import HTML
from base64 import b64encode

# -------------------------------
# Reward Fnction
# -------------------------------
class RewardWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.x_goal = env.unwrapped.goal_position

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        x_t = obs[0]
        a_t = action[0]

        progress_reward = x_t - (-0.5)
        action_penalty = 0.01 * (a_t**2)
        bonus = 100 if x_t >= self.x_goal else 0

        reward = progress_reward - action_penalty + bonus
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

# -------------------------------
# 建立環境
# -------------------------------
env = gym.make("MountainCarContinuous-v0", render_mode="rgb_array")
env = RewardWrapper(env)

# -------------------------------
# 建立 DDPG Agent (小網路)
# -------------------------------
n_actions = env.action_space.shape[0]
action_noise = NormalActionNoise(mean=np.zeros(n_actions),
                                 sigma=0.1*np.ones(n_actions))

model = DDPG(
    "MlpPolicy",
    env,
    action_noise=action_noise,
    verbose=1,
    policy_kwargs={'net_arch':[32,32]}  # 適用於學員CPU 網路
)

# -------------------------------
# 訓練 (Demo 5k steps)
# -------------------------------
model.learn(total_timesteps=5000)

# -------------------------------
# 錄影測試
# -------------------------------
video_dir = "./videos"
test_env = RecordVideo(gym.make("MountainCarContinuous-v0", render_mode="rgb_array"),
                       video_folder=video_dir, episode_trigger=lambda e: True)

obs, _ = test_env.reset()
done = False
while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, _ = test_env.step(action)
    done = terminated or truncated
test_env.close()

# -------------------------------
# 播放影片
# -------------------------------
video_files = glob.glob(video_dir + "/*.mp4")
if video_files:
    mp4 = video_files[-1]
    mp4_ = open(mp4,'rb').read()
    data_url = "data:video/mp4;base64," + b64encode(mp4_).decode()
    display(HTML(f'<video width=400 controls><source src="{data_url}" type="video/mp4"></video>'))
