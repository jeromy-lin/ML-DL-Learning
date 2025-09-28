# ===========================================================
#  主題 : OpenGym RL Pendulum-v1 : 經典倒立擺環境範例展示
#  目標 : 使學員了解以GYM 方法與概念, 以DDPG為例
#  於 COLAB上自動產生Video影片可觀察結果, 避免安裝複雜環境設定
#  作者 : 國立雲林科技大學電機系 林家仁
# ============================================================

# -------------------------------
# Step 1：安裝必要套件
# -------------------------------
!pip install --upgrade gymnasium stable-baselines3[extra] --quiet
# gymnasium: 取代舊版 Gym，提供環境封裝
# stable-baselines3: 提供 DDPG、SAC 等 RL 演算法
# extra:  錄影、渲染等功能

# -------------------------------
# Step 2：匯入套件
# -------------------------------
import warnings
warnings.filterwarnings('ignore')  # 避免 Colab 顯示 DeprecationWarning

import gymnasium as gym  # Gymnasium 環境
import numpy as np
from stable_baselines3 import DDPG  # DDPG 演算法
from stable_baselines3.common.noise import NormalActionNoise  # 探索噪聲
from gymnasium.wrappers import RecordVideo  # 環境錄影
import glob
from IPython.display import HTML  # Colab 播放影片
from base64 import b64encode

# -------------------------------
# Step 3：建立倒立擺環境 (Pendulum)
# -------------------------------
env = gym.make("Pendulum-v1", render_mode="rgb_array")
# "Pendulum-v1": 經典倒立擺環境
# render_mode="rgb_array": 在 Colab 中可錄影播放
# 狀態(state): [角度, 角速度]
# 動作(action): 連續推力 [-2, 2]

# -------------------------------
# Step 4：建立 DDPG Agent
# -------------------------------
n_actions = env.action_space.shape[0]  # 取得動作維度 (Pendulum: 1 維)
action_noise = NormalActionNoise(
    mean=np.zeros(n_actions), sigma=0.1*np.ones(n_actions)
)  
# NormalActionNoise: 高斯噪聲，用於探索
# mean: 平均值 0，sigma: 標準差 0.1

# 建立 DDPG agent
model = DDPG(
    "MlpPolicy",  # 使用多層感知器策略 (MLP)
    env,          # 環境
    action_noise=action_noise,  # 加入探索噪聲
    verbose=1     # 顯示訓練進度
)

# -------------------------------
# Step 5：訓練 Agent
# -------------------------------
model.learn(total_timesteps=5000)  # 課堂 demo 使用 5000 steps
# 訓練過程:
# Actor: 學習輸出連續動作
# Critic: 評估 state+action 的 Q-value
# Noise: 使 agent 有探索能力避免陷入局部最優
# reward: 獎勵函數，越接近直立角度 reward 越高

# -------------------------------
# Step 6：錄影環境並測試 Agent
# -------------------------------
video_dir = "./videos"  # 錄影存放路徑
env = RecordVideo(
    env, video_folder=video_dir, episode_trigger=lambda e: True
)
# RecordVideo: 將每一回合錄影
# episode_trigger=lambda e: True 表示每回合都錄影

obs, _ = env.reset()  # 重置環境，取得初始觀測值
done = False

while not done:
    # 使用 agent 預測行動
    action, _ = model.predict(obs, deterministic=True)  
    # deterministic=True: 測試時不加雜訊

    # 執行動作
    obs, reward, terminated, truncated, _ = env.step(action)
    done = terminated or truncated  # 判斷回合是否結束

env.close()  # 關閉環境

# -------------------------------
# Step 7：在 Colab 播放影片
# -------------------------------
video_files = glob.glob(video_dir + "/*.mp4")  # 取得影片檔案列表
if video_files:
    mp4 = video_files[-1]  # 取最後錄製的影片
    mp4_ = open(mp4,'rb').read()
    data_url = "data:video/mp4;base64," + b64encode(mp4_).decode()
    # 在 Colab Notebook 中播放影片
    HTML(f'<video width=400 controls><source src="{data_url}" type="video/mp4"></video>')
