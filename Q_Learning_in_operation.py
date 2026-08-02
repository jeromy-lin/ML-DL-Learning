# ===========================================================
# 主題：工廠工件排程最佳化－Q-Learning 教學範例
#
# 目標：
# 1. 建立 20 個工件與 4 台異質平行機器
# 2. 每個工件在不同機器上的加工時間不同
# 3. 使用 Q-Learning 學習工件應分派至哪一台機器
# 4. 使最大完工時間 Makespan 逐漸降低
#
# State  ：目前工件編號 + 四台機器目前的忙碌程度
# Action ：選擇一台機器加工目前工件
# Reward ：本次分派造成的 Makespan 增加量取負值
#
# 作者：國立雲林科技大學電機工程系 林家仁
# ===========================================================


# ===========================================================
# 1. 載入套件
# ===========================================================

import random
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from IPython.display import display


# ===========================================================
# 2. 固定亂數種子
#
# 讓每次執行程式時，可以得到相近且可重現的結果
# ===========================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)


# ===========================================================
# 3. 建立工件加工時間表
#
# 每一列代表一個工件
# 每一欄代表一台機器
#
# 例如：
# Job 0 在 Machine 1 的加工時間為 3
# Job 0 在 Machine 2 的加工時間為 5
# ===========================================================

job_times = np.array([
    [3, 5, 4, 6],  # Job 0
    [2, 4, 3, 7],  # Job 1
    [4, 2, 5, 3],  # Job 2
    [3, 3, 6, 4],  # Job 3
    [5, 4, 3, 2],  # Job 4
    [6, 5, 4, 3],  # Job 5
    [7, 3, 5, 6],  # Job 6
    [2, 6, 4, 5],  # Job 7
    [4, 3, 7, 2],  # Job 8
    [5, 4, 6, 3],  # Job 9
    [6, 2, 3, 5],  # Job 10
    [4, 7, 2, 4],  # Job 11
    [3, 5, 6, 3],  # Job 12
    [5, 6, 4, 2],  # Job 13
    [7, 4, 5, 3],  # Job 14
    [3, 3, 4, 6],  # Job 15
    [4, 5, 3, 7],  # Job 16
    [6, 2, 7, 4],  # Job 17
    [5, 4, 6, 3],  # Job 18
    [4, 3, 5, 6],  # Job 19
], dtype=int)


NUM_JOBS = job_times.shape[0]
NUM_MACHINES = job_times.shape[1]


# ===========================================================
# 4. 顯示工件加工時間表
# ===========================================================

job_time_df = pd.DataFrame(
    job_times,
    index=[f"Job {i}" for i in range(NUM_JOBS)],
    columns=[f"Machine {i + 1}" for i in range(NUM_MACHINES)]
)

print("工件在不同機器上的加工時間")
display(job_time_df)


# ===========================================================
# 5. Q-Learning 參數
# ===========================================================

ALPHA = 0.15
# 學習率：
# 控制每次新經驗對 Q 值的影響程度

GAMMA = 1.0
# 折扣因子：
# 本案例重視完整 20 個工件最後的 Makespan，
# 因此設定為 1.0，不降低未來獎勵的重要性

EPSILON_START = 1.0
# 訓練初期探索率為 100%

EPSILON_END = 0.05
# 訓練後期仍保留 5% 隨機探索

EPSILON_DECAY_RATIO = 0.85
# 在前 85% 的訓練過程中，
# 探索率由 1.0 逐漸下降至 0.05

EPISODES = 30000
# 一個 Episode 代表完整安排 20 個工件一次


# ===========================================================
# 6. 機器忙碌程度的簡化方式
#
# 為避免 Q-table 記錄過多非常接近的狀態，
# 將機器負載每 3 個時間單位視為一個忙碌等級。
#
# 實際負載 0～2  → 忙碌等級 0
# 實際負載 3～5  → 忙碌等級 1
# 實際負載 6～8  → 忙碌等級 2
# 實際負載 9～11 → 忙碌等級 3
#
# 這不是改變實際加工時間，
# 只是讓 Agent 較容易辨識「大概有多忙」。
# ===========================================================

LOAD_LEVEL_SIZE = 3


# ===========================================================
# 7. 建立稀疏 Q-table
#
# 每一個 State 對應四個 Q 值，
# 分別表示選擇四台機器的預期回報。
#
# 使用 dictionary，只記錄訓練中真正遇到的狀態。
# ===========================================================

Q = defaultdict(
    lambda: np.zeros(NUM_MACHINES, dtype=float)
)


# ===========================================================
# 8. 狀態編碼函數
#
# 狀態包含：
#
# 1. 現在正在安排第幾個工件
# 2. Machine 1 的忙碌程度
# 3. Machine 2 的忙碌程度
# 4. Machine 3 的忙碌程度
# 5. Machine 4 的忙碌程度
#
# 例如：
#
# state = (5, 2, 3, 1, 4)
#
# 代表：
# 現在正在安排 Job 5，
# 四台機器的忙碌程度分別為 2、3、1、4。
# ===========================================================

def get_state(job_index, machine_loads):
    """
    將目前工件與四台機器負載轉換成 Q-table 的狀態。
    """

    load_levels = tuple(
        (machine_loads // LOAD_LEVEL_SIZE)
        .astype(int)
        .tolist()
    )

    return (job_index,) + load_levels


# ===========================================================
# 9. 從相同最大 Q 值中隨機選擇
#
# 一般 np.argmax() 遇到相同最大值時，
# 永遠會選擇第一台機器。
#
# 此函數可避免訓練初期過度偏向 Machine 1。
# ===========================================================

def choose_best_action(q_values):
    """
    從所有最大 Q 值的行動中隨機選擇一個。
    """

    maximum_q = np.max(q_values)

    candidate_actions = np.flatnonzero(
        np.isclose(q_values, maximum_q)
    )

    return int(np.random.choice(candidate_actions))


# ===========================================================
# 10. 探索率下降函數
#
# 訓練初期：
# 多嘗試不同機器
#
# 訓練後期：
# 多使用 Q-table 中學到的經驗
# ===========================================================

def get_epsilon(episode):
    """
    計算目前 Episode 的探索率。
    """

    decay_episodes = max(
        1,
        int(EPISODES * EPSILON_DECAY_RATIO)
    )

    progress = min(
        episode / decay_episodes,
        1.0
    )

    epsilon = (
        EPSILON_START
        + progress
        * (EPSILON_END - EPSILON_START)
    )

    return epsilon


# ===========================================================
# 11. 計算排程結果
#
# schedule[job] = machine
#
# 例如：
# schedule[0] = 2
#
# 代表 Job 0 分派給程式編號 2 的機器，
# 也就是 Machine 3。
# ===========================================================

def calculate_schedule(schedule):
    """
    計算排程的各機器負載與 Makespan。
    """

    machine_loads = np.zeros(
        NUM_MACHINES,
        dtype=int
    )

    for job_index, machine_index in enumerate(schedule):

        machine_loads[machine_index] += (
            job_times[job_index, machine_index]
        )

    makespan = int(np.max(machine_loads))

    return makespan, machine_loads


# ===========================================================
# 12. Q-Learning 訓練前的紀錄變數
# ===========================================================

makespan_history = []
# 記錄每一個 Episode 的 Makespan

best_history = []
# 記錄截至目前為止的最佳 Makespan

epsilon_history = []
# 記錄每一個 Episode 的探索率

best_makespan = float("inf")
best_schedule = None
best_episode = None

initial_schedule = None
initial_makespan = None


# ===========================================================
# 13. Q-Learning 訓練
# ===========================================================

for episode in range(EPISODES):

    # 每一回合都從四台空機器開始
    machine_loads = np.zeros(
        NUM_MACHINES,
        dtype=int
    )

    schedule = []

    epsilon = get_epsilon(episode)

    # -------------------------------------------------------
    # 依序安排 20 個工件
    # -------------------------------------------------------

    for job_index in range(NUM_JOBS):

        # 1. 觀察目前狀態
        state = get_state(
            job_index,
            machine_loads
        )

        # ---------------------------------------------------
        # 2. 使用 ε-greedy 選擇機器
        # ---------------------------------------------------

        if random.random() < epsilon:

            # 探索：
            # 隨機選擇一台機器
            action = random.randrange(
                NUM_MACHINES
            )

        else:

            # 利用：
            # 選擇目前 Q 值最高的機器
            action = choose_best_action(
                Q[state]
            )

        # ---------------------------------------------------
        # 3. 記錄分派前的 Makespan
        # ---------------------------------------------------

        previous_makespan = int(
            np.max(machine_loads)
        )

        # ---------------------------------------------------
        # 4. 將目前工件分派給所選機器
        # ---------------------------------------------------

        processing_time = job_times[
            job_index,
            action
        ]

        machine_loads[action] += processing_time

        # ---------------------------------------------------
        # 5. 計算分派後的 Makespan
        # ---------------------------------------------------

        current_makespan = int(
            np.max(machine_loads)
        )

        # ---------------------------------------------------
        # 6. 計算 Reward
        #
        # Makespan 沒增加：
        # reward = 0
        #
        # Makespan 增加 3：
        # reward = -3
        #
        # 因此 Agent 會偏好 Makespan 增加較少的選擇。
        # ---------------------------------------------------

        reward = -(
            current_makespan
            - previous_makespan
        )

        # ---------------------------------------------------
        # 7. 計算 Q-Learning 更新目標
        # ---------------------------------------------------

        if job_index == NUM_JOBS - 1:

            # 最後一個工件沒有下一個狀態
            target = reward

        else:

            next_state = get_state(
                job_index + 1,
                machine_loads
            )

            maximum_next_q = np.max(
                Q[next_state]
            )

            target = (
                reward
                + GAMMA * maximum_next_q
            )

        # ---------------------------------------------------
        # 8. 更新 Q 值
        #
        # Q(s,a) ← Q(s,a)
        #          + α[target - Q(s,a)]
        # ---------------------------------------------------

        Q[state][action] += ALPHA * (
            target
            - Q[state][action]
        )

        # 記錄目前工件選擇的機器
        schedule.append(action)

    # -------------------------------------------------------
    # 9. 完成 20 個工件後，計算本回合 Makespan
    # -------------------------------------------------------

    episode_makespan = int(
        np.max(machine_loads)
    )

    # 記錄第一回合結果
    if episode == 0:

        initial_schedule = schedule.copy()
        initial_makespan = episode_makespan

    # -------------------------------------------------------
    # 10. 若本回合更好，記錄為目前最佳方案
    # -------------------------------------------------------

    if episode_makespan < best_makespan:

        best_makespan = episode_makespan
        best_schedule = schedule.copy()
        best_episode = episode + 1

    makespan_history.append(
        episode_makespan
    )

    best_history.append(
        int(best_makespan)
    )

    epsilon_history.append(
        epsilon
    )


# ===========================================================
# 14. 計算第一回合與最佳回合的機器負載
# ===========================================================

initial_makespan, initial_machine_loads = (
    calculate_schedule(initial_schedule)
)

best_makespan, best_machine_loads = (
    calculate_schedule(best_schedule)
)


# ===========================================================
# 15. 計算訓練初期與後期平均效果
# ===========================================================

COMPARE_EPISODES = min(
    500,
    EPISODES
)

early_average = np.mean(
    makespan_history[:COMPARE_EPISODES]
)

late_average = np.mean(
    makespan_history[-COMPARE_EPISODES:]
)

improvement_rate = (
    (initial_makespan - best_makespan)
    / initial_makespan
    * 100
)


# ===========================================================
# 16. 顯示 Q-Learning 訓練摘要
# ===========================================================

summary_df = pd.DataFrame({
    "項目": [
        "訓練回合數",
        "第一回合 Makespan",
        f"前 {COMPARE_EPISODES} 回合平均 Makespan",
        f"後 {COMPARE_EPISODES} 回合平均 Makespan",
        "Q-Learning 最佳 Makespan",
        "最佳結果出現回合",
        "第一回合至最佳結果改善率",
        "Q-table 已記錄狀態數"
    ],

    "結果": [
        EPISODES,
        initial_makespan,
        round(early_average, 2),
        round(late_average, 2),
        best_makespan,
        best_episode,
        f"{improvement_rate:.2f}%",
        len(Q)
    ]
})


print("=" * 60)
print("Q-Learning 訓練結果摘要")
print("=" * 60)

display(summary_df)


# ===========================================================
# 17. 建立最佳排程明細表
# ===========================================================

def build_schedule_table(schedule):
    """
    將排程轉換成工件、機器、開始與完成時間表。
    """

    current_time = np.zeros(
        NUM_MACHINES,
        dtype=int
    )

    records = []

    for job_index, machine_index in enumerate(schedule):

        start_time = int(
            current_time[machine_index]
        )

        processing_time = int(
            job_times[job_index, machine_index]
        )

        finish_time = (
            start_time + processing_time
        )

        records.append({
            "工件": f"Job {job_index}",
            "指派機器": f"Machine {machine_index + 1}",
            "加工時間": processing_time,
            "開始時間": start_time,
            "完成時間": finish_time
        })

        current_time[machine_index] = (
            finish_time
        )

    return pd.DataFrame(records)


best_schedule_df = build_schedule_table(
    best_schedule
)


print("Q-Learning 找到的最佳排程明細")

display(best_schedule_df)


# ===========================================================
# 18. 顯示第一回合與最佳回合的機器負載
# ===========================================================

machine_load_df = pd.DataFrame({
    "機器": [
        f"Machine {i + 1}"
        for i in range(NUM_MACHINES)
    ],

    "第一回合負載": initial_machine_loads,

    "最佳回合負載": best_machine_loads
})


print("Q-Learning 學習前後的機器負載")

display(machine_load_df)


# ===========================================================
# 19. 圖一：Q-Learning 收斂曲線
# ===========================================================

MOVING_WINDOW = 100

moving_average = np.convolve(
    makespan_history,
    np.ones(MOVING_WINDOW) / MOVING_WINDOW,
    mode="valid"
)

moving_x = np.arange(
    MOVING_WINDOW,
    EPISODES + 1
)


plt.figure(figsize=(12, 5))

plt.plot(
    makespan_history,
    alpha=0.20,
    label="Episode Makespan"
)

plt.plot(
    moving_x,
    moving_average,
    linewidth=2,
    label=f"Moving Average ({MOVING_WINDOW})"
)

plt.plot(
    best_history,
    linewidth=2,
    label="Best-so-far Makespan"
)

plt.xlabel("Episode")
plt.ylabel("Makespan")
plt.title("Q-Learning Scheduling Convergence")

plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()


# ===========================================================
# 20. 圖二：Epsilon 探索率曲線
# ===========================================================

plt.figure(figsize=(12, 4))

plt.plot(epsilon_history)

plt.xlabel("Episode")
plt.ylabel("Epsilon")
plt.title("Q-Learning Exploration Rate")

plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()


# ===========================================================
# 21. 圖三：Q-Learning 學習效果比較
#
# 此圖只比較 Q-Learning 自己不同訓練階段，
# 不加入其他最佳化方法。
# ===========================================================

learning_labels = [
    "First Episode",
    f"Early Average\n({COMPARE_EPISODES})",
    f"Late Average\n({COMPARE_EPISODES})",
    "Best Result"
]

learning_values = [
    initial_makespan,
    early_average,
    late_average,
    best_makespan
]


plt.figure(figsize=(10, 5))

bars = plt.bar(
    learning_labels,
    learning_values
)

for bar, value in zip(bars, learning_values):

    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.3,
        f"{value:.1f}",
        ha="center",
        va="bottom"
    )

plt.ylabel("Makespan")
plt.title("Q-Learning Learning Improvement")

plt.grid(
    axis="y",
    alpha=0.3
)

plt.tight_layout()
plt.show()


# ===========================================================
# 22. 圖四：第一回合與最佳回合機器負載比較
# ===========================================================

machine_names = [
    f"Machine {i + 1}"
    for i in range(NUM_MACHINES)
]

x_positions = np.arange(
    NUM_MACHINES
)

bar_width = 0.35


plt.figure(figsize=(10, 5))

plt.bar(
    x_positions - bar_width / 2,
    initial_machine_loads,
    width=bar_width,
    label="First Episode"
)

plt.bar(
    x_positions + bar_width / 2,
    best_machine_loads,
    width=bar_width,
    label="Best Q-Learning Result"
)

plt.xticks(
    x_positions,
    machine_names
)

plt.xlabel("Machine")
plt.ylabel("Total Processing Time")
plt.title("Machine Load Before and After Learning")

plt.legend()
plt.grid(
    axis="y",
    alpha=0.3
)

plt.tight_layout()
plt.show()


# ===========================================================
# 23. 甘特圖繪製函數
# ===========================================================

def draw_gantt_chart(schedule, title):
    """
    繪製指定排程的甘特圖。
    """

    machine_jobs = [
        []
        for _ in range(NUM_MACHINES)
    ]

    current_time = np.zeros(
        NUM_MACHINES,
        dtype=int
    )

    # 建立每台機器上的工件開始與完成時間
    for job_index, machine_index in enumerate(schedule):

        start_time = int(
            current_time[machine_index]
        )

        processing_time = int(
            job_times[job_index, machine_index]
        )

        finish_time = (
            start_time + processing_time
        )

        machine_jobs[machine_index].append(
            (
                start_time,
                finish_time,
                job_index
            )
        )

        current_time[machine_index] = (
            finish_time
        )

    makespan = int(
        np.max(current_time)
    )

    figure, axis = plt.subplots(
        figsize=(14, 6)
    )

    job_colors = plt.cm.tab20.colors

    for machine_index, jobs in enumerate(machine_jobs):

        for start_time, finish_time, job_index in jobs:

            duration = (
                finish_time - start_time
            )

            axis.barh(
                y=machine_index,
                width=duration,
                left=start_time,
                height=0.55,
                color=job_colors[
                    job_index % len(job_colors)
                ],
                edgecolor="black"
            )

            axis.text(
                start_time + duration / 2,
                machine_index,
                f"J{job_index}",
                va="center",
                ha="center",
                fontsize=8
            )

    axis.set_yticks(
        range(NUM_MACHINES)
    )

    axis.set_yticklabels([
        f"Machine {i + 1}"
        for i in range(NUM_MACHINES)
    ])

    axis.set_xlabel("Time")
    axis.set_ylabel("Machine")

    axis.set_title(
        f"{title} | Makespan = {makespan}"
    )

    axis.grid(
        axis="x",
        alpha=0.3
    )

    plt.tight_layout()
    plt.show()


# ===========================================================
# 24. 圖五：第一回合甘特圖
#
# 第一回合探索率很高，
# 主要用來呈現尚未學習時的排程結果。
# ===========================================================

draw_gantt_chart(
    initial_schedule,
    "Gantt Chart of First Q-Learning Episode"
)


# ===========================================================
# 25. 圖六：Q-Learning 最佳排程甘特圖
# ===========================================================

draw_gantt_chart(
    best_schedule,
    "Gantt Chart of Best Q-Learning Schedule"
)


# ===========================================================
# 26. 最後文字輸出
# ===========================================================

print("=" * 60)
print("Q-Learning 最終結果")
print("=" * 60)

print(f"第一回合 Makespan：{initial_makespan}")
print(f"最佳 Makespan：{best_makespan}")
print(f"最佳結果出現在第 {best_episode} 回合")
print(f"改善率：{improvement_rate:.2f}%")

print("\n最佳工件分派結果：")

for job_index, machine_index in enumerate(best_schedule):

    print(
        f"Job {job_index:2d}"
        f" → Machine {machine_index + 1}"
        f"｜加工時間：{job_times[job_index, machine_index]}"
    )
