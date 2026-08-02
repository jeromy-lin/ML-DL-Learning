# ===========================================================
# 主題：異質平行機工件分派最佳化－Q-Learning 教學範例
#
# 目標：
# 1. 將 20 個工件分派至 4 台加工能力不同的機器
# 2. 每個工件在不同機器上具有不同加工時間
# 3. 透過 Q-Learning 搜尋較佳的工件分派方案
# 4. 最小化最大完工時間 Makespan
#
# State  ：目前工件編號 + 各機器目前累積負載
# Action ：選擇一台機器加工目前工件
# Reward ：本次分派所造成的 Makespan 增加量取負值
#
# 作者：國立雲林科技大學電機工程系 林家仁
# ===========================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

from collections import defaultdict


# ===========================================================
# 1. 固定亂數種子，確保每次執行結果可以重現
# ===========================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)


# ===========================================================
# 2. 工件在各機器上的加工時間
#
# 每一列代表一個工件
# 每一欄代表一台機器
#
# 例如：
# Job 0 在 Machine 1 加工時間為 3
# Job 0 在 Machine 2 加工時間為 5
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
# 3. Q-Learning 參數
# ===========================================================

ALPHA = 0.15
# 學習率：
# 數值越大，新的經驗對 Q 值的影響越大

GAMMA = 1.0
# 折扣因子：
# 本案例為固定 20 個步驟的有限期排程問題
# 設定 gamma = 1，可讓累積 Reward 對應 -Makespan

EPSILON_START = 1.0
# 訓練初期以探索為主

EPSILON_END = 0.05
# 訓練後期仍保留 5% 探索機率

EPSILON_DECAY_RATIO = 0.85
# 在前 85% 的 Episodes 中逐漸降低探索率

EPISODES = 30000
# 訓練回合數

LOAD_BIN = 3
# 機器負載離散化區間
#
# LOAD_BIN = 1：
#   保留完整機器負載，狀態較精確，但 Q-table 較大
#
# LOAD_BIN = 3：
#   每 3 個時間單位視為相同負載區間，
#   可降低狀態數量，適合教學與執行展示


# ===========================================================
# 4. 建立稀疏 Q-table
#
# 因為 State 包含：
# 工件編號 + 四台機器目前負載
#
# 狀態數量不像原本只有 21 個，因此改用 dictionary
# 只記錄實際走訪過的狀態
# ===========================================================

Q = defaultdict(
    lambda: np.zeros(NUM_MACHINES, dtype=float)
)


# ===========================================================
# 5. 狀態編碼
#
# State：
# (目前工件編號, M1負載, M2負載, M3負載, M4負載)
#
# 例如：
# (5, 2, 1, 3, 0)
#
# 表示目前正在安排 Job 5，
# 四台機器離散化後的負載分別為 2、1、3、0
# ===========================================================

def encode_state(job_index, machine_loads):
    """
    將目前工件編號與各機器負載轉換成 Q-table 的狀態。
    """

    load_bins = tuple(
        (np.asarray(machine_loads) // LOAD_BIN)
        .astype(int)
        .tolist()
    )

    return (job_index,) + load_bins


# ===========================================================
# 6. 隨機選擇最大 Q 值行動
#
# np.argmax() 遇到相同最大值時，
# 永遠選擇第一個位置，容易偏向 Machine 1。
#
# 此函數會在相同最大值的機器中隨機選擇。
# ===========================================================

def random_argmax(values):
    """
    在所有最大值的位置中隨機選擇一個。
    """

    maximum = np.max(values)

    candidates = np.flatnonzero(
        np.isclose(values, maximum)
    )

    return int(np.random.choice(candidates))


# ===========================================================
# 7. 計算排程 Makespan 與機器負載
# ===========================================================

def calculate_schedule(schedule):
    """
    根據工件分派結果，計算四台機器的負載與 Makespan。

    schedule[job] = machine
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
# 8. 探索率函數
#
# 探索率採線性下降：
# 訓練初期多探索，後期多利用已學到的 Q 值
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
# 9. Q-Learning 訓練
# ===========================================================

makespan_history = []
best_history = []
epsilon_history = []

best_makespan = float("inf")
best_schedule = None
best_episode = None


for episode in range(EPISODES):

    epsilon = get_epsilon(episode)

    # 每一回合重新開始安排 20 個工件
    machine_loads = np.zeros(
        NUM_MACHINES,
        dtype=int
    )

    schedule = []

    # -------------------------------------------------------
    # 依序分派每一個工件
    # -------------------------------------------------------

    for job_index in range(NUM_JOBS):

        # 觀察目前狀態
        state = encode_state(
            job_index,
            machine_loads
        )

        # ---------------------------------------------------
        # ε-greedy 行動選擇
        # ---------------------------------------------------

        if random.random() < epsilon:

            # 探索：
            # 隨機選擇一台機器
            action = random.randrange(
                NUM_MACHINES
            )

        else:

            # 利用：
            # 選擇目前 Q 值最大的機器
            action = random_argmax(
                Q[state]
            )

        # 記錄執行動作前的 Makespan
        previous_makespan = int(
            np.max(machine_loads)
        )

        # 將目前工件指派給所選機器
        processing_time = job_times[
            job_index,
            action
        ]

        machine_loads[action] += processing_time

        # 計算執行動作後的 Makespan
        current_makespan = int(
            np.max(machine_loads)
        )

        # ---------------------------------------------------
        # Reward 設計
        #
        # Makespan 增加越多，負獎勵越大
        #
        # 若 Makespan 沒有增加：
        # reward = 0
        #
        # 若 Makespan 增加 3：
        # reward = -3
        # ---------------------------------------------------

        reward = -(
            current_makespan
            - previous_makespan
        )

        # ---------------------------------------------------
        # 計算 Q-Learning 更新目標
        # ---------------------------------------------------

        if job_index == NUM_JOBS - 1:

            # 最後一個工件沒有下一個狀態
            target = reward

        else:

            next_state = encode_state(
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
        # 更新 Q 值
        #
        # Q(s,a) ← Q(s,a)
        #          + α[r + γ max Q(s',a') - Q(s,a)]
        # ---------------------------------------------------

        Q[state][action] += ALPHA * (
            target
            - Q[state][action]
        )

        schedule.append(action)

    # -------------------------------------------------------
    # 本 Episode 完成後，計算 Makespan
    # -------------------------------------------------------

    episode_makespan = int(
        np.max(machine_loads)
    )

    makespan_history.append(
        episode_makespan
    )

    epsilon_history.append(
        epsilon
    )

    # -------------------------------------------------------
    # 記錄訓練過程中實際出現的最佳排程
    # -------------------------------------------------------

    if episode_makespan < best_makespan:

        best_makespan = episode_makespan
        best_schedule = schedule.copy()
        best_episode = episode + 1

    best_history.append(
        best_makespan
    )


# ===========================================================
# 10. 根據訓練後的 Q-table 建立 Greedy Policy
#
# 注意：
# 「訓練期間最佳方案」與「最後 Q-table 形成的策略」
# 不一定完全相同，因此兩者分開計算。
# ===========================================================

def generate_q_policy_schedule():
    """
    使用訓練完成後的 Q-table 建立排程。
    """

    machine_loads = np.zeros(
        NUM_MACHINES,
        dtype=int
    )

    schedule = []

    for job_index in range(NUM_JOBS):

        state = encode_state(
            job_index,
            machine_loads
        )

        q_values = Q[state]

        maximum_q = np.max(q_values)

        candidates = np.flatnonzero(
            np.isclose(q_values, maximum_q)
        )

        # 若多台機器 Q 值相同，
        # 以分派後完工時間較小者優先
        projected_loads = (
            machine_loads[candidates]
            + job_times[job_index, candidates]
        )

        selected_position = np.argmin(
            projected_loads
        )

        action = int(
            candidates[selected_position]
        )

        schedule.append(action)

        machine_loads[action] += (
            job_times[job_index, action]
        )

    return schedule


policy_schedule = generate_q_policy_schedule()

policy_makespan, policy_loads = (
    calculate_schedule(policy_schedule)
)

best_makespan, best_loads = (
    calculate_schedule(best_schedule)
)


# ===========================================================
# 11. 建立傳統基準方法
# ===========================================================

def fastest_machine_method():
    """
    每一個工件都選擇加工時間最短的機器。
    不考慮機器目前是否已經很忙。
    """

    return np.argmin(
        job_times,
        axis=1
    ).tolist()


def list_scheduling_method():
    """
    每次選擇分派後完工時間最小的機器。
    """

    machine_loads = np.zeros(
        NUM_MACHINES,
        dtype=int
    )

    schedule = []

    for job_index in range(NUM_JOBS):

        projected_loads = (
            machine_loads
            + job_times[job_index]
        )

        machine = int(
            np.argmin(projected_loads)
        )

        schedule.append(machine)

        machine_loads[machine] += (
            job_times[job_index, machine]
        )

    return schedule


fastest_schedule = fastest_machine_method()

fastest_makespan, fastest_loads = (
    calculate_schedule(fastest_schedule)
)


list_schedule = list_scheduling_method()

list_makespan, list_loads = (
    calculate_schedule(list_schedule)
)


# ===========================================================
# 12. 隨機分派基準
# ===========================================================

RANDOM_TESTS = 1000

random_makespans = []

for _ in range(RANDOM_TESTS):

    random_schedule = np.random.randint(
        0,
        NUM_MACHINES,
        size=NUM_JOBS
    )

    random_makespan, _ = calculate_schedule(
        random_schedule
    )

    random_makespans.append(
        random_makespan
    )


random_average = np.mean(
    random_makespans
)

random_best = np.min(
    random_makespans
)


# ===========================================================
# 13. 使用 MILP 計算理論最佳解
#
# 此部分用來提供 Q-Learning 的比較基準。
# Q-Learning 不保證每次都找到數學全域最佳解。
# ===========================================================

optimal_schedule = None
optimal_makespan = None
optimal_loads = None


try:

    from scipy.optimize import (
        milp,
        LinearConstraint,
        Bounds
    )

    # -------------------------------------------------------
    # 決策變數：
    #
    # x[j,m] = 1：
    # Job j 分派至 Machine m
    #
    # 最後一個變數為 Cmax
    # -------------------------------------------------------

    number_x_variables = (
        NUM_JOBS * NUM_MACHINES
    )

    number_variables = (
        number_x_variables + 1
    )

    # 目標函數：
    # 最小化最後一個變數 Cmax
    objective = np.zeros(
        number_variables
    )

    objective[-1] = 1.0

    # x 為整數變數，Cmax 為連續變數
    integrality = np.ones(
        number_variables,
        dtype=int
    )

    integrality[-1] = 0

    # 變數上下界
    lower_bounds = np.zeros(
        number_variables
    )

    upper_bounds = np.ones(
        number_variables
    )

    upper_bounds[-1] = np.inf

    constraints = []

    # -------------------------------------------------------
    # 限制式一：
    # 每一個工件只能選擇一台機器
    # -------------------------------------------------------

    assignment_matrix = np.zeros(
        (
            NUM_JOBS,
            number_variables
        )
    )

    for job_index in range(NUM_JOBS):

        start = (
            job_index * NUM_MACHINES
        )

        end = (
            start + NUM_MACHINES
        )

        assignment_matrix[
            job_index,
            start:end
        ] = 1.0

    constraints.append(
        LinearConstraint(
            assignment_matrix,
            np.ones(NUM_JOBS),
            np.ones(NUM_JOBS)
        )
    )

    # -------------------------------------------------------
    # 限制式二：
    # 每台機器的總加工時間不得超過 Cmax
    # -------------------------------------------------------

    load_matrix = np.zeros(
        (
            NUM_MACHINES,
            number_variables
        )
    )

    for machine_index in range(NUM_MACHINES):

        for job_index in range(NUM_JOBS):

            variable_index = (
                job_index * NUM_MACHINES
                + machine_index
            )

            load_matrix[
                machine_index,
                variable_index
            ] = job_times[
                job_index,
                machine_index
            ]

        load_matrix[
            machine_index,
            -1
        ] = -1.0

    constraints.append(
        LinearConstraint(
            load_matrix,
            -np.inf * np.ones(NUM_MACHINES),
            np.zeros(NUM_MACHINES)
        )
    )

    result = milp(
        c=objective,
        integrality=integrality,
        bounds=Bounds(
            lower_bounds,
            upper_bounds
        ),
        constraints=constraints,
        options={
            "time_limit": 30
        }
    )

    if result.success:

        assignment_result = (
            result.x[:number_x_variables]
            .reshape(
                NUM_JOBS,
                NUM_MACHINES
            )
        )

        optimal_schedule = np.argmax(
            assignment_result,
            axis=1
        ).tolist()

        optimal_makespan, optimal_loads = (
            calculate_schedule(
                optimal_schedule
            )
        )

except Exception as error:

    print(
        "MILP benchmark could not be calculated:",
        error
    )


# ===========================================================
# 14. 顯示各方法比較結果
# ===========================================================

comparison_data = {
    "Method": [
        "Random Assignment Average",
        "Random Assignment Best",
        "Fastest Machine per Job",
        "List Scheduling",
        "Q-Learning Final Policy",
        "Q-Learning Best Observed"
    ],

    "Makespan": [
        round(random_average, 2),
        int(random_best),
        fastest_makespan,
        list_makespan,
        policy_makespan,
        best_makespan
    ]
}


if optimal_makespan is not None:

    comparison_data["Method"].append(
        "MILP Optimal Solution"
    )

    comparison_data["Makespan"].append(
        optimal_makespan
    )


comparison_df = pd.DataFrame(
    comparison_data
)


print("=" * 65)
print("Q-Learning Scheduling Results")
print("=" * 65)

print(
    f"Training Episodes        : {EPISODES}"
)

print(
    f"Visited Q-table States   : {len(Q)}"
)

print(
    f"Best Episode             : {best_episode}"
)

print(
    f"Best Observed Makespan   : {best_makespan}"
)

print(
    f"Final Policy Makespan    : {policy_makespan}"
)

if optimal_makespan is not None:

    print(
        f"MILP Optimal Makespan    : {optimal_makespan}"
    )

print("=" * 65)

display(comparison_df)


# ===========================================================
# 15. 顯示最佳工件分派結果
# ===========================================================

def build_schedule_table(schedule):
    """
    建立工件排程明細表。
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
            job_times[
                job_index,
                machine_index
            ]
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


print("\nQ-Learning Best Observed Schedule")

display(best_schedule_df)


# ===========================================================
# 16. 顯示各機器最終負載
# ===========================================================

machine_load_df = pd.DataFrame({
    "Machine": [
        f"Machine {i + 1}"
        for i in range(NUM_MACHINES)
    ],

    "Total Processing Time": best_loads
})


display(machine_load_df)


# ===========================================================
# 17. 繪製 Q-Learning 收斂曲線
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

if optimal_makespan is not None:

    plt.axhline(
        y=optimal_makespan,
        linestyle="--",
        linewidth=2,
        label="MILP Optimal Makespan"
    )

plt.xlabel("Episode")
plt.ylabel("Makespan")
plt.title("Q-Learning Scheduling Convergence")

plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()


# ===========================================================
# 18. 繪製探索率下降曲線
# ===========================================================

plt.figure(figsize=(12, 4))

plt.plot(epsilon_history)

plt.xlabel("Episode")
plt.ylabel("Epsilon")
plt.title("Epsilon Decay")

plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()


# ===========================================================
# 19. 繪製各方法 Makespan 比較圖
# ===========================================================

plt.figure(figsize=(12, 5))

plt.bar(
    comparison_df["Method"],
    comparison_df["Makespan"]
)

plt.ylabel("Makespan")
plt.title("Scheduling Method Comparison")

plt.xticks(
    rotation=25,
    ha="right"
)

plt.grid(
    axis="y",
    alpha=0.3
)

plt.tight_layout()
plt.show()


# ===========================================================
# 20. 繪製 Q-Learning 最佳方案甘特圖
# ===========================================================

def draw_gantt_chart(schedule, title):
    """
    繪製工件分派甘特圖。
    """

    machine_jobs = [
        []
        for _ in range(NUM_MACHINES)
    ]

    current_time = np.zeros(
        NUM_MACHINES,
        dtype=int
    )

    for job_index, machine_index in enumerate(schedule):

        start_time = int(
            current_time[machine_index]
        )

        processing_time = int(
            job_times[
                job_index,
                machine_index
            ]
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
        f"{title} | Makespan = {max(current_time)}"
    )

    axis.grid(
        axis="x",
        alpha=0.3
    )

    plt.tight_layout()
    plt.show()


draw_gantt_chart(
    best_schedule,
    "Gantt Chart of Best Q-Learning Schedule"
)


# ===========================================================
# 21. 若 MILP 成功，顯示理論最佳解甘特圖
# ===========================================================

if optimal_schedule is not None:

    draw_gantt_chart(
        optimal_schedule,
        "Gantt Chart of MILP Optimal Schedule"
    )
