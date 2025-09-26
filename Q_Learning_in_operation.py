# ===========================================================
#  主題 : 工廠工件排成最佳化設計 以Q_Learning 估測
#  目標 : 使學員了解Q_Learning 方法與概念
#  產生  20 各工件 四台機器 每台預設加工時間 進行Q_Learning 估測  找到最佳解
#  作者 : 國立雲林科技大學電機系 林家仁
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
import random

# ===============================
# Job times (fixed table)
# ===============================
job_times = [
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
]

NUM_JOBS = len(job_times)
NUM_MACHINES = len(job_times[0])

# ===============================
# Q-Learning parameters
# ===============================
Q = np.zeros((NUM_JOBS + 1, NUM_MACHINES))
alpha = 0.1           # learning rate
gamma = 0.9           # discount factor
epsilon = 1.0         # initial exploration rate
epsilon_decay = 0.98
min_epsilon = 0.05
episodes = 3000       # training episodes

random.seed(42)
np.random.seed(42)

# ===============================
# Makespan calculation
# ===============================
def makespan(schedule):
    machine_times = [0] * NUM_MACHINES
    for job, machine in enumerate(schedule):
        machine_times[machine] += job_times[job][machine]
    return max(machine_times)

# ===============================
# Q-Learning training
# ===============================
makespan_history = []

for episode in range(episodes):
    schedule = []
    for step in range(NUM_JOBS):
        state = step
        if random.random() < epsilon:
            action = random.randint(0, NUM_MACHINES - 1)
        else:
            action = np.argmax(Q[state])

        schedule.append(action)

        # reward: incremental change in makespan
        prev_cost = makespan(schedule[:-1]) if step > 0 else 0
        curr_cost = makespan(schedule)
        reward = -(curr_cost - prev_cost)

        next_state = state + 1
        max_next = np.max(Q[next_state]) if next_state < NUM_JOBS else 0
        Q[state, action] += alpha * (reward + gamma * max_next - Q[state, action])

    epsilon = max(min_epsilon, epsilon * epsilon_decay)
    makespan_history.append(makespan(schedule))

# ===============================
# Convergence plot
# ===============================
window = 50
smoothed = np.convolve(makespan_history, np.ones(window)/window, mode="valid")

plt.figure(figsize=(10,5))
plt.plot(makespan_history, alpha=0.3, label="Raw Makespan")
plt.plot(smoothed, label=f"Smoothed (window={window})", linewidth=2)
plt.xlabel("Episode")
plt.ylabel("Makespan")
plt.title("Q-Learning Scheduling Convergence")
plt.legend()
plt.grid(True)
plt.show()

# ===============================
# Best schedule & Gantt chart
# ===============================
best_schedule = np.argmax(Q[:-1], axis=1)
print("Best schedule (Job -> Machine):", best_schedule)
print("Best Makespan:", makespan(best_schedule))

# --- Gantt chart ---
machine_times = [[] for _ in range(NUM_MACHINES)]
current_time = [0] * NUM_MACHINES

for job_idx, machine_id in enumerate(best_schedule):
    start = current_time[machine_id]
    duration = job_times[job_idx][machine_id]
    end = start + duration
    machine_times[machine_id].append((start, end, job_idx))
    current_time[machine_id] = end

colors = plt.cm.tab20.colors
fig, ax = plt.subplots(figsize=(12, 6))

for m_id, jobs in enumerate(machine_times):
    for start, end, job_idx in jobs:
        ax.barh(y=m_id, width=end-start, left=start, height=0.5, 
                color=colors[job_idx % 20])
        ax.text(start + (end-start)/2, m_id, f"Job {job_idx}", 
                va='center', ha='center', color='white', fontsize=8)

ax.set_yticks(range(NUM_MACHINES))
ax.set_yticklabels([f"Machine {i+1}" for i in range(NUM_MACHINES)])
ax.set_xlabel("Time")
ax.set_title("Gantt Chart of Best Schedule (Q-Learning)")
plt.tight_layout()
plt.show()


