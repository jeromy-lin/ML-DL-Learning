# ===========================================================
#  主題 : 工廠工件排成最佳化設計 以Q_Learning 估測
#  目標 : 使學員了解Q_Learning 方法與概念
#  產生  20 各工件 四台機器 每台預設加工時間 進行Q_Learning 估測  找到最佳解
#  作者 : 國立雲林科技大學電機系 林家仁
# ============================================================


import numpy as np
import random
import matplotlib.pyplot as plt

# --- 20 jobs, 4 machines processing times (jobs x machines)
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

Q = np.zeros((NUM_JOBS + 1, NUM_MACHINES))

alpha = 0.1
gamma = 0.9
epsilon = 1.0
epsilon_decay = 0.95
min_epsilon = 0.1

def makespan(schedule):
    times = [0] * NUM_MACHINES
    for job_idx, machine_id in enumerate(schedule):
        times[machine_id] += job_times[job_idx][machine_id]
    return max(times)

# Q-Learning training
for episode in range(1000):
    schedule = []
    total_reward = 0
    for step in range(NUM_JOBS):
        state = step
        if random.random() < epsilon:
            action = random.randint(0, NUM_MACHINES - 1)
        else:
            action = np.argmax(Q[state])
        schedule.append(action)
        cost = makespan(schedule)
        reward = -cost
        next_state = state + 1
        max_next = np.max(Q[next_state]) if next_state < NUM_JOBS else 0
        Q[state, action] += alpha * (reward + gamma * max_next - Q[state, action])
        total_reward += reward
    epsilon = max(min_epsilon, epsilon * epsilon_decay)
    if (episode + 1) % 100 == 0:
        print(f"Episode {episode + 1}, best makespan: {-total_reward / NUM_JOBS:.2f}")

# Test schedule
schedule = []
for step in range(NUM_JOBS):
    state = step
    action = np.argmax(Q[state])
    schedule.append(action)

print("\nBest schedule (Job -> Machine):")
for job_idx, machine_id in enumerate(schedule):
    print(f"Job {job_idx} -> Machine {machine_id + 1}")

print(f"Total makespan: {makespan(schedule)}")

# Plot Gantt chart
machine_times = [[] for _ in range(NUM_MACHINES)]  # store (start, end, job)
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
        ax.barh(y=m_id, width=end-start, left=start, height=0.5, color=colors[job_idx % 20])
        ax.text(start + (end-start)/2, m_id, f"Job {job_idx}", va='center', ha='center', color='white', fontsize=8)

ax.set_yticks(range(NUM_MACHINES))
ax.set_yticklabels([f"Machine {i+1}" for i in range(NUM_MACHINES)])
ax.set_xlabel("Time")
ax.set_title("Factory Scheduling Gantt Chart (Q-Learning)")

plt.tight_layout()
plt.show()
