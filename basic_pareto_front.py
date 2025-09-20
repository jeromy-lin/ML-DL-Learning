# ============================================================
# Topic : Time vs Cost Trade-off (Interactive with Pareto front)
# 目標 : Demonstrate multi-objective trade-off with visible effect
# 作者：國立雲林科技大學 林家仁
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from ipywidgets import interact, FloatSlider

# 1. Generate example data with conflicting Time and Cost
x = np.linspace(1, 10, 50)  # Production quantity

# Define conflicting functions
Time = 10 - 0.5*x + x**1.5       # 非線性，隨x增加時間增加
Cost = 50 - 3*x + 0.5*x**2       # 非線性，隨x增加成本先下降後上升

df = pd.DataFrame({'x': x, 'Time': Time, 'Cost': Cost})

# 2. Function to find Pareto front
def is_pareto_efficient(costs):
    is_efficient = np.ones(costs.shape[0], dtype=bool)
    for i, c in enumerate(costs):
        if is_efficient[i]:
            is_efficient[is_efficient] = np.any(costs[is_efficient] < c, axis=1)
            is_efficient[i] = True
    return is_efficient

# 3. Interactive plotting function
def plot_tradeoff(w_time=0.5, w_cost=0.5):
    # Normalize
    T_norm = (df['Time'] - df['Time'].min()) / (df['Time'].max() - df['Time'].min())
    C_norm = (df['Cost'] - df['Cost'].min()) / (df['Cost'].max() - df['Cost'].min())

    # Weighted sum
    total = w_time * T_norm + w_cost * C_norm
    best_idx = total.idxmin()

    # Pareto front
    costs_array = np.vstack([df['Cost'], df['Time']]).T
    pareto_mask = is_pareto_efficient(costs_array)
    pareto_points = costs_array[pareto_mask]

    # Plot
    plt.figure(figsize=(8,6))
    plt.scatter(df['Time'], df['Cost'], alpha=0.3, label='All solutions')
    plt.plot(pareto_points[:,1], pareto_points[:,0], color='red', linewidth=2, label='Pareto front')  # red line
    plt.scatter(df['Time'][best_idx], df['Cost'][best_idx], color='blue', s=100, label='Best weighted solution')

    plt.xlabel('Time')
    plt.ylabel('Cost')
    plt.title(f'Time vs Cost Trade-off (w_time={w_time:.2f}, w_cost={w_cost:.2f})')
    plt.grid(True)
    plt.legend()
    plt.show()

    print("Best weighted compromise solution:")
    print(df.iloc[best_idx])

# 4. Interactive sliders
interact(plot_tradeoff,
         w_time=FloatSlider(min=0, max=1, step=0.05, value=0.5, description='Time weight'),
         w_cost=FloatSlider(min=0, max=1, step=0.05, value=0.5, description='Cost weight'))
