# ============================================
#  Basic Time and Cost Function
#  目標 : 使學員了解時間與成本函數以及圖形繪製概念
#  作者：國立雲林科技大學 林家仁
# ============================================
import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display, clear_output

# Quantity range
x = np.linspace(0, 100, 100)

# --- Linear Models ---
y_manu_linear = 50 * x + 1000 # 以線條顯示的製造成本
y_logi_linear = 20 * x + 500      # 以線條顯示的運送成本

# --- Real Convex Models () ---
y_manu_real = 1.0 * x**2 + 50 * x + 1000  # 以弧度顯示的製造成本
y_logi_real = 0.8 * x**2 + 20 * x + 500   # 以弧度顯示的運送成本

# --- Output widget ---
out = widgets.Output()

# --- Plot function ---
def plot_model(show_real="Linear"):
    with out:
        clear_output(wait=True)
        plt.figure(figsize=(10,6))
        # Linear always
        plt.plot(x, y_manu_linear, "--", label="Manufacturing (Linear)", color="blue")
        plt.plot(x, y_logi_linear, "--", label="Logistics (Linear)", color="green")
        # Real curves
        if show_real == "Curve":
            plt.plot(x, y_manu_real, label="Manufacturing (Real/Convex)", color="blue")
            plt.plot(x, y_logi_real, label="Logistics (Real/Convex)", color="green")
        plt.title("Quantity vs Cost: Linear vs Real Models")
        plt.xlabel("Quantity (units)")
        plt.ylabel("Cost")
        plt.legend()
        plt.grid(True)
        plt.show()

# --- Dropdown widget ---
dropdown = widgets.Dropdown(
    options=["Linear", "Curve"],
    value="Linear",
    description='Show Real:',
    disabled=False,
)

def on_dropdown_change(change):
    if change['type'] == 'change' and change['name'] == 'value':
        plot_model(show_real=change['new'])

dropdown.observe(on_dropdown_change)

# --- Display ---
display(dropdown)
display(out)

# Initial plot (linear only)
plot_model(show_real="No")
