import numpy as np
import matplotlib.pyplot as plt
from scipy.special import comb
from matplotlib.colors import LinearSegmentedColormap
import pandas as pd
from matplotlib.widgets import Slider, Button, TextBox, CheckButtons
import tkinter as tk
from tkinter import filedialog
import os

# Function to create Bezier curve
def bezier_curve(points, num=200):
    N = len(points)
    t = np.linspace(0, 1, num)
    curve = np.zeros((num, 2))
    for i in range(N):
        curve += np.outer(comb(N - 1, i) * (1 - t)**(N - 1 - i) * t**i, points[i])
    return curve

# Function to open file dialog and select CSV file
def select_csv_file():
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(initialdir="./data", title="Select CSV file",
                                           filetypes=(("CSV files", "*.csv"), ("all files", "*.*")))
    root.destroy()
    return file_path

# Load CSV file
csv_file_path = select_csv_file()
if not csv_file_path:
    print("No file selected. Exiting.")
    exit()

df = pd.read_csv(csv_file_path)
class_column = df.columns[df.columns.str.lower() == 'class']
if not class_column.any():
    print("Error: No 'Class' column found in the CSV file.")
    exit()

class_col = df.columns.get_loc(class_column[0])
class_names = df[class_column[0]].unique()
if len(class_names) != 2:
    print("Error: This script is designed for binary classification.")
    exit()

# Get feature columns (all columns except class)
feature_columns = [col for col in df.columns if col.lower() != 'class']

# Separate data into two classes, using only feature columns
class1 = df[df[class_column[0]] == class_names[0]][feature_columns].values
class2 = df[df[class_column[0]] == class_names[1]][feature_columns].values
data = [class1, class2]

# Initial setup
num_features = len(feature_columns)
coefficients = np.ones(num_features)
allow_negative_coefficients = True
fig, ax = plt.subplots(figsize=(12, 6))
plt.subplots_adjust(bottom=0.5)

# Variables to store selected points and drawing order
selected_points = []
draw_order = [0, 1]  # Default order: class1 then class2

def highlight_selected_points():
    ax.clear()
    for order_idx in draw_order:  # Draw classes in current order
        class_data = data[order_idx]
        base_color = 'blue' if order_idx == 0 else 'red'
        cmap = LinearSegmentedColormap.from_list(f"{base_color}_cmap", [base_color, 'white'])
        colors = cmap(np.linspace(0.2, 0.8, len(class_data)))

        for j, sample in enumerate(class_data):
            modified_sample = sample * coefficients
            cumulative_sum = np.concatenate(([0], np.cumsum(np.abs(modified_sample))))
            y_position = 0
            endpoint_color = 'green' if (order_idx, j) in selected_points else base_color

            ax.scatter(cumulative_sum[:-1], [y_position] * (len(cumulative_sum) - 1), color=colors[j], s=50)
            ax.scatter(cumulative_sum[-1], y_position, color=endpoint_color, s=100)

            points = np.column_stack((cumulative_sum, [y_position] * len(cumulative_sum)))
            for k in range(len(cumulative_sum) - 1):
                x_start, x_end = points[k:k + 2, 0]
                x_mid = (x_start + x_end) / 2
                y_control = 0.05 if order_idx == 0 else -0.05
                
                control_points = np.array([
                    [x_start, y_position],
                    [x_mid, y_control],
                    [x_end, y_position]
                ])
                curve = bezier_curve(control_points)
                ax.plot(curve[:, 0], curve[:, 1], c=colors[j], alpha=0.5)

    ax.axhline(y=0, color='black', linestyle='--', linewidth=1)
    ax.set_xlabel('Cumulative Value')
    ax.set_title(f'In-Line Coordinate Plot of {os.path.basename(csv_file_path)}')
    fig.canvas.draw_idle()

def update_slider(val, idx):
    global coefficients
    coefficients[idx] = sliders[idx].val
    text_boxes[idx].set_val(f"{coefficients[idx]:.2f}")
    highlight_selected_points()

def update_text(val, idx):
    global coefficients
    try:
        coefficients[idx] = float(val)
        sliders[idx].set_val(coefficients[idx])
        highlight_selected_points()
    except ValueError:
        text_boxes[idx].set_val(f"{coefficients[idx]:.2f}")

def solve_separation():
    global coefficients
    if len(selected_points) != 2:
        print("Select exactly two points before solving.")
        return

    print("Solving for separation...")
    
    # Retrieve the selected points
    idx1, idx2 = selected_points
    sample1 = data[idx1[0]][idx1[1]]
    sample2 = data[idx2[0]][idx2[1]]

    # Compute weighted sums
    wsk_a = np.sum(coefficients * sample1)
    wsk_b = np.sum(coefficients * sample2)

    # If already separated, no adjustment is needed
    if wsk_a < wsk_b:
        print("Samples already satisfy WSK(a) < WSK(b). No adjustment needed.")
        return

    # Compute difference vector
    diff = sample2 - sample1

    # Calculate the weighted sum difference
    delta_wsk = wsk_b - wsk_a

    # Compute adjustment magnitude for each coefficient
    abs_diff_sum = np.sum(np.abs(diff))  # Normalize by total feature difference
    if abs_diff_sum == 0:
        print("No feature differences found. Cannot adjust coefficients.")
        return

    adjustment = delta_wsk / abs_diff_sum

    # Apply adjustments to coefficients one at a time
    for i in range(len(coefficients)):
        coefficients[i] -= adjustment * np.sign(diff[i])

        # Update sliders and text boxes to reflect new coefficients
        sliders[i].set_val(coefficients[i])
        text_boxes[i].set_val(f"{coefficients[i]:.2f}")

        # Refresh plot after each coefficient adjustment
        highlight_selected_points()

    # Clear selected points
    selected_points.clear()
    print("Final coefficients after solving:", coefficients)

def toggle_negative_coefficients(label):
    global allow_negative_coefficients
    allow_negative_coefficients = not allow_negative_coefficients
    for slider in sliders:
        slider.valmin = -5.0 if allow_negative_coefficients else 0.0
        slider.ax.set_xlim(slider.valmin, slider.valmax)
    highlight_selected_points()

def bring_to_front(class_idx):
    global draw_order
    if class_idx in draw_order:
        draw_order.remove(class_idx)
        draw_order.append(class_idx)  # Add to end to draw last (on top)
        highlight_selected_points()

def on_click(event):
    if event.inaxes == ax:
        min_dist = float('inf')
        selected_idx = None
        for i, class_data in enumerate(data):
            for j, sample in enumerate(class_data):
                modified_sample = sample * coefficients
                cumulative_sum = np.sum(np.abs(modified_sample))
                dist = abs(event.xdata - cumulative_sum)
                if dist < min_dist:
                    min_dist = dist
                    selected_idx = (i, j)

        if selected_idx:
            if selected_idx in selected_points:
                selected_points.remove(selected_idx)
            selected_points.append(selected_idx)
            if len(selected_points) > 2:
                selected_points.pop(0)
            highlight_selected_points()

# Create sliders and text boxes
slider_axes = [plt.axes([0.1, 0.4 - 0.03 * i, 0.4, 0.02]) for i in range(num_features)]
sliders = [Slider(ax, feature_columns[i], -5.0, 5.0, valinit=1.0) for i, ax in enumerate(slider_axes)]

text_box_axes = [plt.axes([0.55, 0.4 - 0.03 * i, 0.15, 0.02]) for i in range(num_features)]
text_boxes = [TextBox(ax, '', initial=f"{coefficients[i]:.2f}") for i, ax in enumerate(text_box_axes)]

# Link sliders and text boxes
for i in range(num_features):
    sliders[i].on_changed(lambda val, idx=i: update_slider(val, idx))
    text_boxes[i].on_submit(lambda val, idx=i: update_text(val, idx))

# Solve button
solve_button_ax = plt.axes([0.8, 0.05, 0.15, 0.03])
solve_button = Button(solve_button_ax, 'Solve')
solve_button.on_clicked(lambda _: solve_separation())

# Check button for allowing negative coefficients
check_button_ax = plt.axes([0.8, 0.1, 0.15, 0.03])
check_button = CheckButtons(check_button_ax, ['Allow Neg Coef'], [allow_negative_coefficients])
check_button.on_clicked(toggle_negative_coefficients)

# Add buttons for class ordering
class1_button_ax = plt.axes([0.8, 0.15, 0.15, 0.03])
class2_button_ax = plt.axes([0.8, 0.2, 0.15, 0.03])
class1_button = Button(class1_button_ax, f'Bring {class_names[0]} to front')
class2_button = Button(class2_button_ax, f'Bring {class_names[1]} to front')
class1_button.on_clicked(lambda _: bring_to_front(0))
class2_button.on_clicked(lambda _: bring_to_front(1))

fig.canvas.mpl_connect('button_press_event', on_click)
highlight_selected_points()
plt.show()
