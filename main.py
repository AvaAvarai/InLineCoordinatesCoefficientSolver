import numpy as np
import matplotlib.pyplot as plt
from scipy.special import comb
from matplotlib.colors import LinearSegmentedColormap
import pandas as pd
from matplotlib.widgets import Slider, Button, CheckButtons
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
# Parse 'Class' column to lowercase to handle different cases
class_column = df.columns[df.columns.str.lower() == 'class']
if not class_column.any():
    print("Error: No 'Class' column found in the CSV file.")
    exit()

class_col = df.columns.get_loc(class_column[0])
class_names = df[class_column[0]].unique()
if len(class_names) != 2:
    print("Error: This script is designed for binary classification.")
    exit()

# Separate data into two classes, dropping the Class column
class1 = df[df[class_column[0]] == class_names[0]].drop(class_column[0], axis=1).values
class2 = df[df[class_column[0]] == class_names[1]].drop(class_column[0], axis=1).values
data = [class1, class2]

# Initial plot setup
num_features = df.shape[1] - 1  # Exclude Class column
coefficients = np.ones(num_features)  # Initial coefficients

fig, ax = plt.subplots(figsize=(12, 6))
plt.subplots_adjust(bottom=0.3)

# Variables to store selected points
selected_points = []

def highlight_selected_points():
    ax.clear()
    for i, class_data in enumerate(data):
        # Define color gradient per class
        base_color = 'blue' if i == 0 else 'red'
        cmap = LinearSegmentedColormap.from_list(f"{base_color}_cmap", [base_color, 'white'])
        colors = cmap(np.linspace(0.2, 0.8, len(class_data)))

        for j, sample in enumerate(class_data):
            modified_sample = sample * coefficients
            cumulative_sum = np.concatenate(([0], np.cumsum(modified_sample)))
            y_position = 0

            # Set endpoint color based on selection status
            endpoint_color = 'green' if (i, j) in selected_points else base_color

            # Draw points and endpoint - non-endpoints much smaller now
            ax.scatter(cumulative_sum[:-1], [y_position] * (len(cumulative_sum) - 1), color=colors[j], s=10)  # Reduced from 50 to 10
            ax.scatter(cumulative_sum[-1], y_position, color=endpoint_color, s=100)

            # Draw Bezier curve for each segment
            points = np.column_stack((cumulative_sum, [y_position] * len(cumulative_sum)))
            for k in range(len(cumulative_sum) - 1):
                x_start, x_end = points[k:k + 2, 0]
                x_mid = (x_start + x_end) / 2
                y_control = 0.05 if i == 0 else -0.05  # Create a slight upward or downward arc
                
                control_points = np.array([
                    [x_start, y_position],
                    [x_mid, y_control],
                    [x_end, y_position]
                ])
                curve = bezier_curve(control_points)
                ax.plot(curve[:, 0], curve[:, 1], c=colors[j], alpha=0.5)

    # Draw line at y=0
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1)
    ax.set_xlabel('Cumulative Value')
    ax.set_title(f'In-Line Coordinate Plot of {os.path.basename(csv_file_path)}')
    fig.canvas.draw_idle()

def on_click(event):
    if event.inaxes == ax:
        # Find the nearest point and add it to the selected points
        min_dist = float('inf')
        selected_idx = None
        for i, class_data in enumerate(data):
            for j, sample in enumerate(class_data):
                modified_sample = sample * coefficients
                cumulative_sum = np.concatenate(([0], np.cumsum(modified_sample)))
                x, y = cumulative_sum[-1], 0
                dist = np.hypot(event.xdata - x, event.ydata - y)
                
                if dist < min_dist:
                    min_dist = dist
                    selected_idx = (i, j)

        # Update selected points to keep only the two most recent
        if selected_idx:
            if selected_idx in selected_points:
                selected_points.remove(selected_idx)
            selected_points.append(selected_idx)
            if len(selected_points) > 2:
                selected_points.pop(0)

        highlight_selected_points()

def solve_separation():
    global coefficients
    if len(selected_points) != 2:
        print("Select exactly two points before solving.")
        return
    
    print("Solving for separation...")

    # Get selected points data
    idx1, idx2 = selected_points
    sample1 = data[idx1[0]][idx1[1]]
    sample2 = data[idx2[0]][idx2[1]]
    
    # Calculate initial cumulative sums for selected points
    cumulative_sum1 = np.sum(sample1 * coefficients)
    cumulative_sum2 = np.sum(sample2 * coefficients)
    target_relation = cumulative_sum1 < cumulative_sum2  # Target is to ensure this relation holds for all points

    # Find the largest differing attribute
    differences = np.abs(sample2 - sample1)
    largest_diff_idx = np.argmax(differences)

    # Calculate the adjustment needed to ensure all points of one class come before all points of another class
    # This is a simplification assuming the largest differing attribute is the key to separation
    adjustment_needed = 1.0  # A fixed adjustment to ensure separation

    # Adjust the coefficient for the largest differing attribute to ensure separation
    # Ensure the coefficient does not become 0
    if allow_negative_coefficients:
        coefficients[largest_diff_idx] -= adjustment_needed
    else:
        coefficients[largest_diff_idx] = max(0, coefficients[largest_diff_idx] - adjustment_needed)

    for slider, coef in zip(sliders, coefficients):
        slider.set_val(coef)
    
    # Clear selected points and refresh plot
    selected_points.clear()
    update_plot(coefficients)
    print("Final coefficients after solving:", coefficients)

# Update plot function
def update_plot(coef):
    global coefficients
    coefficients = coef
    highlight_selected_points()

# Sliders and Solve button
slider_axes = [plt.axes([0.1, 0.05 + 0.03*i, 0.65, 0.03]) for i in range(num_features)]
sliders = [Slider(ax, f'Feature {i+1}', -5.0, 5.0, valinit=1.0) for i, ax in enumerate(slider_axes)]
solve_button_ax = plt.axes([0.8, 0.05, 0.15, 0.03])
solve_button = Button(solve_button_ax, 'Solve')
solve_button.on_clicked(lambda x: solve_separation())

# Toggle for allowing negative coefficients
allow_negative_coefficients = True
def toggle_negative_coefficients(label):
    global allow_negative_coefficients
    allow_negative_coefficients = not allow_negative_coefficients
    for slider in sliders:
        if allow_negative_coefficients:
            slider.valmin = -10.0
            slider.valmax = 10.0
        else:
            slider.valmin = 0.0
            slider.valmax = 10.0
    print(f"Allowing negative coefficients: {allow_negative_coefficients}")

# Check button for toggling negative coefficients
checkbox_ax = plt.axes([0.8, 0.1, 0.15, 0.03])
checkbox = CheckButtons(checkbox_ax, ['Allow Neg Coef'], [allow_negative_coefficients])
checkbox.on_clicked(toggle_negative_coefficients)

for slider in sliders:
    slider.on_changed(lambda val: update_plot(np.array([slider.val for slider in sliders])))

fig.canvas.mpl_connect('button_press_event', on_click)
update_plot(coefficients)
plt.show()
