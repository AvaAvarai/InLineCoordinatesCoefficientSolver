import numpy as np
import matplotlib.pyplot as plt
from scipy.special import comb
from matplotlib.colors import LinearSegmentedColormap
import pandas as pd
from matplotlib.widgets import Slider, Button
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
    root.withdraw()  # Hide the main window
    file_path = filedialog.askopenfilename(initialdir="./data", title="Select CSV file",
                                           filetypes=(("CSV files", "*.csv"), ("all files", "*.*")))
    root.destroy()
    return file_path

# Get the CSV file path
csv_file_path = select_csv_file()

if not csv_file_path:
    print("No file selected. Exiting.")
    exit()

df = pd.read_csv(csv_file_path)  # Load the selected CSV file

# Find Class column index
class_col = df.columns.get_loc('Class')

# Get unique class names (assuming binary scenario)
class_names = df['Class'].unique()
if len(class_names) != 2:
    print("Error: This script is designed for binary classification. Please use a dataset with exactly two classes.")
    exit()

# Separate data into two classes, dropping the Class column
class1 = df[df['Class'] == class_names[0]].drop('Class', axis=1).values
class2 = df[df['Class'] == class_names[1]].drop('Class', axis=1).values

# Combine data into a single array
data = [class1, class2]

# Dynamically determine figure size based on data
x_max = np.max([np.sum(np.abs(sample)) for class_data in data for sample in class_data])
y_size = max(len(class1), len(class2))
fig, ax = plt.subplots(figsize=(12, max(6, y_size)))
plt.subplots_adjust(bottom=0.3)  # Make room for sliders

base_colors = ['red', 'blue']
markers = ['o', 's']
endpoint_markers = ['v', '^']  # Different markers for endpoints

# Initial coefficients - one for each feature
num_features = df.shape[1] - 1  # Subtract Class column
coefficients = np.ones(num_features)  # Initialize coefficients to 1.0

def solve_separation():
    # Calculate degree vector for all points
    all_points = np.vstack((class1, class2))
    degree_vector = np.zeros(num_features)
    
    # For each feature, calculate sum of absolute differences between all pairs of points
    for i in range(num_features):
        diff_matrix = np.abs(all_points[:, i].reshape(-1, 1) - all_points[:, i].reshape(1, -1))
        degree_vector[i] = np.sum(diff_matrix)
    
    # Normalize degree vector to maintain scale
    new_coefs = degree_vector * num_features / np.sum(np.abs(degree_vector))
    
    # Update sliders and plot
    for i, slider in enumerate(sliders):
        slider.set_val(new_coefs[i])
        
    # Verify separation
    new_weighted_sums_class1 = np.sum(class1 * new_coefs, axis=1)
    new_weighted_sums_class2 = np.sum(class2 * new_coefs, axis=1)
    
    if np.max(new_weighted_sums_class1) >= np.min(new_weighted_sums_class2):
        print("Warning: Perfect separation not achieved with current approach")
    else:
        print("Separation achieved: All Class 1 points are ordered before Class 2 points")

def update_plot(coef):
    ax.clear()
    all_endpoints = []
    for i, class_data in enumerate(data):  # For each class
        # Create a color gradient for each class
        cmap = LinearSegmentedColormap.from_list(f"custom_{base_colors[i]}", [base_colors[i], 'white'])
        colors = cmap(np.linspace(0.2, 0.8, len(class_data)))  # Dynamically create shades based on number of samples

        for j, sample in enumerate(class_data):  # For each sample in the class
            modified_sample = sample * coef
            cumulative_sum = np.concatenate(([0], np.cumsum(np.abs(modified_sample))))  # Start at 0, then use absolute values
            y_position = 0  # All points on the x-axis (y=0)
            
            # Draw intermediate points
            ax.scatter(cumulative_sum[:-1], [y_position] * (len(cumulative_sum)-1), c=[colors[j]], marker=markers[i], s=50)
            
            # Draw endpoint with different marker
            ax.scatter(cumulative_sum[-1], y_position, c=[colors[j]], marker=endpoint_markers[i], s=100)
            
            # Create Bezier curve between points of a single sample
            points = np.column_stack((cumulative_sum, [y_position] * len(cumulative_sum)))
            
            for k in range(len(cumulative_sum) - 1):  # segments between points
                x_start, x_end = points[k:k+2, 0]
                x_mid = (x_start + x_end) / 2
                y_control = 0.05 if i == 0 else -0.05  # Small arc up for class 1, down for class 2
                
                control_points = np.array([
                    [x_start, y_position],
                    [x_mid, y_control],
                    [x_end, y_position]
                ])
                
                curve = bezier_curve(control_points)
                ax.plot(curve[:, 0], curve[:, 1], c=colors[j], alpha=0.5)
            
            # Store endpoint information (last cumulative sum value for each sample)
            all_endpoints.append((cumulative_sum[-1], i, j))

    # Sort endpoints by x-coordinate
    all_endpoints.sort(key=lambda x: x[0])

    # Detect conflicts based on ordering
    conflicts = []
    current_class = None
    for idx, (x_value, class_idx, sample_idx) in enumerate(all_endpoints):
        if current_class is None:
            current_class = class_idx
        elif class_idx != current_class:
            # A conflict is detected when there is a class switch
            conflicts.append((x_value, class_idx, sample_idx))
            current_class = class_idx
    
    # Highlight conflicting endpoints
    for endpoint in conflicts:
        ax.scatter(endpoint[0], 0, c='yellow', s=100, zorder=5, edgecolors='black')

    # Draw line at y=0
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1)

    ax.set_yticks([])  # Remove y-axis ticks as we're only using y=0
    ax.set_title(f'In-Line Coordinate Plot of {os.path.basename(csv_file_path)}')
    ax.set_xlabel('Cumulative Value')
    ax.set_ylabel('Samples')

    # Create custom legend
    legend_elements = [
        plt.Line2D([0], [0], marker=markers[0], color='w', label=f'{class_names[0]} points',
                   markerfacecolor='red', markersize=10),
        plt.Line2D([0], [0], marker=endpoint_markers[0], color='w', label=f'{class_names[0]} endpoints',
                   markerfacecolor='red', markersize=10),
        plt.Line2D([0], [0], marker=markers[1], color='w', label=f'{class_names[1]} points',
                   markerfacecolor='blue', markersize=10),
        plt.Line2D([0], [0], marker=endpoint_markers[1], color='w', label=f'{class_names[1]} endpoints',
                   markerfacecolor='blue', markersize=10),
        plt.Line2D([0], [0], marker='o', color='w', label='Conflict',
                   markerfacecolor='yellow', markersize=5, markeredgecolor='black')
    ]
    ax.legend(handles=legend_elements)

    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, x_max * max(2.0, max(np.abs(coef))))  # Adjust x-axis limit based on coefficients
    fig.canvas.draw_idle()

# Create sliders
slider_axes = [plt.axes([0.1, 0.05 + 0.03*i, 0.65, 0.03]) for i in range(num_features)]
sliders = [Slider(ax, f'Feature {i+1}', -5.0, 5.0, valinit=1.0) for i, ax in enumerate(slider_axes)]

# Create solve button
solve_button_ax = plt.axes([0.8, 0.05, 0.15, 0.03])
solve_button = Button(solve_button_ax, 'Solve')
solve_button.on_clicked(lambda x: solve_separation())

# Update function for sliders
def update(val):
    coef = np.array([slider.val for slider in sliders])
    update_plot(coef)

# Attach update function to sliders
for slider in sliders:
    slider.on_changed(update)

# Initial plot
update_plot(coefficients)

plt.show()
