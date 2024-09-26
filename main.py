# write a demo for the coefswap project
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import comb
from matplotlib.colors import LinearSegmentedColormap
import pandas as pd
from matplotlib.widgets import Slider

# Function to create Bezier curve
def bezier_curve(points, num=200):
    N = len(points)
    t = np.linspace(0, 1, num)
    curve = np.zeros((num, 2))
    for i in range(N):
        curve += np.outer(comb(N - 1, i) * (1 - t)**(N - 1 - i) * t**i, points[i])
    return curve

# Load data from data.csv
df = pd.read_csv('data.csv')

# Separate data into two classes (now handling imbalanced classes)
class1 = df[df['Class'] == 1].iloc[:, 2:].values
class2 = df[df['Class'] == 2].iloc[:, 2:].values

# Combine data into a single array
data = [class1, class2]

# Print the data set
print("Data set:")
for i, class_data in enumerate(data):
    print(f"Class {i+1}:")
    for j, sample in enumerate(class_data):
        print(f"  Sample {j+1}: {sample}")

# Dynamically determine figure size based on data
x_max = np.max([np.sum(np.abs(sample)) for class_data in data for sample in class_data])
y_size = max(len(class1), len(class2))
fig, ax = plt.subplots(figsize=(12, max(6, y_size)))
plt.subplots_adjust(bottom=0.3)  # Make room for sliders

base_colors = ['red', 'blue']
markers = ['o', 's']

# Initial coefficients
num_features = df.shape[1] - 2  # Subtract 2 for 'Class' and 'Sample' columns
coefficients = np.ones(num_features) * 0.5  # Initialize coefficients to 0.5

def update_plot(coef):
    ax.clear()
    for i, class_data in enumerate(data):  # For each class
        # Create a color gradient for each class
        cmap = LinearSegmentedColormap.from_list(f"custom_{base_colors[i]}", [base_colors[i], 'white'])
        colors = cmap(np.linspace(0.2, 0.8, len(class_data)))  # Dynamically create shades based on number of samples

        for j, sample in enumerate(class_data):  # For each sample in the class
            modified_sample = sample * coef
            cumulative_sum = np.concatenate(([0], np.cumsum(np.abs(modified_sample))))  # Start at 0, then use absolute values
            y_position = 0  # All points on the x-axis (y=0)
            scatter = ax.scatter(cumulative_sum, [y_position] * len(cumulative_sum), c=[colors[j]], marker=markers[i], s=50)
            
            # Label points with index in their case (1-indexed)
            for k, x in enumerate(cumulative_sum):
                ax.annotate(f'{j+1},{k+1}', (x, y_position), xytext=(0, 5), 
                             textcoords='offset points', ha='center', va='bottom',
                             fontsize=8, color=colors[j])
            
            # Create Bezier curve between points of a single sample
            points = np.column_stack((cumulative_sum, [y_position] * len(cumulative_sum)))
            
            # Add control points to make curves arc slightly
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

    # Draw line at y=0
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1)

    ax.set_yticks([])  # Remove y-axis ticks as we're only using y=0
    ax.set_title('In-Line Coordinate Plot of generated samples')
    ax.set_xlabel('Cumulative Value')
    ax.set_ylabel('Samples')

    # Create custom legend
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w', label='Class 1',
                                  markerfacecolor='red', markersize=10),
                       plt.Line2D([0], [0], marker='s', color='w', label='Class 2',
                                  markerfacecolor='blue', markersize=10)]
    ax.legend(handles=legend_elements)

    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, x_max * 1.1)  # Set x-axis limit with some padding
    fig.canvas.draw_idle()

# Create sliders
slider_axes = [plt.axes([0.1, 0.05 + 0.03*i, 0.65, 0.03]) for i in range(num_features)]
sliders = [Slider(ax, f'Feature {i+1}', 0.0, 1.0, valinit=0.5) for i, ax in enumerate(slider_axes)]

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
