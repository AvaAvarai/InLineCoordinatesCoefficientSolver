# write a demo for the coefswap project
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import comb
from matplotlib.colors import LinearSegmentedColormap
import pandas as pd

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

# Separate data into two classes
class1 = df[df['Class'] == 1].iloc[:, 2:].values
class2 = df[df['Class'] == 2].iloc[:, 2:].values

# Combine data into a single array
data = np.array([class1, class2])

# Print the data set
print("Data set:")
for i in range(2):
    print(f"Class {i+1}:")
    for j in range(len(data[i])):
        print(f"  Sample {j+1}: {data[i, j]}")

# Dynamically determine figure size based on data
x_max = np.max([np.sum(np.abs(sample)) for class_data in data for sample in class_data])
y_size = max(len(class1), len(class2))
plt.figure(figsize=(12, max(6, y_size)))

base_colors = ['red', 'blue']
markers = ['o', 's']

for i in range(2):  # For each class
    # Create a color gradient for each class
    cmap = LinearSegmentedColormap.from_list(f"custom_{base_colors[i]}", [base_colors[i], 'white'])
    colors = cmap(np.linspace(0.2, 0.8, len(data[i])))  # Dynamically create shades based on number of samples

    for j in range(len(data[i])):  # For each sample in the class
        cumulative_sum = np.cumsum(np.abs(data[i, j]))  # Use absolute values to ensure always increasing
        y_position = 0  # All points on the x-axis (y=0)
        scatter = plt.scatter(cumulative_sum, [y_position] * len(data[i, j]), c=[colors[j]], marker=markers[i], s=50)
        
        # Label points with index in their case
        for k, x in enumerate(cumulative_sum):
            plt.annotate(f'{j},{k}', (x, y_position), xytext=(0, 5), 
                         textcoords='offset points', ha='center', va='bottom',
                         fontsize=8, color=colors[j])
        
        # Create Bezier curve between points of a single sample
        points = np.column_stack((cumulative_sum, [y_position] * len(data[i, j])))
        
        # Add control points to make curves arc slightly
        for k in range(len(data[i, j]) - 1):  # segments between points
            x_start, x_end = points[k:k+2, 0]
            x_mid = (x_start + x_end) / 2
            y_control = 0.05 if i == 0 else -0.05  # Small arc up for class 1, down for class 2
            
            control_points = np.array([
                [x_start, y_position],
                [x_mid, y_control],
                [x_end, y_position]
            ])
            
            curve = bezier_curve(control_points)
            plt.plot(curve[:, 0], curve[:, 1], c=colors[j], alpha=0.5)

# Draw line at y=0
plt.axhline(y=0, color='black', linestyle='--', linewidth=1)

plt.yticks([])  # Remove y-axis ticks as we're only using y=0
plt.title('In-Line Coordinate Plot of generated samples')
plt.xlabel('Cumulative Value')
plt.ylabel('Samples')

# Create custom legend
legend_elements = [plt.Line2D([0], [0], marker='o', color='w', label='Class 1',
                              markerfacecolor='red', markersize=10),
                   plt.Line2D([0], [0], marker='s', color='w', label='Class 2',
                              markerfacecolor='blue', markersize=10)]
plt.legend(handles=legend_elements)

plt.grid(True, alpha=0.3)
plt.xlim(0, x_max * 1.1)  # Set x-axis limit with some padding
plt.show()
