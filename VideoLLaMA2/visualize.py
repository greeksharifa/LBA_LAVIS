import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.size'] = 14  # Base font size
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['legend.fontsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

# Create sample data
# green points (roughly matching the image)
green_x = [-4.5, -4.1, -3.1, -2.4, -1.3, -0.8, -0.9, -0.5]
green_y = [-1.2, -1.9, -4.1, -0.7, -0.5, -3.2, -0.8, -4.1]

# Red points (roughly matching the image)
red_x = [-4.5, -4.2, -3.7, -2.9, -2.1, -1.7, -1.2, -0.8]
red_y = [-4.7, -3.6, -4.1, -1.1, -4.4, -2.7, -1.5, -1.2]

# Create the plot
plt.figure(figsize=(6, 6))

# Plot the points
plt.scatter(green_x, green_y, color='green', label='right → wrong')
plt.scatter(red_x, red_y, color='red', label='wrong → right')

# tau_1
plt.plot([-2, -2], [-5, 0], 'k-', linewidth=1)  # k- means black solid line
# tau_2
plt.plot([-5, 0], [-5, 0], 'k-', linewidth=1)  # k- means black solid line

# Set plot limits and labels
plt.xlim(-5, 0)
plt.ylim(-5, 0)
plt.xlabel('$c(A_{base})$')
plt.ylabel('$c(A_{new})$')

# Add grid (optional)
plt.grid(True, linestyle='--', alpha=0.7)

# Add legend
# plt.legend(loc='lower right')

# Make the plot square
plt.axis('square')

# Show the plot
plt.show()
plt.savefig('scatter_plot.png', dpi=300)