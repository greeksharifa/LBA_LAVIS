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

x_points = [-4.5, -4.2, -4.1, -3.1, -2.4, -1.3, -0.9, -0.8] + [-4.5, -3.7, -2.9, -2.1, -1.7, -1.2, -0.8, -0.5]
y_points = [-1.2, -3.6, -1.9, -4.1, -0.7, -0.5, -0.8, -3.2] + [-4.7, -4.1, -1.1, -4.4, -2.7, -1.5, -1.2, -0.2]

# Create the plot
plt.figure(figsize=(20, 5))


for i in range(4):
    # subplot 2x2
    ax = plt.subplot(1, 4, i+1)
    
    if i == 0:
        green_x = x_points[:8]
        green_y = y_points[:8]
        red_x = x_points[8:]
        red_y = y_points[8:]
        gray_x = []
        gray_y = []
        ax.set_title('Base')
    elif i == 1:
        green_x = x_points[:5]
        green_y = y_points[:5]
        red_x = x_points[8:11]
        red_y = y_points[8:11]
        gray_x = x_points[5:8] + x_points[11:]
        gray_y = y_points[5:8] + y_points[11:]
        # tau_1
        plt.plot([-2.3, -2.3], [-5, 0], 'k-', linewidth=1)
        ax.set_title(r'using $\tau_1$')
    elif i == 2:
        green_x = x_points[:3] + x_points[4:7]
        green_y = y_points[:3] + y_points[4:7]
        red_x = x_points[10:11]
        red_y = y_points[10:11]
        gray_x = x_points[3:4] + x_points[7:10] + x_points[11:]
        gray_y = y_points[3:4] + y_points[7:10] + y_points[11:]
        # tau_1
        plt.plot([-0.7, -0.7], [-5, 0], 'k-', linewidth=1)
        # tau_2
        plt.plot([-5, 0], [-5, 0], 'k-', linewidth=1)
        ax.set_title(r'using $\tau_1$ and $\tau_2$')
    elif i == 3:
        green_x = x_points[:7]
        green_y = [-1.0, -1.6, -2.1, -3.0, -1.1, -0.2, -0.5]
        red_x = x_points[8:9]
        red_y = [-3.6]
        gray_x = x_points[9:14] + x_points[7:8] + x_points[14:]
        gray_y = [-4.8, -3.9, -2.5, -4.2, -2.6, -1.1, -1.5, -0.3]
        # tau_1
        plt.plot([-0.7, -0.7], [-5, 0], 'k-', linewidth=1)
        # tau_2
        plt.plot([-5, 0], [-5, 0], 'k-', linewidth=1)
        ax.set_title(r'using multiple subQAs, $\tau_1$ and $\tau_2$')
    
    plt.scatter(green_x, green_y, color='green', label='right → wrong')
    plt.scatter(red_x, red_y, color='red', label='wrong → right')
    plt.scatter(gray_x, gray_y, color='gray', label='no change')
        

    # Add grid (optional)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Make the plot square
    plt.axis('square')
    # Set plot limits and labels
    # ax.set_xlim(-5, 0)
    # ax.set_ylim(-5, 0)
    ax.axis([-5, 0, -5, 0])
    
    ax.set_xticks(np.arange(-5, 1, 1))
    ax.set_yticks(np.arange(-5, 1, 1))
    ax.set_xlabel('$c(A_{base})$')
    ax.set_ylabel('$c(A_{new})$')

    # Add legend
    # plt.legend(loc='lower right')

    # Show the plot
    plt.show()
    
plt.savefig('scatter_plot.png', dpi=300)

import sys
sys.exit(0)


green_x = [-4.5, -4.2, -4.1, -3.1, -2.4, -1.3, -0.8, -0.9]
green_y = [-1.2, -3.6, -1.9, -4.1, -0.7, -0.5, -3.2, -0.8]

# Red points (roughly matching the image)
red_x = [-4.5, -3.7, -2.9, -2.1, -1.7, -1.2, -0.8, -0.5]
red_y = [-4.7, -4.1, -1.1, -4.4, -2.7, -1.5, -1.2, -4.1]


# Plot the points
plt.scatter(green_x, green_y, color='green', label='right → wrong')
plt.scatter(red_x, red_y, color='red', label='wrong → right')

# tau_1
plt.plot([-2.3, -2.3], [-5, 0], 'k-', linewidth=1)  # k- means black solid line
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