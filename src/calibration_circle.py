import numpy as np
import matplotlib.pyplot as plt
import json

def generate_circle_array(rows, cols, spacing, large_radius, small_radius):
    """
    生成圆阵列，输出[x, y, r]数组。大圆索引为相对中心的(0,2),(3,0),(0,-2),(-3,0),(-1,2)。
    参数：
        rows: 行数
        cols: 列数
        spacing: 圆心间距
        large_radius: 大圆半径
        small_radius: 小圆半径
    返回：
        np.ndarray, shape=(rows*cols, 3)
    """
    x_coords = np.arange(cols)
    y_coords = np.arange(rows)
    xx, yy = np.meshgrid(x_coords, y_coords)
    xx = xx.flatten()
    yy = yy.flatten()

    row_center = (rows - 1) // 2
    col_center = (cols - 1) // 2
    # rel_indices = list(zip(yy - row_center, xx - col_center))
    rel_indices = list(zip(xx - col_center, yy - row_center))
    big_circle_indices = [
        (0, 2),
        (3, 0),
        (0, -2),
        (-3, 0),
        (-1, 2)
    ]
    arr = []
    for x, y, rel_idx in zip(xx, yy, rel_indices):
        r = large_radius if rel_idx in big_circle_indices else small_radius
        arr.append([x * spacing - col_center * spacing, y * spacing - row_center * spacing, r])
    return [[float(x), float(y), float(r), 1] for x, y, r in arr]


def save_circle_array(filename, circle_array):
    """Save circle array to a JSON file"""
    with open(filename, 'w') as f:
        json.dump(circle_array, f, indent=2)

def load_circle_array(filename):
    """Load circle array from a JSON file"""
    with open(filename, 'r') as f:
        return np.array(json.load(f))

def plot_circles(circle_array, output_file='circle_pattern.png'):
    """Plot the circle array and save to file
    
    Args:
        circle_array: Array of circles in format [x, y, r, _]
        output_file: Filename to save the plot (default: 'circle_pattern.png')
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Plot each circle
    for x, y, r, _ in circle_array:
        circle = plt.Circle((x, y), r, fill=False, color='blue', linewidth=1)
        ax.add_patch(circle)
        # Add center point
        plt.plot(x, y, 'ro', markersize=2)
    
    # Set equal aspect ratio and adjust plot limits
    ax.set_aspect('equal', 'box')
    ax.set_xlim(circle_array[:, 0].min() - 1, circle_array[:, 0].max() + 1)
    ax.set_ylim(circle_array[:, 1].min() - 1, circle_array[:, 1].max() + 1)
    plt.title('Circle Array Pattern')
    plt.xlabel('X (mm)')
    plt.ylabel('Y (mm)')
    plt.grid(True)
    
    # Save the figure before showing
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_file}")
    
    # Show the plot
    plt.show()