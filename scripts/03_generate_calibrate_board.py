import numpy as np
import matplotlib.pyplot as plt
from fscan2.calibration_circle import generate_circle_array
import json

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

def main():
    # Parameters for circle array generation
    rows = 22
    cols = 22
    spacing = 1  # mm
    large_radius = 0.4  # mm
    small_radius = 0.275  # mm
    output_path = './output'
    
    # Generate circle array
    circle_array = generate_circle_array(rows, cols, spacing, large_radius, small_radius)
    
    # Save to file
    save_circle_array(output_path + '/circle_array.json', circle_array)
    print(f"Circle array saved to {output_path + '/circle_array.json'}")
    
    # Load from file (optional, for demonstration)
    loaded_array = load_circle_array(output_path + '/circle_array.json')
    
    # Plot the circle array
    plot_circles(loaded_array, output_path + '/circle_pattern.png')

if __name__ == "__main__":
    main()
