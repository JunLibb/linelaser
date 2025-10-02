import numpy as np
import matplotlib.pyplot as plt
import json

# 添加项目根目录到 Python 路径
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.calibration_circle import generate_circle_array, save_circle_array, load_circle_array, plot_circles

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
