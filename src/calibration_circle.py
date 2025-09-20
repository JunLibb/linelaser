import numpy as np
import matplotlib.pyplot as plt

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


