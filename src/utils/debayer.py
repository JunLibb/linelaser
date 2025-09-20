import cv2
import numpy as np
from pathlib import Path

def debayer_image(input_path: str, output_path: str, pattern: str = 'BGGR'):
    """
    对PNG文件进行解拜耳矩阵处理
    
    Args:
        input_path: 输入PNG文件路径
        output_path: 输出文件路径
        pattern: 拜耳阵列模式，可选值：'BGGR', 'GBRG', 'GRBG', 'RGGB'
    """
    # 读取PNG文件
    raw = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    if raw is None:
        raise ValueError(f"无法读取文件: {input_path}")

    # 获取拜耳模式对应的OpenCV常量
    patterns = {
        'BGGR': cv2.COLOR_BAYER_BG2BGR,
        'GBRG': cv2.COLOR_BAYER_GB2BGR,
        'GRBG': cv2.COLOR_BAYER_GR2BGR,
        'RGGB': cv2.COLOR_BAYER_RG2BGR
    }
    
    if pattern not in patterns:
        raise ValueError(f"不支持的拜耳模式: {pattern}")

    # 进行解拜耳处理
    debayered = cv2.cvtColor(raw, patterns[pattern])
    
    # 保存处理后的图像
    cv2.imwrite(output_path, debayered)
    
    return debayered

if __name__ == "__main__":
    pass