import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# 添加项目根目录到系统路径
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.match import match_circles_with_normalization, visualize_circles
from src.detect_circles import detect_circles

def load_and_detect(image_path):
    """加载图片并检测圆"""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"图片文件未找到: {image_path}")
    
    # 读取图片
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"无法读取图片: {image_path}")
    
    # 转换为灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 检测圆 (返回 [x, y, r, circularity] 格式的数组)
    circles = detect_circles(gray)
    
    # 只保留前三个值 (x, y, r) 用于匹配
    circles_xy = np.array([circle[:3] for circle in circles], dtype=np.float32)
    
    return img, circles_xy

def test_normalized_matching():
    # 图片路径
    base_dir = Path(__file__).parent.parent
    img1_path = base_dir / 'tests' / 'input' / 'match' /'0001.png'
    img2_path = base_dir / 'tests' / 'input' / 'match' /'0002.png'
    # 保存结果
    output_dir = Path(__file__).parent.parent / 'tests' / 'output'
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / 'matched_circles.png'

    # 加载图片并检测圆
    print(f"正在加载并处理图片: {img1_path}")
    img1, circles1 = load_and_detect(str(img1_path))
    print(f"检测到 {len(circles1)} 个圆")
    
    print(f"\n正在加载并处理图片: {img2_path}")
    img2, circles2 = load_and_detect(str(img2_path))
    print(f"检测到 {len(circles2)} 个圆")
    
    if len(circles1) == 0 or len(circles2) == 0:
        print("错误：至少有一张图片没有检测到圆")
        return
    
    # 使用归一化匹配
    print("\n开始匹配圆...")
    matched1, matched2, dist = match_circles_with_normalization(
        circles1, circles2, log=True
    )
    
    # 可视化结果
    vis = visualize_circles(img1, img2, circles1, circles2, output_path)
    
    # 显示结果
    plt.figure(figsize=(16, 8))
    plt.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
    plt.title('Detected Circles (Green numbers are circle indices)')
    plt.axis('off')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    test_normalized_matching()
