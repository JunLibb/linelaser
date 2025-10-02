import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path

# 添加项目根目录到系统路径
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.match_circles import match_circles_with_normalization, plot_matched_results_separate
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

def main():
    # 设置命令行参数
    parser = argparse.ArgumentParser(description='检测并匹配两幅图像中的圆')
    parser.add_argument('--img1', type=str, default=str(Path(__file__).parent.parent / 'tests' / 'input' / 'match' / '0001.png'),
                        help='第一张图片的路径')
    parser.add_argument('--img2', type=str, default=str(Path(__file__).parent.parent / 'tests' / 'input' / 'match' / '0002.png'),
                        help='第二张图片的路径')
    parser.add_argument('--output', type=str, default=str(Path(__file__).parent.parent / 'tests' / 'output' / 'matched_circles.png'),
                        help='输出结果图片的路径')
    
    args = parser.parse_args()
    
    # 确保输出目录存在
    output_dir = Path(args.output).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # 加载图片并检测圆
        print(f"正在加载并处理图片: {args.img1}")
        img1, circles1 = load_and_detect(args.img1)
        print(f"检测到 {len(circles1)} 个圆")
        
        print(f"\n正在加载并处理图片: {args.img2}")
        img2, circles2 = load_and_detect(args.img2)
        print(f"检测到 {len(circles2)} 个圆")
        
        if len(circles1) == 0 or len(circles2) == 0:
            print("错误：至少有一张图片没有检测到圆")
            return 1
        
        # 使用归一化匹配
        print("\n开始匹配圆...")
        matched1, matched2, dist = match_circles_with_normalization(
            circles1, circles2, log=True
        )
        
        # 可视化结果
        vis = plot_matched_results_separate(img1, img2, circles1, circles2)
        # 显示结果
        plt.figure(figsize=(16, 8))
        plt.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
        plt.title('Detected Circles (Green numbers are circle indices)')
        plt.axis('off')
        plt.tight_layout()
        plt.show()
        # 保存结果
        cv2.imwrite(args.output, vis)
        print(f"\n匹配完成！结果已保存至: {args.output}")
        return 0
        
    except Exception as e:
        print(f"发生错误: {str(e)}", file=sys.stderr)
        return 1

# 直接执行脚本
if __name__ == "__main__":
    sys.exit(main())
