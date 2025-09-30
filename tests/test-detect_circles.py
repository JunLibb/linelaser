import os
import cv2
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.detect_circles import detect_circles, find_circles, detect_circles_with_subpixel
from src.visualization import plot_circles

if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))
    tests_dir = os.path.join(script_dir, 'tests')
    os.makedirs(tests_dir, exist_ok=True)  # Create tests directory if it doesn't exist
    os.chdir(tests_dir)  # Change working directory to tests
    img_path = os.path.join(tests_dir, '0006.png')
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print('图片读取失败')
    # circles = detect_circles(img, min_circularity=0.95)
    # circles = find_circles(img, min_area=30, roundness_threshold=0.7)
    circles = detect_circles_with_subpixel(img, min_area=30, roundness_threshold=0.7)
    if len(circles) == 0:
        print('未检测到圆')
    print(f'共检测到 {len(circles)} 个圆')
        
    color_img = plot_circles(img, circles, counting=True)
    cv2.imshow('Detected Circles', color_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    #保存图片
    cv2.imwrite(os.path.join(tests_dir, 'detected_circles.png'), color_img)
    
