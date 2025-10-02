import os
import cv2
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.detect_circles import detect_circles, detect_circles_with_subpixel, plot_dectected_circles

def main():
    input_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'input', 'detect_circles')
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)
    img_path = os.path.join(input_dir, '0001.png')
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print('图片读取失败')
    # circles = detect_circles(img, min_circularity=0.95)
    # circles = find_circles(img, min_area=30, roundness_threshold=0.7)
    # circles = detect_circles(img)
    circles = detect_circles_with_subpixel(img)
    if len(circles) == 0:
        print('未检测到圆')
    print(f'共检测到 {len(circles)} 个圆')
        
    color_img = plot_dectected_circles(img, circles, counting=True)
    cv2.imshow('Detected Circles', color_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    #保存图片
    cv2.imwrite(os.path.join(output_dir, 'detected_circles.png'), color_img)

if __name__ == '__main__':
    main()  
