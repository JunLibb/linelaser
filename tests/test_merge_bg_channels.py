import os
import cv2
from pathlib import Path

# 添加项目根目录到系统路径
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.product.merge_bg_image import merge_bgimage

def main():
    input_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'input', 'merge_channels')
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')
    img_path_b = os.path.join(input_dir, '0001.png')
    img_path_gr = os.path.join(input_dir, '0002.png')
    print(f'读取图片: {img_path_b}\n读取图片: {img_path_gr}')
    img_b = cv2.imread(img_path_b, cv2.IMREAD_GRAYSCALE)
    img_gr = cv2.imread(img_path_gr, cv2.IMREAD_COLOR)
    merged = merge_bgimage(img_b, img_gr)
    if merged is None:
        print('处理失败')
        return
    cv2.imshow('合成结果(B=img1, RG=img2变换)', merged)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite(os.path.join(output_dir, 'merged.png'), merged)


if __name__ == '__main__':
    main()
