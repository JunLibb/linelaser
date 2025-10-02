"""

"""
import cv2
import numpy as np
import os

from src.match_circles import match_circles_with_normalization
from src.detect_circles import detect_circles

def merge_image(src_img, dst_img, H, channel='B'):
    """
    根据单应矩阵H，将src_img的channel拼接融合至dst_img。

    返回: merged_img
    """
    h, w = dst_img.shape[:2]
    src_img_warp = cv2.warpPerspective(src_img, H, (w, h))
    if channel == 'B':
        merged = cv2.merge((src_img_warp, dst_img[:, :, 1], dst_img[:, :, 2]))
    elif channel == 'G':
        merged = cv2.merge((dst_img[:, :, 0], src_img_warp, dst_img[:, :, 2]))
    elif channel == 'R':
        merged = cv2.merge((dst_img[:, :, 0], dst_img[:, :, 1], src_img_warp))
    return merged



def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    img_path_b = os.path.join(script_dir, '0006.png')
    img_path_gr = os.path.join(script_dir, '0007.png')
    print(f'读取图片: {img_path_b}\n读取图片: {img_path_gr}')
    img_b = cv2.imread(img_path_b, cv2.IMREAD_GRAYSCALE)
    img_gr = cv2.imread(img_path_gr, cv2.IMREAD_COLOR)
    H, merged = merge_image(img_b, img_gr, channel='B')
    if H is None or merged is None:
        print('处理失败')
        return
    print('单应矩阵 H:')
    print(H)
    cv2.imshow('合成结果(B=img1, RG=img2变换)', merged)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # cv2.imwrite('merged_homography.png', merged)
    # print('合成图像已保存为 merged_homography.png')

if __name__ == '__main__':
    main()
