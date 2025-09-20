"""
输入两张图片路径，自动配对圆心，计算单应矩阵，并将img2的RG通道变换后与img1蓝通道合成。
合并的逻辑是根据口扫设备定制的
"""
import cv2
import numpy as np
import os

from .match import match_all_circles
from .detect_circles import detect_circles

def stitch_images(img1, img2_color, H):
    """
    根据单应矩阵H，将img1和img2_color拼接融合。
    img1: 灰度图，img2_color: 彩色图
    返回: merged_img
    """
    h, w = img2_color.shape[:2]
    img1_b_warp = cv2.warpPerspective(img1, H, (w, h))
    merged = np.zeros((h, w, 3), dtype=np.uint8)
    merged[..., 0] = img1_b_warp  # B通道
    merged[..., 1] = img2_color[..., 1]  # G通道
    # merged[..., 2] = img2_color[..., 2]  # R通道（如有需要可取消注释）
    return merged

def getHomography_bgimage(img_b, img_gr):
    """
    输入两张图片（img_b为灰度图，img_gr为彩色图），自动配对圆心，计算单应矩阵，并将img_gr的G通道变换后与img_b蓝通道合成。
    返回: H
    """
    img1 = img_b  # 灰度图
    img2_color = img_gr  # 彩色图
    img2 = img2_color[:, :, 1]  # 只保留绿色通道

    if img1 is None or img2 is None:
        print('图片输入无效')
        return None

    circles1 = detect_circles(img1)
    circles2 = detect_circles(img2)
    matched_circles1, matched_circles2, _ = match_all_circles(circles1, circles2)
    matched_pts1 =  np.array([[c[0], c[1]] for c in matched_circles1])
    matched_pts2 =  np.array([[c[0], c[1]] for c in matched_circles2])
    if matched_pts1 is None or matched_pts2 is None:
        print('圆心配对失败')
        return None
    H, mask = cv2.findHomography(matched_pts1, matched_pts2, cv2.RANSAC, 5.0)
    if H is None:
        return None

    return H, matched_pts1, matched_pts2


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    img_path_b = os.path.join(script_dir, '0006.png')
    img_path_gr = os.path.join(script_dir, '0007.png')
    print(f'读取图片: {img_path_b}\n读取图片: {img_path_gr}')
    img_b = cv2.imread(img_path_b, cv2.IMREAD_GRAYSCALE)
    img_gr = cv2.imread(img_path_gr, cv2.IMREAD_COLOR)
    H, merged = merge_bgimage(img_b, img_gr)
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
