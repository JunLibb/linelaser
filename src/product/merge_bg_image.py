"""
输入两张图片路径，自动配对圆心，计算单应矩阵，并将img2的RG通道变换后与img1蓝通道合成。
合并的逻辑是根据口扫设备定制的
"""
import cv2  
import numpy as np  
import os   

from ..merge_channels import merge_image
from ..detect_circles import detect_circles
from ..match_circles import match_circles_with_normalization


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
    matched_circles1, matched_circles2, _ = match_circles_with_normalization(circles1, circles2)
    matched_pts1 =  np.array([[c[0], c[1]] for c in matched_circles1])
    matched_pts2 =  np.array([[c[0], c[1]] for c in matched_circles2])
    if matched_pts1 is None or matched_pts2 is None:
        print('圆心配对失败')
        return None
    H, mask = cv2.findHomography(matched_pts1, matched_pts2, cv2.RANSAC, 5.0)
    if H is None:
        return None

    return H, matched_pts1, matched_pts2   

def merge_bgimage(img_b, img_gr):
    H, _, _ = getHomography_bgimage(img_b, img_gr)
    if H is None:
        return None
    return merge_image(img_b, img_gr, H, channel='B')   