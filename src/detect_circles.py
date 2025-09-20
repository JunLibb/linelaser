
import cv2
import numpy as np
from typing import List, Optional, Tuple
import os

def find_subpixel_edges(image: np.ndarray, 
                       blur_kernel: tuple = (5, 5),
                       canny_thresholds: tuple = (50, 150),
                       win_size: int = 3,
                       max_iter: int = 40,
                       epsilon: float = 0.001) -> np.ndarray:
    """
    使用亚像素精度检测图像边缘
    
    参数:
        image: 输入图像 (BGR或灰度)
        blur_kernel: 高斯模糊核大小
        canny_thresholds: Canny边缘检测的阈值 (低, 高)
        win_size: 亚像素计算的窗口大小
        max_iter: 最大迭代次数
        epsilon: 迭代终止精度
        
    返回:
        edge_points: 亚像素级边缘点坐标 (N, 2)
    """
    # 转换为灰度图
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # 检查图像大小是否足够
    min_dim = win_size * 2 + 5
    if gray.shape[0] < min_dim or gray.shape[1] < min_dim:
        return np.array([], dtype=np.float32).reshape(0, 2)
    
    # 高斯模糊降噪
    blurred = cv2.GaussianBlur(gray, blur_kernel, 1.5)
    
    # Canny边缘检测
    edges = cv2.Canny(blurred, canny_thresholds[0], canny_thresholds[1])
    
    # 查找轮廓
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    # 准备亚像素角点检测参数
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, max_iter, epsilon)
    
    all_edge_points = []
    
    for contour in contours:
        if len(contour) < 5:  # 至少需要5个点
            continue
            
        # 转换为浮点型
        contour = np.float32(contour).reshape(-1, 2)
        
        # 检查轮廓点是否在图像边界内
        h, w = gray.shape
        valid_mask = ((contour[:, 0] >= win_size) & 
                     (contour[:, 0] < w - win_size) &
                     (contour[:, 1] >= win_size) & 
                     (contour[:, 1] < h - win_size))
        
        if not np.any(valid_mask):
            continue
            
        # 只保留有效点
        valid_contour = contour[valid_mask]
        
        try:
            # 亚像素精确化
            cv2.cornerSubPix(
                gray, 
                valid_contour,
                (win_size, win_size),
                (-1, -1),  # 零区域，表示没有
                criteria
            )
            all_edge_points.append(valid_contour)
        except:
            # 如果亚像素处理失败，跳过这个轮廓
            continue
    
    if not all_edge_points:
        return np.array([], dtype=np.float32).reshape(0, 2)
        
    return np.vstack(all_edge_points)


def fit_ellipse_subpixel(image: np.ndarray, 
                        min_points: int = 10,
                        roundness_threshold: float = 0.7) -> Optional[Tuple[tuple, tuple, float]]:
    """
    使用亚像素边缘拟合椭圆
    
    参数:
        image: 输入图像
        min_points: 拟合椭圆所需的最小点数
        roundness_threshold: 圆形度阈值
        
    返回:
        ((center_x, center_y), (major_axis, minor_axis), angle) 或 None
    """
    # 获取亚像素边缘点
    edge_points = find_subpixel_edges(image)
    
    if len(edge_points) < min_points:
        return None
    
    # 拟合椭圆
    ellipse = cv2.fitEllipse(edge_points)
    (center, axes, angle) = ellipse
    
    # 计算圆形度
    major_axis, minor_axis = max(axes), min(axes)
    area = np.pi * major_axis * minor_axis / 4.0
    perimeter = np.pi * (1.5 * (major_axis + minor_axis) - np.sqrt(major_axis * minor_axis))
    roundness = (4 * np.pi * area) / (perimeter * perimeter)
    
    if roundness < roundness_threshold:
        return None
        
    return (center, axes, angle, roundness)

def detect_circles_with_subpixel(image: np.ndarray,
                               min_area: int = 30,
                               roundness_threshold: float = 0.7) -> List[np.ndarray]:
    """
    检测图像中的圆形（使用亚像素精度）
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # 二值化
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # 查找轮廓
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    circles = []
    min_roi_size = 20  # 最小ROI大小
    
    for contour in contours:
        if len(contour) < 5:  # 至少需要5个点才能拟合椭圆
            continue
            
        area = cv2.contourArea(contour)
        if area < min_area:
            continue
            
        # 获取轮廓的边界框
        x, y, w, h = cv2.boundingRect(contour)
        
        # 确保ROI足够大
        if w < min_roi_size or h < min_roi_size:
            expand = max(min_roi_size - w, min_roi_size - h) // 2 + 1
        else:
            expand = 5  # 默认扩展5个像素
            
        # 计算扩展后的ROI，确保不超出图像边界
        x1 = max(0, x - expand)
        y1 = max(0, y - expand)
        x2 = min(gray.shape[1], x + w + expand)
        y2 = min(gray.shape[0], y + h + expand)
        
        # 提取ROI
        roi = gray[y1:y2, x1:x2]
        
        # 确保ROI足够大
        if roi.size == 0 or roi.shape[0] < min_roi_size or roi.shape[1] < min_roi_size:
            continue
        
        # 使用亚像素边缘拟合椭圆
        result = fit_ellipse_subpixel(roi, roundness_threshold=roundness_threshold)
        
        if result is not None:
            (center, axes, angle, roundness) = result
            # 将ROI坐标转换回原图坐标
            center = (center[0] + x1, center[1] + y1)
            radius = (axes[0] + axes[1]) / 4.0  # 平均半径
            
            circles.append(np.array([center[0], center[1], radius, roundness]))
    
    return circles
    

def detect_circles(gray_img, min_area=50, min_circularity=0.95, min_radius=3, max_radius=100):
    """
    检测灰度图中的所有黑色圆，返回亚像素圆心坐标和半径
    
    参数:
        gray_img: 灰度图像
        min_area: 最小面积阈值
        min_circularity: 最小圆度阈值
        min_radius: 最小半径
        max_radius: 最大半径
    """
    # 1. 二值化，提取黑色区域
    _, thresh = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 2. 查找轮廓
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    circles = []
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue

        # 椭圆拟合
        ellipse = cv2.fitEllipse(cnt)
        (cx, cy), (minor_axis, major_axis), angle = ellipse
        radius = (minor_axis + major_axis) / 4.0  # 平均半径
        
        # 计算拟合椭圆的面积
        ellipse_area = np.pi * (major_axis/2) * (minor_axis/2)
        circularity = area / ellipse_area
        if circularity < min_circularity:
            continue
            
        if radius < min_radius or radius > max_radius:
            continue
            
        circles.append([float(cx), float(cy), float(radius), float(circularity)])
        
    return circles

def find_circles(img_binary: np.ndarray, min_area: int = 30, roundness_threshold: float = 0.7) -> List[np.ndarray]:
    """
    在二值图像中检测圆形并返回其属性
    
    参数:
        img_binary: 输入的二值图像（numpy数组）
        min_area: 最小轮廓面积阈值，小于此值的轮廓将被忽略
        roundness_threshold: 圆形度阈值(0-1)，用于判断轮廓是否为圆形
        
    返回:
        包含检测到圆形的列表，每个元素是一个numpy数组，格式为[圆心x坐标, 圆心y坐标, 半径, 圆形度]
    """
    # 查找轮廓
    _, thresh = cv2.threshold(img_binary, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 2. 查找轮廓
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    circles = []
    
    for contour in contours:
        # 跳过点数不足的轮廓（需要至少5个点才能拟合椭圆）
        if len(contour) < 5:
            continue
            
        # 计算轮廓面积
        area = cv2.contourArea(contour)
        # 跳过面积过小的轮廓
        if area < min_area:
            continue
            
        # 拟合椭圆
        ellipse = cv2.fitEllipse(contour)
        # 解构椭圆参数：中心点坐标、轴长、旋转角度
        (center_x, center_y), (major_axis, minor_axis), angle = ellipse
        
        # 计算轮廓周长
        perimeter = cv2.arcLength(contour, True)
        # 计算圆形度：(4π × 面积) / (周长²)
        if perimeter > 0:
            roundness = (4 * np.pi * area) / (perimeter * perimeter)
        else:
            roundness = 0
            
        # 跳过圆形度过低的轮廓
        if roundness < roundness_threshold:
            continue
            
        # 计算半径（取长短轴的平均值）
        radius = (major_axis + minor_axis) / 4.0
        
        # 将圆形信息添加到结果列表 [x坐标, y坐标, 半径, 圆形度]
        circles.append(np.array([center_x, center_y, radius, roundness]))
    
    return circles



