from .detect_circles import detect_circles

from scipy.spatial import cKDTree
from sklearn.cluster import KMeans

import numpy as np
import cv2
import os

# 计算每对点之间的距离矩阵
def compute_distance_matrix(points):
    n = len(points)
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                dist_matrix[i, j] = np.linalg.norm(points[i] - points[j])
    return dist_matrix

# 计算每对点之间的角度矩阵（相对于x轴）
def compute_angle_matrix(points):
    n = len(points)
    angle_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                vec = points[j] - points[i]
                angle_matrix[i, j] = np.arctan2(vec[1], vec[0])
    return angle_matrix

# 计算相对角度差矩阵
def compute_relative_angles(angle_matrix):
    n = angle_matrix.shape[0]
    rel_angles = np.zeros((n, n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                for k in range(n):
                    if k != i and k != j:
                        # 计算两个向量之间的最小角度差
                        angle_diff = (angle_matrix[i, k] - angle_matrix[i, j]) % (2 * np.pi)
                        angle_diff = min(angle_diff, 2 * np.pi - angle_diff)
                        rel_angles[i, j, k] = angle_diff
    return rel_angles

# 计算两个点集之间的匹配分数
def compute_match_score(pts1, pts2, dist_threshold=0.1, angle_threshold=0.2):
    if len(pts1) != len(pts2):
        return float('-inf')
        
    # 计算距离和角度矩阵
    dist1 = compute_distance_matrix(pts1)
    dist2 = compute_distance_matrix(pts2)
    angle1 = compute_angle_matrix(pts1)
    angle2 = compute_angle_matrix(pts2)
    
    # 计算相对角度
    rel_angles1 = compute_relative_angles(angle1)
    rel_angles2 = compute_relative_angles(angle2)
    
    # 尝试所有可能的参考点对
    best_score = float('-inf')
    best_mapping = None
    
    # 选择第一个点集中的一个参考点
    for ref1 in range(len(pts1)):
        # 在第二个点集中尝试所有可能的参考点
        for ref2 in range(len(pts2)):
            # 尝试匹配其他点
            score = 0
            mapping = {ref1: ref2}
            
            # 对于第一个点集中的每个其他点
            for i in range(len(pts1)):
                if i == ref1:
                    continue
                    
                best_match = -1
                best_match_score = float('inf')
                
                # 在第二个点集中寻找最佳匹配
                for j in range(len(pts2)):
                    if j == ref2 or j in mapping.values():
                        continue
                        
                    # 计算距离比率
                    dist_ratio1 = dist1[ref1, i] / dist1[ref1, :].mean()
                    dist_ratio2 = dist2[ref2, j] / dist2[ref2, :].mean()
                    dist_score = abs(dist_ratio1 - dist_ratio2)
                    
                    # 计算相对角度相似性
                    angle_score = 0
                    for k, m in mapping.items():
                        if k != ref1 and i != k:
                            angle_diff1 = rel_angles1[ref1, k, i]
                            angle_diff2 = rel_angles2[ref2, m, j]
                            angle_score += abs(angle_diff1 - angle_diff2)
                    
                    total_score = dist_score + angle_score
                    
                    if total_score < best_match_score:
                        best_match_score = total_score
                        best_match = j
                
                if best_match != -1 and best_match_score < 1.0:  # 调整阈值
                    mapping[i] = best_match
                    score += 1 - best_match_score
            
            if score > best_score:
                best_score = score
                best_mapping = mapping
    
    return best_score, best_mapping


def match_big_circles(circles1, circles2, num_big=5):
    """
    输入两组圆（x,y,r）,使用大圆进行匹配，能够适应图像旋转的情况。
    返回: matched_pts1, matched_pts2
    """
    # 获取大圆
    big_circles1 = sorted(circles1, key=lambda c: c[2], reverse=True)[:num_big]
    big_circles2 = sorted(circles2, key=lambda c: c[2], reverse=True)[:num_big]
    
    # 转换为numpy数组方便计算
    circles1_arr = np.array(big_circles1)[:, :2]  # 只需要x,y坐标
    circles2_arr = np.array(big_circles2)[:, :2]

    # 计算匹配分数和映射关系
    score, mapping = compute_match_score(circles1_arr, circles2_arr)
    
    if not mapping or len(mapping) < 3:  # 至少需要3个匹配点
        print('无法找到足够的匹配点')
        return None, None
    
    # 根据映射关系返回匹配的点对
    matched_circles1 = []
    matched_circles2 = []
    for i, j in mapping.items():
        matched_circles1.append(big_circles1[i])
        matched_circles2.append(big_circles2[j])
    
    return matched_circles1, matched_circles2


def match_all_circles(circles1, circles2):
    """
    输入两组圆（x,y,r）,先用大圆配对算单应矩阵，再对所有圆做最小距离配对。
    返回: matched_circles1, matched_circles2, dist
    """

    # 1. 大圆配对
    circles1_big, circles2_big = match_big_circles(circles1, circles2)
    if circles1_big is None or circles2_big is None:
        print('大圆配对失败')
        return None, None, None
    
    pts1_big = np.array([[c[0], c[1]] for c in circles1_big], dtype=np.float32)
    pts2_big = np.array([[c[0], c[1]] for c in circles2_big], dtype=np.float32)
    H, mask = cv2.findHomography(pts2_big, pts1_big, cv2.RANSAC, 5.0)
    
    # 4. 所有圆心坐标
    pts1 = np.array([[c[0], c[1]] for c in circles1], dtype=np.float32)
    pts2 = np.array([[c[0], c[1]] for c in circles2], dtype=np.float32)
    # 5. 用H将图2所有圆心变换到图1坐标系
    pts2_homo = np.concatenate([pts2, np.ones((pts2.shape[0], 1))], axis=1)  # (N,3)
    pts2_warped = (H @ pts2_homo.T).T
    pts2_warped = pts2_warped[:, :2] / pts2_warped[:, 2:]

    # 6. 最近邻一一配对
    tree = cKDTree(pts1)
    dist, idx = tree.query(pts2_warped)
    
    # 使用K-means将距离分为两类（有效匹配和异常值）
    dist_reshaped = dist.reshape(-1, 1)
    kmeans = KMeans(n_clusters=2, random_state=0).fit(dist_reshaped)
    clusters = kmeans.labels_
    # 找出距离较小的类别
    if kmeans.cluster_centers_[0] < kmeans.cluster_centers_[1]:
        valid_cluster = 0
    else:
        valid_cluster = 1
    # 创建有效掩码
    valid_mask = (clusters == valid_cluster)
    # 计算最大距离阈值（可以加上一些余量）
    max_distance = np.max(dist[valid_mask]) * 1.1  # 增加10%的余量

    # 筛选出距离小于阈值的匹配对
    valid_mask = dist < max_distance
    
    # 确保单一匹配：
    # 1. 创建已匹配标记
    matched_mask1 = np.zeros(len(pts1), dtype=bool)  # 第一个点集的匹配标记
    matched_mask2 = np.zeros(len(pts2), dtype=bool)  # 第二个点集的匹配标记
    
    # 2. 只保留有效的最近邻匹配
    final_valid_mask = np.zeros(len(pts2), dtype=bool)
    for i in range(len(pts2)):
        if valid_mask[i] and not matched_mask1[idx[i]] and not matched_mask2[i]:
            # 如果距离小于阈值且两个点都还没有被匹配，就接受这个匹配
            final_valid_mask[i] = True
            matched_mask1[idx[i]] = True
            matched_mask2[i] = True
    
    # 3. 获取最终匹配结果
    matched_circles1 = np.array(circles1)[idx[final_valid_mask]]
    matched_circles2 = np.array(circles2)[final_valid_mask]
    valid_dist = dist[final_valid_mask]

    
    return matched_circles1, matched_circles2, valid_dist


def match_all_circles_normalized(circles1, circles2):

    def normalize_scale(pts):
        """
        将点集归一化到中心为0、尺度为空间相邻点平均间距的空间。
        返回归一化点、中心、尺度
        """
        if len(pts) < 2:
            return pts, np.zeros(2) if len(pts) == 0 else np.mean(pts, axis=0), 1.0
        # 使用KD树找到每个点的最近邻点
        tree = cKDTree(pts)
        # 查询每个点的最近邻点（不包括自己）
        distances, _ = tree.query(pts, k=2)  # 第一个最近邻是自己，取第二个
        # 获取所有点到其最近邻点的距离
        neighbor_distances = distances[:, 1]
        # 使用相邻点平均距离作为尺度因子
        scale = np.mean(neighbor_distances)
        # 防止除零错误
        scale = scale if scale > 1e-10 else 1.0
        return scale

    if len(circles1) == 0 or len(circles2) == 0:
        print('圆检测失败')
        return None, None, None

    # 1. 提取圆心和半径
    pts1 = np.array([[c[0], c[1]] for c in circles1], dtype=np.float32)
    r1 = np.array([c[2] for c in circles1], dtype=np.float32)
    pts2 = np.array([[c[0], c[1]] for c in circles2], dtype=np.float32)
    r2 = np.array([c[2] for c in circles2], dtype=np.float32)

    # 2. 归一化
    scale1 = normalize_scale(pts1)
    scale2 = normalize_scale(pts2)
    circles1_norm = circles1 / scale1
    circles2_norm = circles2 / scale2

    # 4. 用归一化后的圆参数进行匹配
    matched_circles1_norm, matched_circles2_norm, dist  = match_all_circles(circles1_norm, circles2_norm)
    matched_circles1 = matched_circles1_norm * scale1
    matched_circles2 = matched_circles2_norm * scale2

    return matched_circles1, matched_circles2, dist
    

