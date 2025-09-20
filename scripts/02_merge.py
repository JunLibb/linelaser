import os
import cv2
import numpy as np
from fscan2.merge import getHomography_bgimage, stitch_images

# 输入和输出路径
input_dir = './output/rgb'
output_dir = './output/merge'
output_dir_stitch = './output'

# 创建输出文件夹
os.makedirs(output_dir, exist_ok=True)

# 获取所有png图片并排序
img_files = sorted([f for f in os.listdir(input_dir) if f.lower().endswith('.png')])

matched_pts1 = []
matched_pts2 = []

# 每4张为一组
for idx in range(0, len(img_files) - 1, 4):
    img1_path = os.path.join(input_dir, img_files[idx])
    img2_path = os.path.join(input_dir, img_files[idx + 1])
    img3_path = os.path.join(input_dir, img_files[idx + 2])
    img4_path = os.path.join(input_dir, img_files[idx + 3])

    # 用OpenCV读取图片
    img3 = cv2.imread(img3_path, cv2.IMREAD_GRAYSCALE)
    img4 = cv2.imread(img4_path, cv2.IMREAD_COLOR)

    # 合成
    H, pts1, pts2 = getHomography_bgimage(img3, img4)
    if H is None:
        print('单应矩阵计算失败')
        continue
    matched_pts1.extend(pts1)
    matched_pts2.extend(pts2)

matched_pts1 = np.array(matched_pts1)
matched_pts2 = np.array(matched_pts2)
H, mask = cv2.findHomography(matched_pts1, matched_pts2, cv2.RANSAC, 5.0)

# 每4张为一组
for idx in range(0, len(img_files) - 1, 4):
    img1_path = os.path.join(input_dir, img_files[idx])
    img2_path = os.path.join(input_dir, img_files[idx + 1])
    img3_path = os.path.join(input_dir, img_files[idx + 2])
    img4_path = os.path.join(input_dir, img_files[idx + 3])

    # 用OpenCV读取图片
    img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(img2_path, cv2.IMREAD_COLOR)
    img3 = cv2.imread(img3_path, cv2.IMREAD_GRAYSCALE)
    img4 = cv2.imread(img4_path, cv2.IMREAD_COLOR)

    # 合成
    merged1 = stitch_images(img1, img2, H)
    merged2 = stitch_images(img3, img4, H)

    # 用OpenCV保存
    out_path = os.path.join(output_dir, f"{idx//2:04d}.png")
    cv2.imwrite(out_path, merged1)
    out_path = os.path.join(output_dir, f"{idx//2+1:04d}.png")
    cv2.imwrite(out_path, merged2)

# 保存单应矩阵H到文件
homography_path = os.path.join(output_dir_stitch, 'homography_matrix.npy')
np.save(homography_path, H)
print(f"单应矩阵已保存到: {homography_path}")
print("之后可以使用以下代码加载单应矩阵:")
print(f"H_loaded = np.load('{homography_path}')")
print("批量合成完成！")