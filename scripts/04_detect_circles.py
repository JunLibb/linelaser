import cv2
import numpy as np
import os

from ..src.detect_circles import find_circles
from ..src.visualization import plot_circles

workpath = './output'
imagepath = './output/merge'
outputpath = './output/circles'

if not os.path.exists(outputpath):
    os.makedirs(outputpath)

# 获取所有png图片并排序
img_files = sorted([f for f in os.listdir(imagepath) if f.lower().endswith('.png')])
# 每2张为一组
for idx in range(0, len(img_files) - 1, 2):
    img_path = os.path.join(imagepath, img_files[idx + 1])
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    circles = find_circles(img, min_area=40, roundness_threshold=0.8)
    if len(circles) == 0:
        print('未检测到圆')
    print(f'共检测到 {len(circles)} 个圆')
    color_img = plot_circles(img, circles, counting=True)
    cv2.imwrite(os.path.join(outputpath, img_files[idx + 1]), color_img)