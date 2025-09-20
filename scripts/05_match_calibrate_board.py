import cv2
import numpy as np
import os
import json
from fscan2.match import match_all_circles_normalized
from fscan2.detect_circles import detect_circles
from fscan2.visualization import plot_matched_circles


workpath = './output'
imagepath = './output/merge'
calibpath = './output/calib'

if not os.path.exists(calibpath):
    os.makedirs(calibpath)

# 读取标定板阵列
with open(workpath + '/circle_array.json', 'r') as f:
    calibrate_circle_array = np.array(json.load(f))

# 获取所有png图片并排序
img_files = sorted([f for f in os.listdir(imagepath) if f.lower().endswith('.png')])
# 每2张为一组
for idx in range(0, len(img_files) - 1, 2):
    img_path = os.path.join(imagepath, img_files[idx + 1])
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    circles = detect_circles(img)
    matched_circles1, matched_circles2, dist = match_all_circles_normalized(circles, calibrate_circle_array)

    vis = plot_matched_circles(img, matched_circles1, matched_circles2)
    # cv2.imshow('Matched Circles', vis)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    cv2.imwrite(os.path.join(calibpath, img_files[idx + 1]), vis)