import cv2
import os
import numpy as np
# 定义文件路径和文件名
# 同目录下的文件夹
OLD_DIR = "/Users/caizhenghua/Desktop/deep-learning-for-image-processing/data_set/dataset_row/normal/"
# NEW_DIR = "/Users/caizhenghua/Desktop/deep-learning-for-image-processing/data_set/dataset_row/NEW/"
NEW_DIR = "/Users/caizhenghua/Desktop/deep-learning-for-image-processing/data_set/dataset_new/normal/"

# 确保保存结果的文件夹存在
if not os.path.exists(NEW_DIR):
    os.makedirs(NEW_DIR)
# if not os.path.exists(NEW2_DIR):
#     os.makedirs(NEW2_DIR)
# 获取OLD文件夹中所有的PNG文件名
files = os.listdir(OLD_DIR)
# files = [f for f in files if f.endswith('.png')]

# 遍历所有的PNG文件
for f in files:
    # 读取f并截取中间部分60%
    img = cv2.imread(OLD_DIR + f)
    h, w, _ = img.shape
    img = img[int(h * 0.2):int(h * 0.8), int(w * 0.2):int(w * 0.8), :]
    h, w, _ = img.shape
    # 保存
    cv2.imwrite(NEW_DIR + f, img)

# 读取NEW文件夹中所有的PNG文件名
files = os.listdir(NEW_DIR)
# files = [f for f in files if f.endswith('.png')]

# 遍历所有的PNG文件
# for f in files:
#     img = cv2.imread(NEW_DIR + f)
#
#     # 二值化
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     ret, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_TRIANGLE)
#
#     # 根据binary图像，画出一个最小的矩形，将其作为裁剪区域
#     contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#     cnt = sorted(contours, key=cv2.contourArea, reverse=True)[0]
#     x, y, w, h = cv2.boundingRect(cnt)
#
#     center = (int(x + w / 2), int(y + h / 2))
#
#     # 矩形resize为r倍,设置太大会导致出现负数
#     r = 1.0
#     w = int(w * r)
#     h = int(h * r)
#     x = int(center[0] - w / 2)
#     y = int(center[1] - h / 2)
#
#     # 读出彩色图像
#     img_color = cv2.imread(NEW_DIR + f)
#
#     # # 根据矩形裁剪彩色图像
#     img_color = img_color[y:y + h, x:x + w, :]
#
#     # 保存结果到NEW-2文件夹
#     cv2.imwrite(NEW2_DIR + f, img_color)




