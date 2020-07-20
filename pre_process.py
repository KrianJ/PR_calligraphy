# -*- coding:utf-8 -*-
__Author__ = "KrianJ wj_19"
__Time__ = "2020/6/11 11:20"
__doc__ = """ 对单字图像进行预处理
1. 归一化：将图像归一到相同大小.
    method_1:先对原图像提取特征，然后再对特征做归一化，不会丢失信息
    method_2:先将图像归一化，再做特征提取
2. 二值化：转换成灰度图
3. 向量化：转换成向量(csv存储)"""

import os
import numpy as np
import cv2


def threhold(o_img):
    """图像滤波,阈值处理"""
    def isBlackchar(img):
        return np.sum(img > 100) > np.sum(img < 100)

    length = np.min(o_img.shape)
    length = length if length%2==1 else length-1
    img = o_img
    # 中值滤波
    img = cv2.blur(img, (3, 3))
    img = cv2.medianBlur(img, 7)
    # 均值滤波（自适应阈值）
    img = cv2.blur(img, (3, 3))
    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, length, 2)
    # img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, length, 2)

    if not isBlackchar(img):
        img = 255 - img
    return img


def patch_read_gray(img_paths):
    """批量读入图像路径，返回二值化后的灰度图像(list)和图像名"""
    imgs = []
    names = []
    for ipath in img_paths:
        gray_img = cv2.imdecode(np.fromfile(ipath, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        # 过滤切分过小的无效图像，过小的图像会导致cv2灰度转换报错
        if gray_img.shape[0] > 10 and gray_img.shape[1] > 10:
            resize_gray = cv2.resize(gray_img, (84, 84))
            img = threhold(resize_gray)
            imgs.append(img)
            name = ipath.split('\\')[-1]
            names.append(name)

    return imgs, names


def patch_read_rgb(img_paths):
    """批量读取路径，返回rbg图像和图像名"""
    imgs = []
    names = []
    for ipath in img_paths:
        rgb_img = cv2.imread(ipath)
        # 过滤切分过小的无效图像
        if rgb_img.shape[0] > 10 and rgb_img.shape[1] > 10:
            imgs.append(rgb_img)
            name = ipath.split('\\')[-1]
            names.append(name)
    return imgs, names


def pre_process():
    # 获取所有图像文件路径
    root = r'D:\Pyproject\PR_calligraphy\characters'
    img_paths = []
    for folder in os.listdir(root):
        f_path = root + '\\' + folder
        f_img_path = [f_path + '\\' + img_name for img_name in os.listdir(f_path) if '.jpg' in img_name]
        img_paths += f_img_path

    # 二值化(转换成灰度图)
    try:
        imgs, names = patch_read_gray(img_paths)
    except AssertionError:
        print('Assertion Error!')
    finally:
        return imgs, names


if __name__ == '__main__':
    imgs, names = pre_process()
    print(len(imgs), len(names), len(set(names)))
