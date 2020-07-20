# -*- coding:utf-8 -*-
__Author__ = "KrianJ wj_19"
__Time__ = "2020/6/11 16:44"
__doc__ = """ 对二值化图像进行特征提取"""

import numpy as np
import math
import cv2
import pandas as pd
import sys
from pre_process import pre_process


def guass_smooth(gray_img):
    """(2k+1)*(2k+1)高斯滤波器
    滤波器元素计算公式：
    H[i,j] = 1/(2*pi*sigma^2) * exp(-1/2sigma^2 * [(i-k-1)^2)+(j-k-1)^2]"""
    # 生成高斯滤波器
    guassian = np.zeros([5, 5])
    sigma = 1.9
    guass_sum = 0
    for i in range(5):
        for j in range(5):
            guassian[i, j] = math.exp((-1/(2*sigma**2))*(np.square(i-3) + np.square(j-3))) / (2*math.pi*sigma**2)
            guass_sum += guassian[i, j]

    # 归一化滤波器
    guassian = guassian / guass_sum

    # 高斯滤波
    w, h = gray_img.shape
    new_gray = np.zeros([w-5, h-5])
    for i in range(w-5):
        for j in range(h-5):
            new_gray[i, j] = np.sum(gray_img[i:i+5, j:j+5] * guassian)

    return new_gray


def gradient(new_gray):
    """
    计算梯度幅值：通过计算梯度判断像素处于字体内部还是外部（边缘检测）
    :type: image which after smooth
    :rtype:
        dx: gradient in the x direction
        dy: gradient in the y direction
        M: gradient magnitude
        theta: gradient direction
    """
    W, H = new_gray.shape
    dx = np.zeros([W - 1, H - 1])
    dy = np.zeros([W - 1, H - 1])
    M = np.zeros([W - 1, H - 1])
    theta = np.zeros([W - 1, H - 1])

    for i in range(W - 1):
        for j in range(H - 1):
            dx[i, j] = new_gray[i + 1, j] - new_gray[i, j]
            dy[i, j] = new_gray[i, j + 1] - new_gray[i, j]
            # 图像梯度幅值作为图像强度值
            M[i, j] = np.sqrt(np.square(dx[i, j]) + np.square(dy[i, j]))
            # 计算  θ - artan(dx/dy)
            theta[i, j] = math.atan(dx[i, j] / (dy[i, j] + 0.000000001))

    return dx, dy, M, theta


def NMS(M, dx, dy):
    """非极大值抑制"""
    d = np.copy(M)
    W, H = M.shape
    NMS = np.copy(d)
    NMS[0, :] = NMS[W - 1, :] = NMS[:, 0] = NMS[:, H - 1] = 0

    for i in range(1, W - 1):
        for j in range(1, H - 1):

            # 如果当前梯度为0，该点就不是边缘点
            if M[i, j] == 0:
                NMS[i, j] = 0

            else:
                gradX = dx[i, j]  # 当前点 x 方向导数
                gradY = dy[i, j]  # 当前点 y 方向导数
                gradTemp = d[i, j]  # 当前梯度点

                # 如果 y 方向梯度值比较大，说明导数方向趋向于 y 分量
                if np.abs(gradY) > np.abs(gradX):
                    weight = np.abs(gradX) / np.abs(gradY)  # 权重
                    grad2 = d[i - 1, j]
                    grad4 = d[i + 1, j]

                    # 如果 x, y 方向导数符号一致
                    # 像素点位置关系
                    # g1 g2
                    #    c
                    #    g4 g3
                    if gradX * gradY > 0:
                        grad1 = d[i - 1, j - 1]
                        grad3 = d[i + 1, j + 1]

                    # 如果 x，y 方向导数符号相反
                    # 像素点位置关系
                    #    g2 g1
                    #    c
                    # g3 g4
                    else:
                        grad1 = d[i - 1, j + 1]
                        grad3 = d[i + 1, j - 1]

                # 如果 x 方向梯度值比较大
                else:
                    weight = np.abs(gradY) / np.abs(gradX)
                    grad2 = d[i, j - 1]
                    grad4 = d[i, j + 1]

                    # 如果 x, y 方向导数符号一致
                    # 像素点位置关系
                    #      g3
                    # g2 c g4
                    # g1
                    if gradX * gradY > 0:

                        grad1 = d[i + 1, j - 1]
                        grad3 = d[i - 1, j + 1]

                    # 如果 x，y 方向导数符号相反
                    # 像素点位置关系
                    # g1
                    # g2 c g4
                    #      g3
                    else:
                        grad1 = d[i - 1, j - 1]
                        grad3 = d[i + 1, j + 1]

                # 利用 grad1-grad4 对梯度进行插值
                gradTemp1 = weight * grad1 + (1 - weight) * grad2
                gradTemp2 = weight * grad3 + (1 - weight) * grad4

                # 当前像素的梯度是局部的最大值，可能是边缘点
                if gradTemp >= gradTemp1 and gradTemp >= gradTemp2:
                    NMS[i, j] = gradTemp

                else:
                    # 不可能是边缘点
                    NMS[i, j] = 0

    return NMS


def double_threhold(NMS):
    """双阈值边缘选取"""
    W, H = NMS.shape
    DT = np.zeros([W, H])
    # 定义高低阈值
    TL = 0.1 * np.max(NMS)
    TH = 0.3 * np.max(NMS)

    for i in range(1, W - 1):
        for j in range(1, H - 1):
            # 双阈值选取
            if (NMS[i, j] < TL):
                DT[i, j] = 0
            elif (NMS[i, j] > TH):
                DT[i, j] = 1
            # 连接
            elif (NMS[i - 1, j - 1:j + 1] < TH).any() \
                    or (NMS[i + 1, j - 1:j + 1].any()
                        or (NMS[i, [j - 1, j + 1]] < TH).any()):
                DT[i, j] = 1
    return DT


def get_csv():
    # 预处理
    imgs, img_names = pre_process()
    canny_imgs = []
    # 手写canny边缘检测
    # new_gray = guass_smooth(gray_img)
    # dx, dy, M, theta = gradient(new_gray)
    # nms = NMS(M, dx, dy)
    # dt = double_threhold(nms)
    # dt[dt == 1] = 255
    # plt.imshow(dt)
    # plt.show()
    # canny算法边缘检测
    """cv2的canny算法"""
    for img in imgs:
        img = cv2.GaussianBlur(img, (3, 3), 0)
        canny = cv2.Canny(img, 50, 150)
        canny_imgs.append(canny.ravel())
        # print(canny_imgs[0])

    """存储带(无)标签样本：通过二值化 & canny边缘检测"""
    label_df = pd.read_excel('after_process\character.xlsx', sheet_name='Sheet1')             # 标签名，文件名   13128唯一样本，2013个汉字
    img_df = pd.DataFrame({
        'file_name': img_names,
        'img': canny_imgs
    })                                                              # 图像，文件名     13462(含重复)样本
    img_df = img_df.drop_duplicates('file_name', keep='first')                                # 删除img_df文件名重复项  13430个唯一样本

    # 让数组全显示，由于csv写入的时_str_，不全显示则会写入省略号
    np.set_printoptions(threshold=sys.maxsize)

    # 按照文件名合并，得到每个图象对应标签
    label_img_df = pd.merge(label_df, img_df, on='file_name')              # 图像，文件名，标签名  13120个带标签唯一样本
    label_img_df.to_csv(r'after_process\label_character.csv')

    # 再从img_df中获取剩下的无标签样本
    name_list = label_img_df['file_name'].tolist()
    unlabel_img_df = img_df[~img_df['file_name'].isin(name_list)]           # 图像，文件名      310个无标签唯一样本
    unlabel_img_df.to_csv(r'after_process\unlabel_character.csv')

    return None


if __name__ == '__main__':
    get_csv()
