# -*- coding:utf-8 -*-
__Author__ = "KrianJ wj_19"
__Time__ = "2020/6/13 15:07"
__doc__ = """ """

text = """
1. 现在已经做到结合jpg文件和mdb数据统计出有标签(13120)和无标签(310)的数据
有标签：做交叉验证集；无标签: 测试集
2. 所有图像均已做二值化和canny最优边缘检测，只剩下轮廓点(255)和背景(0)
3. 根据存入的二值化图像，做形状匹配
    1). 以图像中每个轮廓点为中心，划分8个象限，统计每个象限中的轮廓点数，构造该图像的特征矩阵
    2). 用测试集中的图像轮廓特征矩阵与交叉验证集中特征矩阵做比对，得出结论
"""

import pandas as pd
import numpy as np


def shape_describe(m):
    """讲矩阵m按照方向切割成8个：上下左右+四个斜方向
    统计每个方向非零像素点的个数
    :return: DataFrame -> 轮廓点数 * 8
            index:轮廓点, columns:特征"""
    m[m > 1] = 1
    row, col = m.shape[0], m.shape[1]

    # 按行遍历轮廓点
    direction_record = {}
    for c in range(col):
        for r in range(row):
            # 获取轮廓点m[r, c]的方向信息
            if m[r, c] == 1:
                west_m = m[r, :c]         # 左
                east_m = m[r, c+1:]       # 右
                north_m = m[:r, c]        # 上
                south_m = m[r+1:, c]      # 下
                nw_m = m[:r, :c]          # 左上
                ne_m = m[:r, c+1:]        # 右上
                ws_m = m[r+1:, :c]        # 左下
                es_m = m[r+1:, c+1:]      # 右下
                direction = [np.count_nonzero(nw_m), np.count_nonzero(north_m),
                             np.count_nonzero(ne_m), np.count_nonzero(west_m),
                             np.count_nonzero(east_m), np.count_nonzero(ws_m),
                             np.count_nonzero(south_m), np.count_nonzero(es_m)]     # 方向信息：从左至右从上到下记录
                coordinate = '(%d,%d)' % (r, c)
                direction_record[coordinate] = direction
    record_df = pd.DataFrame(direction_record,
                             index=['north-west', 'north', 'north_east', 'west', 'east', 'west-south', 'south', 'east-south'])
    # 转置一下
    record_df = pd.DataFrame(record_df.values.T, index=record_df.columns, columns=record_df.index)
    return record_df


def normlize_shape(img_array):
    """归一化轮廓特征矩阵
    :param: ndarray
    :return: ndarray"""
    row, col = img_array.shape[0], img_array.shape[1]
    norm_m = np.zeros((row, col))
    for j in range(col):
        for i in range(row):
            i_length = np.sum(img_array[i:])       # 轮廓点i的特征长度
            norm_m[i, j] = img_array[i, j] / i_length
    return np.round(norm_m, 4)


def shape_similarity(img1, img2):
    """比较两张图像的轮廓特征矩阵相似度
    :param img1. img2: df  二值化灰度图
    :return: ndarray            img1和img2的相似性度量矩阵 """
    img1 = shape_describe(img1).values
    img2 = shape_describe(img2).values

    norm_img2 = normlize_shape(img2)
    similarity = np.dot(img1, norm_img2.T)
    return similarity


def same_label_sim(label, cv_mtx, cv_label):
    """计算给定数据集中相同标签样本的相似性度量值均值"""
    # 所有具有相同标签的索引值
    cv_label = list(cv_label)
    label_index = []
    for i in range(len(cv_label)):
        if label in cv_label[i:]:
            label_index.append(cv_label.index(label, i))
    label_index = list(set(label_index))
    # 读取相同标签的data
    label_data = cv_mtx[label_index]
    # 计算平均相似度
    length = label_data.shape[0]
    same_sim = [sim_measure(label_data[m1, :], label_data[m2, :]) for m1 in range(length) for m2 in range(length)]
    return np.mean(same_sim)


def str_2_arr(str):
    import re
    """将一个字符串类型的矩阵转换成ndarray: uint8版本
    “[[0,0,0,0],[1,0,0,1],[1,1,1,0]]” ---> [0,0,0,0,1,0,0,1,1,1,1,0]"""
    number = re.findall(r'\d+', str)
    # d = int(sqrt(len(number)))              # 由于图像已经处理成方针，可以直接开根
    # arr = np.array(number).reshape((d, d))
    arr = np.array(number).astype('uint8')
    return arr


def str_to_arr(str):
    """将坐标字符串转换成ndarray: int版本"""
    import re
    number = re.findall(r'\d+', str)
    arr = np.array(number).astype('int')
    return arr


def sim_measure(m1, m2):
    """
    :param m1, m2: (un)labeled_character.csv中的img(行向量), 即二值化的灰度图
    :return: m1和m2的相似性度量
    """
    m1 = m1.reshape((84, 84))
    m2 = m2.reshape((84, 84))
    m1_index = shape_describe(m1).index
    m2_index = shape_describe(m2).index

    similarity = shape_similarity(m1, m2)

    # 找出m1与m2轮廓点之间最相似的一对(行最大)，并求其之间距离
    max_sim = np.max(similarity, axis=1)
    dists = []
    # 遍历相似度矩阵的每行
    for i in range(len(max_sim)):
        row = similarity[i, :]
        sim = max_sim[i]
        j = list(row).index(sim)

        # 确定该行最大值对应的两个坐标
        x = str_to_arr(str(m1_index[i]))
        y = str_to_arr(str(m2_index[j]))

        # 计算两个坐标之间的距离
        dist = np.sqrt(np.sum((x-y)**2))
        dists.append(dist)
    dists = np.array(dists)

    # 最终通过max_sim和dists的均值衡量两张图像的相似度
    avg_sim = max_sim.mean()
    avg_dist = dists.mean()

    # return avg_dist
    return np.round(100 * (avg_dist / avg_sim), 4)


if __name__ == '__main__':
    """Function Test demo"""
    x = np.array([[1,1,0,0,0,0],
                  [0,0,1,0,0,0],
                  [0,0,0,1,0,0],
                  [0,0,0,0,1,1],
                  [0,0,0,0,0,1],
                  [0,0,0,0,0,1]])
    print(shape_describe(x))

