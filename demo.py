# -*- coding:utf-8 -*-
__Author__ = "KrianJ wj_19"
__Time__ = "2020/6/14 11:20"
__doc__ = """ """

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from similarity import str_2_arr, sim_measure, same_label_sim
import random


df = pd.read_csv(r'D:\Pyproject\PR_calligraphy\after_process\label_character.csv')
labels = df.values[:, -3]
imgs = df.values[:, -1]


def get_word_img(str, imgs):
    """给定汉字，获得标签集中所有符合该标签的图像列表"""
    word_index = []
    for k, v in enumerate(labels):
        if v == str:
            word_index.append(k)
    yue_img = [str_2_arr(img) for img in imgs[word_index]]
    return yue_img


def get_index(str, arr):
    """取arr中所有str的索引"""
    from collections import defaultdict
    dd = defaultdict(list)
    for k, va in [(v, i) for i, v in enumerate(arr)]:
        dd[k].append(va)
    return dd[str]


if __name__ == '__main__':
    """由于字符集太多，运行所有样本这台破笔记本太慢了，故采只抽取20个字符作为数据集"""

    """找出所有字符在数据集中的样本个数"""
    # num = {}
    # labels = list(labels)
    # for val in list(set(labels)):
    #     num[val] = labels.count(val)
    # df = pd.DataFrame(num, index=['count']).T
    # df.to_csv('after_process/label_count.csv')
    # exit()

    """随机抽取字频大于10的20个字符组成数据集"""
    df = pd.read_csv('after_process\label_count.csv').values
    characters = [df[i][0] for i in range(df.shape[0]) if df[i][1] >= 10]
    characters = random.sample(characters, 20)
    print(characters)

    """获取包含指定字符的所有数据集"""
    img_data = np.zeros((7056, ))
    img_label = []
    for str in characters:
        # 获取指定字符str的所有样本
        word_imgs = np.array(get_word_img(str, imgs))
        img_data = np.vstack((img_data, word_imgs))
        img_label += [str]*len(word_imgs)
    img_data = np.delete(img_data, 0, 0)
    img_label = np.array(img_label)

    # 从数据集中每个字符各抽取2个作为验证集
    validation = np.zeros((7056, ))
    validation_label = []
    for c in characters:
        c_index = get_index(c, img_label)[:2]
        c_data = img_data[c_index, :]
        validation = np.vstack((validation, c_data))
        if len(c_index) == 1:
            validation_label.append(c)
        else:
            validation_label += [c, c]
    validation = np.delete(validation, 0, 0)
    validation_label = np.array(validation_label)

    """获取validation与数据集中所有图像的相似性度量"""
    for i in range(validation.shape[0]):
        img = validation[i, :]
        label = validation_label[i]

        # 计算与训练集所有样本的相似性度量
        i_dist = [sim_measure(img, img_data[j, :]) for j in range(img_data.shape[0])]

        # 计算同标签的的相似性度量
        self_sim = same_label_sim(label=label, cv_mtx=img_data, cv_label=img_label)

        # 两者相减，得到偏差-相似性度量
        bia_dist = [abs(i - self_sim) for i in i_dist]

        # bia_dist的最小的前9个样本的索引和值
        import copy
        temp = copy.deepcopy(bia_dist)
        value_dist = []
        index_dist = []
        Inf = 9999
        for i in range(9):
            a = min(temp)
            value_dist.append(a)
            index_dist.append(temp.index(a))
            temp[temp.index(a)] = Inf

        # 输出匹配图像
        match_labels = img_label[index_dist]
        match_imgs = img_data[index_dist, :]
        print('一共有%d个%s' % (list(img_label).count(label), label))
        print('与%s最相似的前9个汉字是' % label, match_labels)
        print("相似度为", np.round(np.array(value_dist), 4))

        plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文标签
        plt.rcParams['axes.unicode_minus'] = False

        for i in range(match_imgs.shape[0]):
            match_word = match_imgs[i, :]
            plt.title('与"%s"最相似的汉字' % label)
            plt.subplot(3, 3, i+1)
            plt.imshow(match_word.reshape((84, 84)))
        plt.show()
        # break


