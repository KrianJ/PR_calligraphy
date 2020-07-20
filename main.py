# -*- coding:utf-8 -*-
__Author__ = "KrianJ wj_19"
__Time__ = "2020/6/11 23:11"
__doc__ = """ 数据集划分：
带标签(label_character.csv)作为交叉验证集(5折)-训练集和验证集；
无标签(unlabel_character.csv)作为测试集"""

import pandas as pd
import numpy as np
import scipy.io as io
import matplotlib.pyplot as plt
# from sklearn.decomposition import KernelPCA
# from sklearn.manifold import LocallyLinearEmbedding
from similarity import sim_measure, str_2_arr, same_label_sim


def get_gpu():
    import os
    os.environ['CUDA_DEVICES_ORDER'] = "PCI_BUS_ID"
    os.environ['"CUDA_VISIBLE_DEVICES"'] = '0'


def read_data():
    """读取数据"""
    cv_data = pd.read_csv(r'after_process/label_character.csv', index_col=0)
    cv_set = cv_data.values[:, -1]  # 数据集
    cv_label = cv_data.values[:, -3]  # 标签集

    test_data = pd.read_csv(r'after_process/unlabel_character.csv', index_col=0)
    test_data = test_data.values[:, -1]  # 测试集

    # 数据集样本原始特征矩阵
    cv_mtx = np.array([str_2_arr(str(img)) for img in cv_set])          # shape: (13120, 7056)
    test_data = np.array([str_2_arr(str(img)) for img in test_data])
    return cv_mtx, cv_label, test_data


if __name__ == '__main__':
    from time import time
    start_time = time()

    # # 调用GPU运算
    # get_gpu()

    # cv_mtx, cv_label, test_data = read_data()
    # io.savemat('after_process/cv_mtx.mat', {'data': cv_mtx})
    # io.savemat('after_process/cv_label.mat', {'data': cv_label})
    # io.savemat('after_process/test_data.mat', {'data': test_data})
    # exit()
    cv_mtx = io.loadmat('after_process/cv_mtx.mat')['data']
    cv_label = io.loadmat('after_process/cv_label.mat')['data'].reshape((13120,))
    test_data = io.loadmat('after_process/test_data.mat')['data']
    # 检查数据集和标签集对应
    # for i in range(len(cv_mtx)):
    #     img = cv_mtx[i].reshape((84, 84))
    #     print(cv_label[i])
    #     plt.imshow(img)
    #     plt.show()

    cv_mtx = cv_mtx[:2000]
    cv_label = cv_label[:2000]

    # LLE对训练集特征矩阵降维
    # lle = LocallyLinearEmbedding(n_components=1024, n_neighbors=20,
    #                              eigen_solver='arpack', method='standard', neighbors_algorithm='kd_tree')
    # cv_reduced = lle.fit_transform(cv_mtx)

    """交叉验证"""
    num_folds = 5
    X_folds = np.array_split(cv_mtx, num_folds)
    y_folds = np.array_split(cv_label, num_folds)

    for fold in range(num_folds):
        # 划分训练集和验证集（4：1）
        print("”**************************第%d次交叉验证*******************************" % fold)
        validate_data = X_folds[fold]
        validate_label = y_folds[fold]
        train_data = np.concatenate(X_folds[:fold] + X_folds[fold+1:])
        train_label = np.concatenate(y_folds[:fold] + y_folds[fold+1:])

        # 验证结果
        v_word = []
        v_res = []
        v_sim =[]

        # 利用验证集验证相似性度量准确性
        for i in range(validate_data.shape[0]):
            word = validate_data[i, :]
            word_label = validate_label[i]

            # 验证集中第i个样本和训练集中所有样本的相似度
            i_dist = [sim_measure(word, train_data[i, :]) for i in range(train_data.shape[0])]
            # 计算自身相似度
            self_sim = same_label_sim(word_label, cv_mtx, cv_label)
            # 相减取绝对值得到偏差-相似性度量
            bia_sim = [abs(i - self_sim) for i in i_dist]
            # 距离最小的前20个样本的索引和值
            value_dist = []
            index_dist = []
            Inf = 9999
            for i in range(20):
                a = min(bia_sim)
                value_dist.append(a)
                index_dist.append(bia_sim.index(min(bia_sim)))
                bia_sim[bia_sim.index(min(bia_sim))] = Inf
            value_dist = np.round(np.array(value_dist), 4)

            # 输出对应标签
            match_labels = train_label[index_dist]
            print('和"%s"最相似的前20个汉字是' % word_label, match_labels)
            print("相似度为", value_dist)
            # 记录单字匹配结果
            v_word.append(word_label)
            v_res.append(match_labels)
            v_sim.append(value_dist)

            # 输出匹配的图像
            # for idx in index_dist:
            #     match_word = train_data[idx, :].reshape((84, 84))
            #     plt.imshow(match_word)
            #     plt.show()

            # 将相似性匹配结果写入csv
            # dict = {'target_word': word_label, '匹配结果': str(match_labels), '相似性度量': str(value_dist)}

        i_df = pd.DataFrame({
            '目标字': v_word, '匹配字': v_res, '相似度': v_sim
        })
        i_df.to_csv('shape_match_result/%d_th_validation.csv')

    run_time = time() - start_time
    print("运行时间：", run_time)


