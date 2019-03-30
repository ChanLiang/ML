# -*- coding: utf-8 -*-
# @Time    : 2019/3/29 9:35
# @Author  : ChanLiang
# @FileName: kNN.py
# @Software: PyCharm
# @Github  ：https://github.com/ChanLiang
import numpy as np

# 生成训练集
def createDataSet():
    features = np.array([[0, 0], [0, 0.1], [1, 1], [1, 1.1]])   # 两类：A，B
    labels = ['A', 'A', 'B', 'B']
    return features, labels

# kNN算法：根据最近的k个训练样本，来预测新样本的类别
def classify(new_example, trainingDatas, labels, k):
    # 重要trick: 将单个测试样本在行向量方向上重复了trainingDatas.shape[0]次，使之与traningDatas维度相同
    new_example = np.tile(new_example, (trainingDatas.shape[0], 1))
    # 这样就不用写一个显式的for循环，依次来计算单个样本和每一个测试样本间的距离了，用矩阵运算来代替for循环（依靠并行计算提速）
    distVector = (((new_example - trainingDatas) ** 2).sum(axis=1) ** 0.5)
    # 返回distVector按距离从小到大排序后的元素的索引值
    sortedDistIndex = distVector.argsort()
    classCount = {}
    for i in range(k):  # 找出最近的k个训练样本，用字典记录他们的类别
        labelsI = labels[sortedDistIndex[i]]
        classCount[labelsI] = classCount.get(labelsI, 0) + 1
    # 对字典逆序排序，使出现最多的类别处在最前面
    tupleList = sorted(classCount.items(), key = lambda e:e[1], reverse = True)
    return tupleList[0][0]

# 测试算法
def main():
    features, labels = createDataSet()
    new_example1 = [0.1, 0.3]
    new_example2 = [0.9, 1.3]
    classOfExample1 = classify(new_example1, features, labels, 2)
    classOfExample2 = classify(new_example2, features, labels, 2)
    print ('the class of the new example ' + str(new_example1) + ' is :' + classOfExample1)
    print ('the class of the new example ' + str(new_example2) + ' is :' + classOfExample2)

if __name__ == '__main__':
    main()






