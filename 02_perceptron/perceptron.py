# -*- coding: utf-8 -*-
# @Time    : 2019/4/4 9:38
# @Author  : ChanLiang
# @FileName: perceptron.py
# @Software: PyCharm
# @Github  ：https://github.com/ChanLiang
import numpy as np

# 说明：后续将在应用环节使用该代码
class Perceptron(object):
    def __init__(self, X, Y):
        # 初始化参数w,b
        self.w = np.zeros((len(X[0]), 1))  # w和样本x维度相同,注意要用列向量，不要用rank=1的array
        self.b = 0
        self.X, self.Y = X, Y

    def train(self):
        loop = True
        step = 0.01
        while loop:  # 不断重复，直到训练集中没有误分类的点为止
            loop = False
            for i in range(len(self.X)):
                x, y = self.X[i].reshape((len(self.X[i]), 1)), self.Y[i]
                # 每次选取一个误分点x，梯度下降更新参数w,b，直到模型能将x分对为止
                while y * (np.dot(self.w.T, x) + self.b) <= 0:
                    self.w += step * y * x
                    self.b += step * y
                    loop = True

    def predict(self, x):
        if np.dot(self.w, x) + self.b > 0:
            return 1
        else:
            return -1
