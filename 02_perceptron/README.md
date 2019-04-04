# 一. 感知机-模型实现
本节插入图片为手写，见谅见谅.....<br>
## 1. 原理部分
原理部分按照李航老师的《统计学习方法》来讲解，一个统计学习方法由3部分组成：__`模型+策略+算法`__<br><br>
首先假设数据集T是`线性可分的`:
<img src='https://github.com/ChanLiang/ML/blob/master/02_perceptron/image/%E7%BA%BF%E6%80%A7%E5%8F%AF%E5%88%86.png'><br>
### (1)模型
*   模型的假设空间：定义在特征空间中的所有线性分类模型
*   几何解释：寻找一个能够正确分割所有正负样本点的分割超平面<br>
<img src='https://github.com/ChanLiang/ML/blob/master/02_perceptron/image/%E6%A8%A1%E5%9E%8B.png'>

### (2)策略
模型的假设空间里有无数个线性分类模型，我们如何才能在里边找出一个最好的呢？首先需要有一个最好的标准：能够最小化经验风险函数的模型就是假设空间里最好的模型。
那么问题又来了，如何定义经验风险函数呢？
*   __`0-1损失`__：很直观的想法，记录被分错样本点的个数，个数越少，模型就越好。简答直接，但是问题就在于这个函数对参数w,b是不连续的，那么也一定不可导，这样给我们后续的优化带来问题，NP-hard
*   __`所有误分类点到超平面的距离之和`__：关于参数w,b连续可导，方便后续优化
<img src='https://github.com/ChanLiang/ML/blob/master/02_perceptron/image/%E7%AD%96%E7%95%A5.png'>

### (3)算法
统计学习的问题最终还是归结为风险函数的最优化问题，那么这里的算法就是求解最优化问题的算法，这里采用的的是`随机梯度下降`。<br>
*   算法具体步骤如下：(截图自《统计学习方法》)
<img src="https://github.com/ChanLiang/ML/blob/master/02_perceptron/image/%E7%AE%97%E6%B3%95%E6%AD%A5%E9%AA%A4.png"><br><br>

## 2. 代码实现部分
代码so easy!注意一个问题就行：<br>
*   维度问题：w和x是同纬度的，在numpy中初始化w时，不要用rank=1的array（shape=(1.)），最好用列向量（shape=(n,1)），仅用一个reshape操作（O(1)）就可以完成，这样可以避免很多不必要的问题。<br>
