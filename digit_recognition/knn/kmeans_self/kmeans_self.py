#!/usr/bin/env python
# encoding: utf-8

import numpy as np

class KMeans(object):
    """
    n_clusters: 聚类个数，即k
    initCent: 质心初始化方式，可选"random"或指定一个具体的array，默认random,即随机初始化
    max_iter: 最大迭代次数
    """
    def __init__(self, n_clusters = 5, initCent = 'random', max_iter=300):
        if hasattr(initCent, '__array__'):
            n_clusters = initCent.shape[0]
            self.centroids = np.asarray(initCent, dtype=np.float)
        else:
            self.centroids = None

        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.initCent = initCent
        self.clusterAssment = None
        self.labels = None
        self.sse = None
# 计算两点的欧氏距离
    def _distEclud(self, vecA, vecB):
        return np.linalg.norm(vecA - vecB)

# 随机选取k个质心，必须在数据集的边界内
    def _randCent(self,X,k):
        n = X.shape[1]        # 特征维数
        centroids = np.empty((k,n))  # k*n的矩阵，用于存储质心
        for j in range(n):
            minj = min(X[:,j])
            rangej = float(max(X[:,j]) - minj)
            centroids[:,j] = (minj+rangej*np.random.rand(k,1)).flatten()
        return centroids

    def fit(self,X):
        # 类型检查
        if not isinstance(X, np.ndarray):
            try:
                X = np.asarray(X)
            except:
                raise TypeError("numpy.ndarray required for X")

        m = X.shape[0]  # 代表样本数量
        self.clusterAssment = np.empty((m,2))  # m*2的数组，第一列存储样本点所属族的索引值
                                               # 第二列存储该点与所属族的质心的平方误差
        if self.initCent == 'random':
            self.centroids = self._randCent(X, self.n_clusters)

        clusterChanged = True
        for r in range(self.max_iter):
            clusterChanged = False
            for i in range(m):   # 将每个样本点分配到离它最近的质心所属的族
                minDist = np.inf; minIndex = -1
                for j in range(self.n_clusters):
                    disji = self._distEclud(self.centroids[j,:], X[i,:])
                    if distji < minDist:
                        minDist = distji; minIndex = j
                if self.clusterAssment[i,0] != minIndex:
                    clusterChanged = True
                    self.clusterAssment[i,:] = minIndex, minDist**2














































