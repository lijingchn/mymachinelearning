#!/usr/bin/env python
# encoding: utf-8

import cPickle as pickle
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans

if __name__ == "__main__":
    # 加载数据
    X, y = pickle.load(open('data.pkl', 'r'))
    print X

    n_clusters = 10
    k_means = KMeans(init='k-means++',n_clusters=n_clusters,n_init=20)
    k_means.fit(X)

    labels = k_means.labels_
    print '==============================='
    print labels
    cents = k_means.cluster_centers_
    print '==============================='
    print cents

    # 画出聚类结果，每一类用一种颜色表示
    colors = ['b','g','r','k','c','m','y','#e24fff','#524C90','#845868']
    for i in range(n_clusters):
        index = np.nonzero(labels == i)[0]
        x0 = X[index,0]
        x1 = X[index,1]
        y_i = y[index]
        for j in range(len(x0)):
            plt.text(x0[j],x1[j],str(int(y_i[j])), color=colors[i],fontdict={'weight':'bold', 'size':9})
        plt.scatter(cents[i,0],cents[i,1],marker='x',color=colors[i],linewidths=6)
    plt.axis([-30,30,-30,30])
    plt.show()












































