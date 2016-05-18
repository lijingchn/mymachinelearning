#!/usr/bin/env python
# encoding: utf-8

from time import time
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D
from sklearn import manifold, datasets, decomposition, ensemble, lda, random_projection

# 加载数据
digits = datasets.load_digits(n_class=5)
X = digits.data
y = digits.target

n_img_per_row = 20
img = np.zeros((10 * n_img_per_row, 10 * n_img_per_row))
ylabel = []
for i in xrange(n_img_per_row):
    ix = 10 * i + 1
    for j in range(n_img_per_row):
        iy = 10 * j + 1
        img[ix:ix + 8, iy:iy + 8] = X[i * n_img_per_row + j].reshape((8,8))
        ylabel.append(y[i * n_img_per_row + j])
#plt.imshow(img, cmap='gray')
plt.imshow(img, cmap=plt.cm.binary)
plt.title("A selection from the 64-dimensional digits dataset")
print ylabel

#plt.show()

# 将降维后的数据可视化，2维
def plot_embedding_2d(X, title=None):
    # 坐标缩放到[0,1]区间
    x_min, x_max = np.min(X,axis=0), np.max(X,axis=0)
    X = (X - x_min) / (x_max - x_min)

    # 降维后的坐标为（X[i,0], X[i,1]）,在该位置画出对应的digits
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    for i in xrange(X.shape[0]):
#        ax.text(X[i,0],X[i,1],str(digits.target[i]),
        ax.text(X[i,0],X[i,1],str(ylabel[i]),
                color=plt.cm.Set1(y[i]/10.),
                fontdict={'weight':'bold', 'size':9})

        if title is not None:
            plt.title(title)


# 将降维后的数据可视化，3维
def plot_embedding_3d(X, title=None):
    #坐标缩放到[0,1]区间
    x_min,x_max = np.min(X,axis=0),np.max(X,axis=0)
    X = (X - x_min)/(x_max - x_min)

    #降维后的坐标为(X[i,0],X[i,1],X[i,2]),在该位置画出对应的digits
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1,projection='3d')
    for i in xrange(X.shape[0]):
#        ax.text(X[i,0],X[i,1],X[i,2],str(digits.target[i]),
        ax.text(X[i,0],X[i,1],X[i,2],str(ylabel[i]),
                color=plt.cm.Set1(y[i]/10.),
                fontdict={'weight':'bold', 'size':9})
        if title is not None:
            plt.title(title)


# t-SNE
# ####################################3
print 'Computing t-SNE embedding'
tsne = manifold.TSNE(n_components=3, init='pca', random_state=0)
t0 = time()
#X_tsne = tsne.fit_transform(X)
X_tsne = tsne.fit_transform(img)
print X_tsne.shape
plot_embedding_2d(X_tsne[:,0:2], "t-SNE 2D")
plot_embedding_3d(X_tsne, "t-SNE 3D (time %.2fs)" % (time()-t0))

plt.show()












































