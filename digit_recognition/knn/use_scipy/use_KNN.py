#!/usr/bin/env python
# encoding: utf-8

from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from numpy import *
import csv
import time

def toInt(array):
    array = mat(array)
    m,n = shape(array)
    newArray = zeros((m,n))
    for i in xrange(m):
        for j in xrange(n):
            newArray[i,j] = int(array[i,j])
    return newArray

def normalizing(array):
    m,n = shape(array)
    for i in xrange(m):
        for j in xrange(n):
            if array[i,j] != 0:
#                array[i,j] = 1
                array[i,j] = array[i,j]/255
    return array


def loadTrainData():
    l = []
    with open('train.csv','rb') as file:
        lines = csv.reader(file)
        for line in lines:
            l.append(line)
    l.remove(l[0])
    l = np.array(l)
    label = l[:,0]
    data = l[:,1:]
    return normalizing(toInt(data)),toInt(label)

def loadTestData():
    l = []
    with open('test.csv', 'rb') as file:
        lines = csv.reader(file)
        for line in lines:
            l.append(line)
        l.remove(l[0])
        data = np.array(l)
        return normalizing(toInt(data))

def saveResult(result,csvName):
    with open(csvName, 'wb') as file:
        mywriter = csv.writer(file)
        j = 0
        for i in result:
            j = j+1
            tmp = []
            tmp.append(j)
            tmp.append(int(i))
            mywriter.writerow(tmp)

def knnClassify(trainData,trainLabel,testData):
    knnClassifier = KNeighborsClassifier(algorithm = 'kd_tree')
    knnClassifier.fit(trainData,ravel(trainLabel))
    testLabel = knnClassifier.predict(testData)
    saveResult(testLabel,'sklearn_knn_result_02.csv')
    return testLabel

print time.asctime(time.localtime(time.time()))
trainData,trainLabel = loadTrainData()
testData = loadTestData()
print time.asctime(time.localtime(time.time()))
knnClassify(trainData,trainLabel,testData)
print time.asctime(time.localtime(time.time()))




















































