# -*- coding=utf-8 -*-
import codecs
from math import exp
from numpy import mat, shape, ones


class LogisticRegression:

    def loadDataSet(self):
        dataMat = []
        labelMat = []
        '''
        从文件中装载数据集
        测试数据集格式：
        “-0.017612	14.053064	0”
        '''
        fr = codecs.open('data\\LRTestData\\testSet.txt', 'r', 'utf-8')
        for line in fr.readline():
            lineArray = line.strip().split()
            '''
            构建特征数组，该数组是二维数组
            每列代表数组的不同特征，每行则代表一个训练样本
            '''
            dataMat.append([1, float(lineArray[0]), float(lineArray[1])])
            labelMat.append(int(lineArray[2]))
        return dataMat,labelMat

    def sigmoid(self,inX):
        return 1.0 / (1 + exp(-inX))

    def gradAscent(self,dataMatIn, classLabel):
        dataMatrix = mat(dataMatIn)
        labelMatrix = mat(classLabel).transpose()
        m,n = shape(dataMatrix)
        '''
        alpha代表每次移动的步长，梯度代表的是每次移动的方向
        '''
        alpha = 0.001
        maxCycle = 500
        weight = ones((n,1))
        for k in range(maxCycle):
            h = self.sigmoid(dataMatrix * weight)
            '''
            error相当于是y的变化量
            '''
            error = (labelMatrix - h)
            weight = weight + alpha * dataMatrix.transpose() * error
        return weight

