# -*- coding=utf-8 -*-
import codecs
import random

from numpy import mat, shape, ones, exp, array, arange
import matplotlib.pyplot as plt


class LogisticRegression():


    def loadDataSet(self):
        dataMat = []
        labelMat = []
        '''
        从文件中装载数据集
        测试数据集格式：
        “-0.017612	14.053064	0”
        '''
        fr = codecs.open('../data/LRTestData/testSet.txt', 'r', 'utf-8')
        fileLine = fr.readlines()
        for line in fileLine:
            lineArray = line.strip().split('\t')

            '''
            构建特征数组，该数组是二维数组
            每列代表数组的不同特征，每行则代表一个训练样本

            if lineArray[0].startswith('-') and lineArray[1].startswith('-'):
                dataMat.append([1, -float(lineArray[0][1:]),-float[lineArray[1][1:]]])
            elif lineArray[0].startswith('-'):
                dataMat.append([1, -float(lineArray[0][1:]), float(lineArray[1])])
            elif lineArray[1].startswith('-'):
                dataMat.append([1, float(dataArr[0], -float(dataArr[1][1:]))])
            else:
                dataMat.append([1, float(lineArray[0]), float(lineArray[1])])
            '''
            dataMat.append([1, float(lineArray[0]), float(lineArray[1])])
            labelMat.append(int(lineArray[2]))
        return dataMat,labelMat

    def sigmoid(self,inX ):
        return 1.0 / (1 + exp(-inX))

    '''
    通过梯度计算出最后的权重矩阵
    传入参数指定特征矩阵，标签矩阵和进行迭代的次数
    该算法的特征是没一次权重更新都将遍历整个矩阵的特征值项，这对于特征较多的情况时，该方法计算复杂度过高
    '''
    def gradAscent(self,dataMatIn, classLabel, alphaDis = 0.001,itrationTimes=500):
        dataMatrix = mat(dataMatIn)
        labelMatrix = mat(classLabel).transpose()
        m,n = shape(dataMatrix)
        '''
        alpha代表每次移动的步长，梯度代表的是每次移动的方向
        '''
        alpha = alphaDis
        #设置迭代次数
        maxCycle = itrationTimes
        weight = ones((n,1))
        #每次进行迭代的时候，都将遍历整个特征矩阵，对于样本较多的数据，这个遍历的代价是相当高的
        #梯度回归系数迭代公式：梯度系数 = 梯度系数（原）+ 步长 * 梯度
        for k in range(maxCycle):
            h = self.sigmoid(dataMatrix * weight)
            '''
            error相当于是y的变化量
            '''
            error = (labelMatrix - h)
            weight = weight + alpha * dataMatrix.transpose() * error
        #返回权重矩阵
        return weight

    '''
    随机梯度算法，该算法每次在更新权重矩阵的值时，一次仅使用一个样本来进行权重矩阵的更新，这样计算复杂度相对降低了很多。
    在其中，添加了alpha每次迭代的更改，每进行一次迭代alpha就减小，但是总是不会小于0，因此也使得新特征数据总是会有影响产生。
    在每次随机样本选择、进行了权重更新后，就将该随机选择的特征重数据中删除，也使得计算度降低。
    '''
    def randomGradAscent(self, dataMat, classlabel, iterTime = 200):
        m, n = shape(dataMat)
        weights = ones(n)
        for j in range(iterTime):
            dataIndex = range(m)
            for i in range(m):
                alpha = 4 / (1.0 + i + j) + 0.01
                randIndex = int(random.uniform(0, len(dataIndex)))
                h = self.sigmoid(sum(dataMat[randIndex] * weights) )
                bis = classlabel[randIndex] - h
                weights = weights + alpha * bis * dataMat[randIndex]
                del(dataIndex[randIndex])
        return weights



    def plotBestFit(self,weights1, weight2):
        dataMat ,labelMat = self.loadDataSet()
        dataArr = array(dataMat)
        n = shape(dataArr)[0]
        xcord1 = [] ; ycord1 = []
        xcord2 = [] ; ycord2 = []
        for i in range(n):
            if int(labelMat[i]) == 1:
                xcord1.append(dataArr[i, 1])
                ycord1.append(dataArr[i, 2])
            else:
                xcord2.append(dataArr[i, 1])
                ycord2.append(dataArr[i, 2])
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
        ax.scatter(xcord2, ycord2, s=30, c='green')
        x = arange(-3.0, 3.0, 1.0)
        y1 = (-weights1[0]-weights1[1]*x)/weights1[2]
        y2 = (-weight2[0]-weight2[1]*x)/weight2[2]
        ax.plot(x, y1, linestyle="dashed", color="blue", linewidth=3)
        ax.plot(x, y2, color="red")
        plt.xlabel("X1"); plt.ylabel("Y1")
        plt.show()

if __name__ == "__main__":
    lr = LogisticRegression()
    dataArr ,classlabel = lr.loadDataSet()
    weights1 = lr.randomGradAscent(array(dataArr), classlabel,150)
    weights2 = lr.randomGradAscent(array(dataArr), classlabel)
    lr.plotBestFit(weights1, weights2)