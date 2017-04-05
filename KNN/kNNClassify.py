# -*-coding:utf-8 -*-
from numpy import *
import operator
from KNN import *

class kNNClassify():

    def createDateSet(self):
        group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
        labels = ['A', 'A', 'B', 'B']
        return group, labels

    def classify0(self, inX, dataSet, labels, k):
        # 获取数组第一维的大小
        dataSetSize = dataSet.shape[0]

        # 获取距离
        diffMat = tile(inX, (dataSetSize, 1)) - dataSet
        sqDiffMat = diffMat ** 2
        # 没有axis参数表示全部相加，axis＝0表示按列相加，axis＝1表示按照行的方向相加
        sqDistances = sqDiffMat.sum(axis=1)
        distances = sqDistances ** 0.5
        sortedDistIndicies = distances.argsort()
        classCount = {}
        '''
            选择距离最小的k个点
            并根据最小距离进行排序
        '''
        for i in range(k):
            voteILabel = labels[sortedDistIndicies[i]]
            classCount[voteILabel] = classCount.get(voteILabel, 0) + 1
        sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
        return sortedClassCount[0][0]

    def main(self):
        group, labels = self.createDateSet()
        result = self.classify0([0, 0], group, labels, 3)
        print result

if __name__ == '__main__':
    knn = kNNClassify()
    knn.main()