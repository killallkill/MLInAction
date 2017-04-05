# -*- coding=utf-8 -*-
import codecs

from numpy import array

from LR import LogisticRegression


class HorseColicTest():

    '''
    该函数用于测试回归后的回归参数效果
    将测试用例用回归参数进行回归，得到的结果与真实值进行比较
    '''
    def checkVector(self, inX ,weights):
        prob = LogisticRegression().sigmoid(sum(inX * weights))
        if prob > 0.5 :
            return 1.0
        else:
            return 0.0


    def cloicTest(self):
        trainDataFile = codecs.open("../data/LRTestData/horseColicTraining.txt", 'r', 'utf-8')
        trainDataset = []
        trainDataLabel = []
        '''
        加载训练数据到数据集，并训练数据，得出逻辑回归参数
        '''
        for line in trainDataFile.readlines():
            curLine = line.strip().split('\t')
            lineArr = []
            for i in range(21):
                lineArr.append(float(curLine[i]))
            trainDataset.append(lineArr)
            trainDataLabel.append(float(lineArr[-1]))
        trainWeights = LogisticRegression().randomGradAscent(array(trainDataset), trainDataLabel, 150)
        print trainWeights

        '''
        使用测试集来对回归模型进行测试，并计算该模型的错误率
        '''
        errorCount = 0.0
        numTestVec = 0.0
        frTest = codecs.open("../data/LRTestData/horseColicTest.txt", 'r', 'utf-8')
        for line in frTest.readlines():
            numTestVec += 1
            curLine = line.strip().split('\t')
            lineArr = []
            for i in range(21):
                lineArr.append(float(curLine[i]))
            if int(self.checkVector(array(lineArr),trainWeights)) != int(curLine[-1]):
                errorCount += 1

        errorRate = (float(errorCount/numTestVec))
        print "the error rate is %f" % errorRate
        return errorRate

    def multyTest(self):
        numTests = 10;errorSum = 0.0
        for k in range(numTests):
            errorSum += self.cloicTest()
        print "after %d iterations the average error rate is %f" % (numTests,(float(errorSum/numTests)))


if __name__ == "__main__":
    hoc = HorseColicTest()
    hoc.multyTest()
