# -*- coding="utf-8" -*-

from numpy import *

class NativeBayes():

    def loadDataSet(self):
        postingList = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                       ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                       ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                       ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                       ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                       ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
        #1代表侮辱性文字，0代表正常
        classVec = [0, 1, 0, 1, 0, 1]
        return postingList, classVec


    def createVacobTable(self,dataSet):
        #创建词汇表
        vacobTable = set([])
        for doc in dataSet:
            vacobTable = vacobTable | set(doc)

        return list(vacobTable)

    def setOfWords2Vec(self,vocabList,inputSet):
        #创建一个全为0的向量
        returnVec = [0]*len(vocabList)
        for word in inputSet:
            if word in vocabList:
                returnVec[vocabList.index(word)]=1
            else : print "the word %s is not in the vocabulary." % word

        return returnVec

