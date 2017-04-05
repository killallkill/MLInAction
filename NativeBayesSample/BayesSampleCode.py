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
        #1�������������֣�0��������
        classVec = [0, 1, 0, 1, 0, 1]
        return postingList, classVec


    def createVacobTable(self,dataSet):
        #�����ʻ��
        vacobTable = set([])
        for doc in dataSet:
            vacobTable = vacobTable | set(doc)

        return list(vacobTable)

    def setOfWords2Vec(self,vocabList,inputSet):
        #����һ��ȫΪ0������
        returnVec = [0]*len(vocabList)
        for word in inputSet:
            if word in vocabList:
                returnVec[vocabList.index(word)]=1
            else : print "the word %s is not in the vocabulary." % word

        return returnVec

