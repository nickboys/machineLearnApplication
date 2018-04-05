#! /usr/bin/env python
# -*- coding: gbk -*-

from numpy import *
import operator
import matplotlib
import matplotlib.pyplot as plt

def createDataSet():
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances ** 0.5
    sortedDistIndicies = distances.argsort()
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1

    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

def filematrix(filename):
    fr=open(filename)
    arrayOLines=fr.readlines()
    numberOflines=len(arrayOLines)
    returnMat=zeros((numberOflines,3))
    classLabelVector=[]
    index=0
    for line in arrayOLines:
        line=line.strip()
        listFromLine=line.split("\t")
        returnMat[index,:]=listFromLine[0:3]
        classLabelVector.append((listFromLine[-1]))
        index+=1
    return returnMat,classLabelVector

def autonorm(dataSet):
    minVals=dataSet.min(0)
    maxVals=dataSet.max(0)
    ranges=maxVals-minVals
    normDataSet=zeros(shape(dataSet))
    m=dataSet.shape[0]
    normDataSet=dataSet-tile(minVals,(m,1))
    normDataSet=normDataSet/tile(ranges,(m,1))
    return normDataSet,ranges,minVals

def datingClassTest():
    hoRatio=0.10
    datingDataMat,datingLabels=filematrix('datingTestSet2.txt')
    normMat,ranges,minVals=autonorm(datingDataMat)
    m=normMat.shape[0]
    numTestVecs=int(m*hoRatio)
    errorCount=0.0
    for i in range(numTestVecs):
        classifierResult=classify0(normMat[i,:],normMat[numTestVecs:m,:],datingLabels[numTestVecs:m],3)
        print("the classifier came back with :"+classifierResult+",the real answer is :"+datingLabels[i])
    if (classifierResult != datingLabels[i]): errorCount += 1.0
    print('the total error rate is :%f' % int(errorCount / numTestVecs))


if __name__ == '__main__':
    group, labels = createDataSet()
    classify0([0, 0], group, labels, 3)
    print(classify0([0, 1.1], group, labels, 3))
    datingDataMat, datingLabels =filematrix('datingTestSet2.txt')
    # print(datingDataMat,datingLabels)
    # fig = plt.figure();
    # ax = fig.add_subplot(111)
    # ax.scatter(datingDataMat[:, 1], datingDataMat[:, 2])
    # plt.show()
    datingClassTest()