import numpy as np
import operator
from os import listdir
import xlsxwriter as xs

def createDataSet():
    group = np.array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


def img2vector(filename):
    Vect = np.zeros((1, 1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            Vect[0, 32 * i + j] = int(lineStr[j])

    return Vect


def classify0(inX, dataSet, labels, k):
    datasize = dataSet.shape[0]
    diffMat = np.tile(inX, (datasize, 1)) - dataSet
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


def handwritingClassTest(k):
    hwLabels = []
    trainingFileList = listdir('digits/trainingDigits')
    m = len(trainingFileList)
    trainingMat = np.zeros((m, 1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        trainingMat[i, :] = img2vector('digits/trainingDigits/%s' % fileNameStr)
    testFileList = listdir('digits/testDigits')
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])

        vectorUnderTest = img2vector('digits/testDigits/%s' % fileNameStr)
        classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, k)
        print("the classifier came back with: %d, the real answer is: %d" % (classifierResult, classNumStr))
        if (classifierResult != classNumStr): errorCount += 1.0
    print("\n the total number of errors is: %d" % errorCount)
    print("\n the total error rate is: %f" % (errorCount / float(mTest) * 100))
    f = open("2.txt", "w")
    res = {'NumofTrain': 0, 'NumofTest': 0, 'Numofk': 0, 'T': 0, 'F': 0, 'Acc': 0}
    res['NumofTrain'] = m
    res['NumofTest'] = mTest
    res['Numofk'] = k
    res['T'] = mTest - errorCount
    res['F'] = errorCount
    res['Acc'] = 100 * (mTest - errorCount) / mTest
    f.write('Num of Train:%d,Num of Test:%d,Num of k:%d,T:%d,F:%d,Acc:%d'
            % (m, mTest, k, res['T'], res['F'], res['Acc']))
    f.close()
    temp = xs.Workbook('1.xlsx')
    temp2 = temp.add_worksheet('s001')
    temp2.write(0,0,'Num of Train')
    temp2.write(0,1,'Num of Test')
    temp2.write(0,2,'Num of k')
    temp2.write(0,3,'T')
    temp2.write(0,4,'F')
    temp2.write(0,5,'Acc')
    temp2.write(1,0,m)
    temp2.write(1,1,mTest)
    temp2.write(1,2,k)
    temp2.write(1,3,res['T'])
    temp2.write(1,4,res['F'])
    temp2.write(1,5,res['Acc'])


if __name__ == "__main__":
    k = input("Please set the number of k:")
    kn = int(k)
    handwritingClassTest(kn)
