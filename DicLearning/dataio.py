import numpy as np

def loadData(fileName, arrSize=3, spliter=' '):
    fin = open(fileName, "r")
    lines = fin.readlines()
    cnt = len(lines)
    data = np.zeros([cnt,arrSize])
    for i in range(cnt):
        data[i] = lines[i].split(spliter)[-3:]
        # print(data[i])
    fin.close()
    return data

def outputData(fileName, data):
    fout = open(fileName, "w")
    for i in range(data.shape[0]):
        fout.write(str(data[i][0]) + ' ' + str(data[i][1]) + ' ' + str(data[i][2]) + '\n')
    fout.close()
    return

def outPutFace(fileName, data) :
    fout = open(fileName, "w+")
    data = data.reshape(-1,3)
    for i in range(data.shape[0]):
        fout.write(str(data[i][0]) + ' ' + str(data[i][1]) + ' ' + str(data[i][2]) + '\n')
    fout.close()
    return