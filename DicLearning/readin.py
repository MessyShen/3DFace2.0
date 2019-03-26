#coding=utf-8

import dataio
import numpy as np
import os
import time

def combineData(xyzData, bndData, normalize=False) :
    n_face, _ = xyzData.shape
    n_mark, _ = bndData.shape
    X = np.zeros((n_face + n_mark, 3))
    # print(xyzData)
    X[:n_face,:] = xyzData
    X[n_face:,:] = bndData
    X = X.reshape((-1,1))
    if normalize:
        X -= np.mean(X, axis=0)
        X /= np.std(X, axis=0)
    
    # print(X.shape)
    return X

def FetchAllData(TrainDataPath) :
    # ExportFace = "ExportBnd"
    curPath = os.getcwd()
    FileCnt = 1000
    X = np.zeros((FileCnt, 23349))
    list_dirs = os.walk(TrainDataPath)
    i = 0
    for root, _, files in list_dirs :
        for f in files:
            if f[-3:] == 'xyz' :
                if i >= FileCnt : break
                # print(f)
                bndFile = f[:-3] + 'bnd'
                xyzData = dataio.loadData(os.path.join(TrainDataPath,f))
                bndData = dataio.loadData(os.path.join(TrainDataPath,bndFile),
                                          spliter='\t\t')
                # print(bndData.shape)   
                F = combineData(xyzData, bndData)
                X[i, :] = F.ravel()
                i += 1
                
    # print(X)
    return X

def FetchBU3DData(DataPath, facePointMult3=21081, fileCnt=1000, printTime=False, valCnt=10,
                  trainDirName="train_Resampled", testDirName="val_Resampled", landmarkDirName="landmarks") :
    startTime = time.time()
    if printTime:
        print("Loading Data...")
    X = np.zeros((fileCnt, facePointMult3 + 83*3))
    Y = np.zeros((valCnt, facePointMult3 + 83*3))
    cnt = 0
    trainDataPath = os.path.join(DataPath, trainDirName)
    testDataPath  = os.path.join(DataPath, testDirName)
    landmarkPath  = os.path.join(DataPath, landmarkDirName)
    # print(landmarkPath)
    list_dirs = os.walk(trainDataPath)
    for _, _, files in list_dirs :
        for f in files:
            if f[-3:] == 'xyz':
                if cnt >= fileCnt : break
                bndFile = f[:-3] + 'bnd'
                xyzData = dataio.loadData(os.path.join(trainDataPath, f))
                bndData = dataio.loadData(os.path.join(landmarkPath, bndFile))
                F = combineData(xyzData, bndData)
                X[cnt, :] = F.ravel()
                cnt += 1
    list_dirs = os.walk(testDataPath)
    cnt = 0
    for _, _, files in list_dirs :
        for f in files:
            if f[-3:] == 'xyz':
                if cnt >= valCnt : break
                bndFile = f[:-3] + 'bnd'
                xyzData = dataio.loadData(os.path.join(testDataPath, f))
                bndData = np.zeros((83, 3))
                F = combineData(xyzData, bndData)
                Y[cnt, :] = F.ravel()
                cnt += 1
    print("Data loaded.\nTrain Shape:", X.shape, " | Test Shape:", Y.shape)
    if printTime:
        print("Cost {} seconds.".format(time.time() - startTime))
    return X, Y

def FetchXYZData(TestDataPath) :
    curPath = os.getcwd()
    FileCnt = 20
    X = np.zeros((FileCnt, 23349))
    list_dirs = os.walk(TestDataPath)
    i = 0
    for root, _, files in list_dirs :
        for f in files:
            if f[-3:] == 'xyz' :
                if i >= FileCnt : break
                print(f)
                xyzData = dataio.loadData(os.path.join(TestDataPath,f))
                bndData = np.zeros((83, 3))
                #print(bndData.shape)
                F = combineData(xyzData, bndData, normalize=False)
                X[i, :] = F.ravel()
                i += 1
    # print(X)
    return X
