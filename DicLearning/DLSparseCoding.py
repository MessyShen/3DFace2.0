# coding=utf-8
from time import time

import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
# import readin
# from LCC_DictionaryLearning import MiniBatchDictionaryLearning
from sparseDL import MiniBatchDictionaryLearning
# from simpleDictLearning import MiniBatchDictionaryLearning
# from originDictLearning import MiniBatchDictionaryLearning
from sklearn.feature_extraction.image import extract_patches_2d
from sklearn.feature_extraction.image import reconstruct_from_patches_2d
#from sklearn.utils.testing import SkipTest
from sklearn.utils.fixes import sp_version
from sklearn import preprocessing
from readin import FetchAllData, FetchXYZData, FetchBU3DData
import dataio

# data = FetchAllData("TrainingSet")
data, val = FetchBU3DData("BU3D", printTime=True)
# print("dataShape", data.shape)
scaler = preprocessing.StandardScaler().fit(data)
data = scaler.transform(data)
data /= 500.0

# Learning the Dictionary

print("learning the Dictionary...")
t0 = time()
# print(data)
dico = MiniBatchDictionaryLearning(n_components=300, alpha=1./21330, n_iter=500, verbose=True, batch_size=1)
dico.set_params(dict_init=np.loadtxt('tempData/dictionary.txt'))
V = dico.fit(data).components_
dt = time() - t0
print("done in %.2fs" % dt)
print("dictionary:")
#- print(V)
print(np.linalg.norm(V, axis=1, ord=2))


#Save Dictionary To File
np.savetxt("tempData/dictionary.txt", np.array(V))
'''
outDic = open("tempData/dictionary.txt", "w")
for x in V:
    for y in x :
        outDic.write(str(y) + ' ')
    outDic.write('\n')
outDic.close()
'''

print("Reconstructing...")



# mean = np.zeros(test.shape[0])
# std = np.zeros(test.shape[0])
# for i in range(test.shape[0]) :
#     mean[i] = np.mean(test[i,:], axis=0) 
#     test[i, :] -= mean[i]
#     std[i] = np.std(test[i,:], axis=0)
#     # i -= np.mean(X, axis=0)
#     test[i, :] /= std[i]
    
# intercept = np.mean(test, axis=0)
# test -= 
dico.set_params(transform_algorithm='lars')
testData = scaler.transform(val)
print("test data shape:", testData.shape)
testData /= 500.0
code = dico.transform(testData)
Rec = np.dot(code, V)
print("Rec.shape", Rec.shape)
Rec *= 500
output = scaler.inverse_transform(Rec)

#write code to file
outCode = open("newcode.txt", "w")
for i in code:
    for j in i :
        outCode.write(str(j) + ' ')
    outCode.write('\n')
outCode.close()


for i in range(testData.shape[0]):
    fileName = "tempData/facenew"+str(i)+".xyz"
    dataio.outPutFace(fileName, output[i, :])
# for i in range(test.shape[0]):
    # code = dico.transform(test)
    # Rec = np.dot(code, V)
    # print(Rec)
    # for i in range(test.shape[0]) :
    #     # Rec[i, :] *= std
    #     # Rec[i, :] += mean
    # dataio.outPutFace("learned.xyz", Rec)

dt = time() - t0
print("done in %.2fs" % dt)