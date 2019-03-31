# coding=utf-8
from time import time

import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
# import readin
# from LCC_DictionaryLearning import MiniBatchDictionaryLearning
from sparseDL import MiniBatchDictionaryLearning
# from simpleDictLearning import MiniBatchDictionaryLearning
# from sklearn.decomposition import MiniBatchDictionaryLearning
from sklearn.feature_extraction.image import extract_patches_2d
from sklearn.feature_extraction.image import reconstruct_from_patches_2d
#from sklearn.utils.testing import SkipTest
from sklearn.utils.fixes import sp_version
from sklearn import preprocessing
from readin import FetchAllData, FetchXYZData, FetchBU3DData
import dataio

# data = FetchAllData("TrainingSet")
data, val, valGT = FetchBU3DData("BU3D", printTime=True, valCnt=200)
# print("dataShape", data.shape)
scaler = preprocessing.StandardScaler().fit(data)
data = scaler.transform(data)
data /= 500.0

dico = MiniBatchDictionaryLearning(n_components=300, alpha=1./21330, n_iter=0, verbose=True, batch_size=1)
dico.set_params(dict_init=np.loadtxt('tempData/dictionary.txt'))
V = dico.fit(data).components_
print("Dictionary Loaded...")

t0 = time()
print("Reconstructing...")

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
outCode = open("code.txt", "w")
for i in code:
    for j in i :
        outCode.write(str(j) + ' ')
    outCode.write('\n')
outCode.close()

for i in range(testData.shape[0]):
    fileName = "tempData/Learned/Face"+str(i)+"_.xyz"
    groundT  = "tempData/GT/Face"+str(i)+".xyz"
    dataio.outPutFace(groundT, valGT[i, :])
    dataio.outPutFace(fileName, output[i, :])

dt = time() - t0
print("Reconstruction done in %.2fs" % dt)
