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

