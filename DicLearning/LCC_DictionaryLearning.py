import time
import sys
import itertools

from math import sqrt, ceil

import numpy as np
import scipy as sp
import scipy.linalg as LA

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import (check_array, check_random_state, gen_even_slices,
                     gen_batches)#, _get_n_jobs)
from sklearn.utils.extmath import randomized_svd, row_norms
from sklearn.linear_model import Lars
from sklearn.utils.validation import check_is_fitted

#def _sparse_encode(X, dictionary, gram, cov, regularization=Nose, c)

def sparse_encode(X, dictionary, max_iter = 1000, n_nonzero_coefs=None) :
    dictionary = check_array(dictionary)
    X = check_array(X)

    n_k, _ = dictionary.shape
    Y = np.eye(n_k, dtype=float)
    # print("gene_Y...")
    for i in range(n_k):
        #print(i)
        Dj = dictionary[i,:].reshape(-1,1)
        Y[i][i] = np.linalg.norm(Dj - X.T, ord=2) ** 2
    # print("Y_generated ", Y.shape)
    Y_ = np.linalg.inv(Y)
    dictionary_ = np.dot(dictionary.T, Y_).T

    n_samples, n_features = X.shape
    n_components = dictionary_.shape[0]

    gram = np.dot(dictionary_, dictionary_.T)
    cov = np.dot(dictionary_, X.T)
    #lars
    regularization = n_nonzero_coefs
    if regularization is None :
        regularization = min(max(n_features / 10, 1), n_components)
    
    lars = Lars(fit_intercept=False, normalize=False,
                precompute=gram, n_nonzero_coefs=int(regularization),
                fit_path=False)
    lars.fit(dictionary_.T, X.T, Xy=cov)
    new_code = lars.coef_
    ret_code = np.dot(Y_, new_code.T).T
    # print("ret shape", ret_code.shape)
    return ret_code


def update_dictionary(dictionary, Y, code, random_state=None):
    n_components = len(code)
    n_samples = Y.shape[0]
    random_state = check_random_state(random_state)
    R = -np.dot(dictionary, code)
    R += Y
    R = np.asfortranarray(R)
    ger, = LA.get_blas_funcs(('ger',),(dictionary, code))
    for k in range(n_components) :
        R = ger(1.0, dictionary[:, k], code[k, :], a=R, overwrite_a=True)
        dictionary[:, k] = np.dot(R, code[k, :].T)
        # Scale k'th atom
        atom_norm_square = np.dot(dictionary[:, k], dictionary[:, k])
        if atom_norm_square < 1e-20:
            dictionary[:, k] = random_state.randn(n_samples)
            # Setting corresponding coefs to 0
            code[k, :] = 0.0
            dictionary[:, k] /= sqrt(np.dot(dictionary[:, k],
                                            dictionary[:, k]))
        else:
            dictionary[:, k] /= sqrt(atom_norm_square)
            # R <- -1.0 * U_k * V_k^T + R
            R = ger(-1.0, dictionary[:, k], code[k, :], a=R, overwrite_a=True)
    return dictionary

def dict_learning_online(X, n_components=2, alpha=1, n_iter=100,
                         return_code=True, batch_size=1, dict_init=None, random_state=None,
                         shuffle=False, inner_stats=None, iter_offset=0, n_jobs=1):
    '''    
    X : (n_samples, n_components)
    '''

    if n_components is None:
        n_components = X.shape[1]

    t0 = time.time()
    n_samples, n_features = X.shape
    alpha = float(alpha)
    random_state = check_random_state(random_state)
    #random_state = check_random_state(random_state)

    if dict_init is not None:
        dictionary = dict_init
    else :
        #init V with SVD of X
        # _, S, dictionary = randomized_svd(X, n_components,
        #                                   random_state=random_state)
        # dictionary = S[:, np.newaxis] * dictionary
        dictionary = np.random.random(X.shape)
    r = len(dictionary)


    if n_components <= r:
        dictionary = dictionary[:n_components, :]
    else:
        dictionary = np.r_[dictionary,
                           np.zeros((n_components - r, dictionary.shape[1]))]

    if shuffle:
        X_train = X.copy()
        random_state.shuffle(X_train)
    else :
        X_train = X

    # print("Dictionary Init")
    # for dic in dictionary:
    #     print(dic)
    # print("Dictionary Shape:", dictionary.shape)
    print("X.shape ", X_train.shape)
    print("Dinctionay Shape", dictionary.shape)

    dictionary = check_array(dictionary.T, order='F', dtype=np.float64, copy=False)
    X_train = check_array(X_train, order='C', dtype=np.float64, copy=False)

    batches = gen_batches(n_samples, batch_size)
    batches = itertools.cycle(batches)

    if inner_stats is None:
        A = np.zeros((n_components, n_components))
        B = np.zeros((n_features, n_components))

    ii = iter_offset - 1

    for ii, batch in zip(range(iter_offset, iter_offset + n_iter), batches):
        this_X = X_train[batch] 
        dt = (time.time() - t0)
        print("now at iter {} of {}".format(ii, n_iter)) 
        this_code = sparse_encode(this_X, dictionary.T).T

        # print("ii: ", ii)

        if ii < batch_size - 1:
            theta = float((ii + 1) * batch_size)
        else :
            theta = float(batch_size ** 2 + ii + 1 - batch_size)
        beta = (theta + 1 - batch_size) / (theta + 1)

        A *= beta
        A += np.dot(this_code, this_code.T)
        A += np.diag(this_code.ravel()) * 2
        # for i in this_code.
        B *= beta
        B += np.dot(this_X.T, this_code.T)
        B += 2 * np.dot(this_X.T, np.abs(this_code.T))

        dictionary = update_dictionary(dictionary, B, A)

    return dictionary.T, (A, B), ii - iter_offset + 1

class SparseCodingMixin(TransformerMixin):
    """Sparse coding mixin"""

    def _set_sparse_coding_params(self, n_components,
                                  transform_algorithm='omp',
                                  transform_n_nonzero_coefs=None,
                                  transform_alpha=None, split_sign=False,
                                  n_jobs=1):
        self.n_components = n_components
        self.transform_algorithm = transform_algorithm
        self.transform_n_nonzero_coefs = transform_n_nonzero_coefs
        self.transform_alpha = transform_alpha
        self.split_sign = split_sign
        self.n_jobs = n_jobs

    def transform(self, X):
        check_is_fitted(self, 'components_')

        X = check_array(X)
        n_samples, n_features = X.shape
        print("fit: X with shape", X.shape)
        code = np.zeros((n_samples, self.n_components))
        # for i in range(10000):
        #     if i % 1000 == 0: print("...", i/100, "%")
        code[0, :] = sparse_encode(
            np.reshape(X[0, :], (1, -1)), self.components_, 
            n_nonzero_coefs=self.transform_n_nonzero_coefs)

        if self.split_sign:
            # feature vector is split into a positive and negative side
            n_samples, n_features = code.shape
            split_code = np.empty((n_samples, 2 * n_features))
            split_code[:, :n_features] = np.maximum(code, 0)
            split_code[:, n_features:] = -np.minimum(code, 0)
            code = split_code

        return code

class MiniBatchDictionaryLearning(BaseEstimator, SparseCodingMixin):
    def __init__(self, n_components=None, alpha=1, n_iter=1000,
                 fit_algorithm='lasso_lars', n_jobs=1, batch_size=1,
                 shuffle=True, dict_init=None, transform_algorithm='omp',
                 transform_n_nonzero_coefs=5, transform_alpha=None,
                 verbose=False, split_sign=False, random_state=None):

        self._set_sparse_coding_params(n_components, transform_algorithm,
                                       transform_n_nonzero_coefs,
                                       transform_alpha, split_sign, n_jobs)
        self.alpha = alpha
        self.n_iter = n_iter
        self.fit_algorithm = fit_algorithm
        self.dict_init = dict_init
        self.verbose = verbose
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.split_sign = split_sign
        self.random_state = random_state

    def fit(self, X, y=None):
        random_state = check_random_state(self.random_state)
        X = check_array(X)

        U, (A, B), self.n_iter_ = dict_learning_online(
            X, self.n_components, self.alpha, self.n_iter)
        print("iter", self.n_iter_)
        self.components_ = U
        # Keep track of the state of the algorithm to be able to do
        # some online fitting (partial_fit)
        self.inner_stats_ = (A, B)
        self.iter_offset_ = self.n_iter
        return self