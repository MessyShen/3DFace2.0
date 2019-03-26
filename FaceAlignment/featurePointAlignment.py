import numpy as np
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def calcDis(P, Q):
    P = np.array(P)
    Q = np.array(Q)
    dis = np.zeros(P.shape[0])

    for i in range(P.shape[0]):
        dis[i] = np.linalg.norm(P[i] - Q[i], ord = 2)
        
    return dis

def find_optimal_transform(P, Q):
    meanP = np.mean(P, axis = 0)
    meanQ = np.mean(Q, axis = 0)
    P_ = P - meanP
    Q_ = Q - meanQ

    W = np.dot(Q_.T, P_)
    U, S, VT = np.linalg.svd(W)
    R = np.dot(U, VT)
    if np.linalg.det(R) < 0:
       R[2, :] *= -1

    T = meanQ.T - np.dot(R, meanP.T)
    return R, T

def NoseICP(src, dst, face, maxIteration=50, tolerance=0.001):
    A = np.array(face)
    
    P = np.array(src)
    Q = np.array(dst)

    lastErr = 0
    
    for i in range(maxIteration):
        print("Iteration : " + str(i) + " with Err : " + str(lastErr))
        dis = calcDis(P, Q)
        R, T = find_optimal_transform(P, Q)
        A = np.dot(R, A.T).T + np.array([T for j in range(A.shape[0])])
        P = np.dot(R, P.T).T + np.array([T for j in range(P.shape[0])])

        meanErr = np.sum(dis) / dis.shape[0]
        if abs(lastErr - meanErr) < tolerance:
            break
        lastErr = meanErr

        # visualization
        # ax = plt.subplot(1, 1, 1, projection='3d')
        # ax.scatter(P[:, 0], P[:, 1], P[:, 2], c='r')
        # ax.scatter(Q[:, 0], Q[:, 1], Q[:, 2], c='g')
        # plt.show(block = False)
    print('Err:', lastErr)
    # R, T = find_optimal_transform(A,np.array(src))
    # print(R)
    # print("====")
    # print(T)
    return A, P # np.dot(R, src.T).T + np.array([T for j in range(src.shape[0])])
