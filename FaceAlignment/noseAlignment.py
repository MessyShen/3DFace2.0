'''
Align face according to nose landmark...
'''

import numpy as np

import myICP


def rotateByNose(faceA, noseA, faceB, noseB):
    '''
    Function: 
        rotate A to B by its nose

    Input : 
        faceA : shape(Na, 3)
        faceB : shape(Nb, 3)
        noseA : (12, 3)
        noseB : (12, 3)
    
    Return: 
        retNose
    '''
    R, T, A = myICP.icp(noseA, noseB, maxIteration=10, controlPoints=5)
    face = np.array(faceA)
    nose = np.array(noseA)
    retFace = np.dot(R, face.T).T + np.array([T for j in range(faceA.shape[0])])
    retNose = np.dot(R, nose.T).T + np.array([T for j in range(noseA.shape[0])])

    return retFace, retNose 


# Test
if __name__ == '__main__':
    faceA = np.loadtxt("FaceAlignment/Experiment/faceA.xyz")
    faceB = np.loadtxt("FaceAlignment/Experiment/faceB.xyz")
    noseA = np.loadtxt("FaceAlignment/Experiment/noseA.xyz")
    noseB = np.loadtxt("FaceAlignment/Experiment/noseB.xyz")
    faceA_, noseA_  = rotateByNose(faceA, noseA, faceB, noseB)
    np.savetxt("FaceAlignment/Experiment/faceA_.xyz", faceA_)
