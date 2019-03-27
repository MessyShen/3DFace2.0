import os
import numpy as np

def sliceBND(landmark):
    '''
    Slice land mark into following 8 part:
    RE  = right eye,
    LE  = left  eye,
    REB = right eye brow,
    LEB = left  eye brow,
    NS  = Nose,
    OUL = Outer Upper Lips
    IUL = Inner Upper Lips
    OC  = Outer Contour
    '''
    SliceList = [0,   8,   16,    26,    36,   48,    55,    60,    65,    68,   83]
    Name      = [  'RE', 'LE', 'REB', 'LEB', 'NS', 'OUL', 'OLL', 'IUL', 'ILL', 'OC'] 
    ret = []
    for i in range(len(SliceList) - 1):
        ret.append(np.array(landmark[SliceList[i]:SliceList[i+1],:]))
    return ret
    

def EvaluateBND(learnedLandmark, groundTruth, partNum=10):
    # Select left OC and right OC as face distance
    Y_ = np.array(learnedLandmark)
    Y  = np.array(groundTruth)
    Y_ = sliceBND(Y_)
    Y  = sliceBND(Y)
    MaxDis = np.linalg.norm(Y[partNum-1][0, :]-Y[partNum-1][-1 ,:])
    Err = []
    for i in range(partNum):
        Err.append((Y[i] - Y_[i]))
    MeanErr = [np.mean(np.linalg.norm(Err[i], axis=1)) for i in range(partNum)]
    return MeanErr
    
def test():
    learnedLandmark = np.loadtxt("Evaluate/learned.bnd")
    groundTruth     = np.loadtxt("Evaluate/groundTruth.bnd")
    EvaluateBND(learnedLandmark, groundTruth)

if __name__ == '__main__':
    test()