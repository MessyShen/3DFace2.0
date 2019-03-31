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
    

def EvaluateBND(learnedLandmark, groundTruth, partNum=10, scale_factor=0.70):
    # Select left OC and right OC as face distance
    Y_ = np.array(learnedLandmark)
    Y  = np.array(groundTruth)
    Y_ = sliceBND(Y_)
    Y  = sliceBND(Y)
    MaxDis = np.linalg.norm(Y[partNum-1][0, :]-Y[partNum-1][-1 ,:])
    Err = []
    for i in range(partNum):
        Err.append((Y[i] - Y_[i]))
    MeanErr = [np.mean(np.linalg.norm(Err[i], axis=1))*scale_factor for i in range(partNum)]
    Std     = [np.std(np.linalg.norm(Err[i], axis=1) *scale_factor) for i in range(partNum)]
    max_acc_distance = 5. / scale_factor
    
    Acc_rate = []
    for k in range(partNum):
        Acc_rate.append( 
            np.count_nonzero(
            [1 if np.linalg.norm(i) < max_acc_distance else 0 for i in Err[k]]) / len(Err[k]))
    # print(Acc_rate)
    # print(MeanErr)
    # print(Std)
    return Acc_rate, MeanErr, Std
    
def test():
    learnedLandmark = np.loadtxt("Evaluate/learned.bnd")
    groundTruth     = np.loadtxt("Evaluate/groundTruth.bnd")
    EvaluateBND(learnedLandmark, groundTruth)

def evaluateFaces(learned_faces, ground_truths):
    '''
    learned_faces : [ list of 83*3 array ]
    ground_truths : [ list of 83*3 array ]
    '''
    Acc  = []
    Mean = []
    Std  = []
    for i in zip(learned_faces, ground_truths):
        _acc, _mean, _std = EvaluateBND(i[0], i[1])
        Acc.append(_acc)
        Mean.append(_mean)
        Std.append(_std)
    print(np.mean(Acc, axis=0))
    print(np.mean(Mean, axis=0))
    print(np.mean(Std, axis=0))

if __name__ == '__main__':

    learned_dir = os.path.join(os.getcwd() + '/tempData/Learned/')
    GT_dir = os.path.join(os.getcwd()+'/tempData/GT/')
    learned_faces = []
    ground_truths = []
    learned_walk = os.walk(learned_dir)
    GT_walk = os.walk(GT_dir)

    for root, _, files in learned_walk:
        for f in files:
            a = np.loadtxt(os.path.join(learned_dir, f))
            gt = np.loadtxt(os.path.join(GT_dir, f[:-5]+'.xyz'))
            # print(f)
            learned_faces.append(a[-83:,:])
            ground_truths.append(gt[-83:, :])
            
    evaluateFaces(learned_faces, ground_truths)