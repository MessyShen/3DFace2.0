# coding = utf-8
import os

import numpy as np


def convertBND2XYZ(bndFile, writePath):
    bnd = np.loadtxt(bndFile)
    savePath = bndFile[:-4] + "Export.xyz"
    np.savetxt(savePath, bnd, fmt='%.7f')
    '''
    # TODO: find the mapping relationship
    RE  = right eye,
    LE  = left  eye,
    REB = right eye brow,
    LEB = left  eye brow,
    NS  = Nose,
    OUL = Outer Upper Lips
    IUL = Inner Upper Lips
    
    '''
    SliceList = [0,   8,   16,    26,    36,   48,    55,    60,    65,    68,   83]
    Name      = [  'RE', 'LE', 'REB', 'LEB', 'NS', 'OUL', 'OLL', 'IUL', 'ILL', 'OC'] 
    for i in range(len(Name)):
        savePath = bndFile[:-4] + Name[i] + ".xyz"
        np.savetxt(savePath, bnd[SliceList[i]:SliceList[i+1], :], fmt="%.7f") 


if __name__ == '__main__':
    convertBND2XYZ('FaceAlignment/Experiment/noseA.bnd', 'FaceAlignment/Experiment/')
    # a = np.loadtxt('FaceAlignment/Experiment/noseB.bnd')
    # np.savetxt('FaceAlignment/Experiment/noseB_.bnd', a[:,1:], fmt='%.7f')
