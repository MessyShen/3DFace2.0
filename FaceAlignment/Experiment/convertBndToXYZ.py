# coding = utf-8
import os

import numpy as np


def convertBND2XYZ(bndFile, writePath, saveAll=False, saveFullBND=False):
    '''
    RE  = right eye,
    LE  = left  eye,
    REB = right eye brow,
    LEB = left  eye brow,
    NS  = Nose,
    OUL = Outer Upper Lips
    IUL = Inner Upper Lips
    OC  = Outer Contour
    '''
    bnd = np.loadtxt(bndFile)
    if saveFullBND == True:    
        savePath = bndFile[:-4] + "Export.xyz"
        np.savetxt(savePath, bnd, fmt='%.7f')
    
    SliceList = [0,   8,   16,    26,    36,   48,    55,    60,    65,    68,   83]
    Name      = [  'RE', 'LE', 'REB', 'LEB', 'NS', 'OUL', 'OLL', 'IUL', 'ILL', 'OC'] 
    if saveAll == True:
        for i in range(len(Name)):
            savePath = bndFile[:-4] + Name[i] + ".xyz"
            np.savetxt(savePath, bnd[SliceList[i]:SliceList[i+1], :], fmt="%.7f") 
    
    nosePath = savePath = bndFile[:-4] + ".xyz"
    np.savetxt(nosePath, bnd[36:48, :], fmt="%.7f") 

if __name__ == '__main__':
    convertBND2XYZ('FaceAlignment/Experiment/noseA.bnd', 'FaceAlignment/Experiment/')
    convertBND2XYZ('FaceAlignment/Experiment/noseB.bnd', 'FaceAlignment/Experiment/')
    # a = np.loadtxt('FaceAlignment/Experiment/noseA.bnd')
    # np.savetxt('FaceAlignment/Experiment/noseA.bnd', a[:,1:], fmt='%.7f')
