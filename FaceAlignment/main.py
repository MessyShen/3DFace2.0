# coding = utf-8
import os
import time

import dataio
import myICP


def ListTrainFile(rootDir, writeDir):
    curPath = os.getcwd()
    trainPath = os.path.join(curPath, rootDir)
    listDirs = os.walk(trainPath)
    trainFiles = []
    for root, _, files in listDirs:
        for f in files:
            trainFiles.append([os.path.join(root, f), 
                               os.path.join(writeDir, f)])
            prefixFileName = f[:-4]
            landmarkFileName = prefixFileName + '.bnd'
    referenceFaceFile = trainFiles[0][0]
    referenceFace = dataio.loadData(referenceFaceFile)
    print("Select Reference Face:", referenceFaceFile, "...")
    # print(referenceFace)
    totalNum = 10
    for face in trainFiles[:totalNum]:
        targetFace = dataio.loadData(face[0])
        _, _, data = myICP.icp(targetFace, referenceFace, 
                               maxIteration=3,
                               tolerance=0.0001,
                               controlPoints=125)
        dataio.outputData(face[1], data)


def main():
    ListTrainFile('BU3D/train', 'ExportBU3D/train')

if __name__ == '__main__':
    main()
