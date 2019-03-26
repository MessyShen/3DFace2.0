import os
import numpy as np
import wrlWriter

def fetchOriginLandmark(fileName, landmarkPath="BU3D/landmarks", writePath="combineMeshAndPoint"):
    '''
    fileName:"F000x_xxx_xxx.txt"
    '''
    landmarkPath = "BU3D/landmarks"
    targetFile = os.path.join(landmarkPath, fileName[:-3]+"bnd")
    X = np.loadtxt(targetFile)
    writeFile = os.path.join(writePath, fileName[:-4]+"_origin.xyz")
    np.savetxt(writeFile, X)


def sliceBndFromFullOutput(filePath) :
    X = np.loadtxt(filePath)
    bnd = X[-83:, :]
    writeFile = filePath[:-4]+"_learned.xyz"
    np.savetxt(writeFile, bnd)

def combineMesh2Point(fileName, referenceMesh="mesh.fc"):
    wrlWriter.writeWRL(fileName, filePath="combineMeshAndPoint/")

if __name__ == '__main__':
    fetchOriginLandmark("F0001_DI02WH_F3D.xyz")
    sliceBndFromFullOutput("combineMeshAndPoint/F0001_DI02WH_F3D.xyz")
    combineMesh2Point("F0001_DI02WH_F3D.xyz")
