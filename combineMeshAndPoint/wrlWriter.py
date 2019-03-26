#coding=utf-8

import time, re, linecache
import numpy as np
import random
from sys import argv
import os

def writeWRL(filename, filePath) :
    targetFile = os.path.join(filePath, filename)
    in_file = open(targetFile, "r+")

    f_prefix = filename.split('.')[0]
    out_filename = '{frefix}.wrl'.format(frefix=f_prefix)
    outFile = os.path.join(filePath, out_filename)
    fc_filename = os.path.join(filePath, 'mesh.fc')
    fcinput = open(fc_filename)
    output = open(outFile, "w+")

    output.write("#VRML V2.0 utf8\n#Tricorder Technology plc - 2000\nDEF Tricorder_object Transform {\nchildren [\nShape {\ngeometry IndexedFaceSet {\n")
    output.write("ccw TRUE\nsolid FALSE\nconvex TRUE\n")

    pre_str = "coord Coordinate {\npoint [\n"

    output.write(pre_str)

    cnt = 0

    for line in in_file.readlines():
        s = line
        if s[-1] == '\n' :
            s = s[:-1]
        cd = line.split(' ')[:3]
        output.write(s + ',\n')
        cnt += 1

    print("total cnt{}".format(cnt))

    output.write(" ]\n}\ncoordIndex [\n")

    for line in fcinput.readlines():
        s = line
        if s[-1] == '\n' :
            s = s[:-1]
        output.write(s + ',\n')

    output.write("]\n}\n}\n]\n}")
    fcinput.close()
    in_file.close()
    output.close()
