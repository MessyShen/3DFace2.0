#coding=utf-8

import time, re, linecache
import numpy as np
import random
from sys import argv

def writeWRL(filename) :
    in_file = open(filename, "r+")

    f_prefix = filename.split('.')[0]
    out_filename = '{frefix}.wrl'.format(frefix=f_prefix)
    fc_filename = '{prefix}.fc'.format(prefix=f_prefix)
    fcinput = open(fc_filename)
    output = open(out_filename, "w+")

    output.write("#VRML V2.0 utf8\n#Tricorder Technology plc - 2000\nDEF Tricorder_object Transform {\nchildren [\nShape {\ngeometry IndexedFaceSet {\n")
    output.write("ccw TRUE\nsolid FALSE\nconvex TRUE\n")

    pre_str = "coord Coordinate {\npoint [\n"

    output.write(pre_str)

    cnt = 0

    for line in in_file.readlines():
        s = line
        cd = line.split(' ')[:3]
        if s[-1] == '\n' :
            s = s[:-1]
        output.write(cd[0] + ' ' + cd[1] + ' '  + cd[2] + ',\n')
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

if __name__ == '__main__' :
    writeWRL('Resampling/F0001.xyz')