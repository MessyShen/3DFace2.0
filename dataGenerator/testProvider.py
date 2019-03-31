import provider
import numpy as np


if __name__ == '__main__' :
    a = np.loadtxt("dataGenerator/data/F.xyz")
    a = np.reshape(a, (1, -1, 3))
    b = provider.rotate_point_cloud_xyz(a, max_range=0.1*np.pi)
    np.savetxt("dataGenerator/data/F_rotate.xyz", b, fmt="%.6f")
