import cython
import numpy as np
cimport numpy as np

cdef extern from "math.h":
    float sqrt(float x)
    int floor(float x)

@cython.boundscheck(False)
@cython.wraparound(False)
def cy_rdf(unsigned int n_atoms, unsigned int n_bins, float r_min, float r_max,
        np.ndarray[float, ndim=2] xyz, np.ndarray[float, ndim=1] box,
        np.ndarray[unsigned int, ndim=1] g_r):
    """
    """
    cdef unsigned int i, j, k, added
    cdef float dist, diff, image, temp
    cdef np.ndarray[float, ndim=2] distances = np.zeros(
            dtype="float32", shape=(n_atoms, n_atoms-1))

    # calculate pairwise distances
    for i in xrange(xyz.shape[0]):
        added = 0
        for j in xrange(xyz.shape[0]):
            if i != j:
                dist = 0
                for k in xrange(3):
                    diff = xyz[i, k] - xyz[j, k]
                    temp = diff / box[k]
                    if temp >= 0.5:
                        image = 1.0
                    elif temp < -0.5:
                        image = -1.0
                    else:
                        image = 0.0
                    diff -= box[k] * image
                    dist += diff * diff
                distances[i, added] = sqrt(dist)
                added += 1

    cdef unsigned int bin_num
    cdef float binsize = (r_max - r_min) / n_bins

    # do distance histogram
    for i in xrange(n_atoms):
        for j in xrange(n_atoms-1):
            bin_num = floor((distances[i, j] - r_min) / binsize)
            g_r[bin_num] += 1

    return g_r
