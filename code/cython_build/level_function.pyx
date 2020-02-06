cimport cython

import numpy as np
cimport numpy as np

@cython.boundscheck(False)
@cython.wraparound(False)
def cy_medal_organizer(np.double_t[:] d,np.double_t[:] c, np.int64_t [:] l, int n):
    cdef int ind = 0
    l[0] = 0
    cdef int i
    for i in range(1, n):
        if d[i] == d[i-1] and c[i] == c[i-1]:
            l[i] = l[i-1]
        elif d[i] == d[ind]:
            l[i] = l[i-1] + 1
        elif c[i] < c[ind]:
            l[i] = 0
            ind = i
        else:
            l[i] = 1
            for j in range(1, i):
                if d[i] > d[j] and c[i] > c[j] and l[j] >= l[i]:
                    l[i] = l[j] + 1