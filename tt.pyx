import numpy as np
cimport numpy as np
cimport cython
np.import_array()

"""def tt(np.ndarray arr):
    cdef double ti = 0
    cdef double tv = 0
    cdef np.ndarray purines = np.array(('A','G'))
    cdef np.ndarray pyrimidines = np.array(('C','T'))
    cdef np.ndarray pps = (arr == purines[0]) | (arr == purines[1])    #0 for pyrimidine, 1 for purine
    cdef np.ndarray dashes = arr == '-'

    cdef int i,j
    cdef int allen = arr.shape[0]

    cdef np.ndarray muts = np.zeros((), dtype=np.bool)
    cdef np.ndarray transv = np.zeros((), dtype=np.bool)
    cdef np.ndarray anydash = np.zeros((), dtype=np.bool)
    cdef np.ndarray transitions = np.zeros((), dtype=np.bool)
    cdef np.ndarray transversions = np.zeros((), dtype=np.bool)

    for i in xrange(allen-1):
        for j in xrange(i+1, allen):
            muts = (arr[i]!=arr[j])
            transv = (pps[i]!=pps[j])
            anydash = (dashes[i]|dashes[j])
            transitions = muts & ~transv & ~anydash
            transversions = muts & transv & ~anydash
            print muts, transv, anydash
            ti += sum(transitions)
            tv += sum(transversions)
    return ti/tv"""

def tt(np.ndarray chars):
    cdef np.ndarray[np.uint8_t,cast=True, ndim=2] pps = ((chars == 'A') | (chars == 'G'))    #0 for pyrimidine, 1 for purine
    cdef np.ndarray[np.uint8_t,cast=True, ndim=2] dashes = chars == '-'
    cdef np.ndarray[np.int64_t, ndim=2] chints = np.array([map(ord, r) for r in chars],dtype=np.int)
    cdef np.ndarray[np.int64_t, ndim=1] ti = np.zeros(chars.shape[1], dtype=np.int)
    cdef np.ndarray[np.int64_t, ndim=1] tv = np.zeros(chars.shape[1], dtype=np.int)

    cdef int i,j
    cdef int allen = chars.shape[0]

    cdef np.ndarray[np.uint8_t,cast=True, ndim=1] muts = np.zeros(allen, dtype=np.bool)
    cdef np.ndarray[np.uint8_t,cast=True, ndim=1] transv = np.zeros(allen, dtype=np.bool)
    cdef np.ndarray[np.uint8_t,cast=True, ndim=1] anydash = np.zeros(allen, dtype=np.bool)
    cdef np.ndarray[np.uint8_t,cast=True, ndim=1] transitions = np.zeros(allen, dtype=np.bool)
    cdef np.ndarray[np.uint8_t,cast=True, ndim=1] transversions = np.zeros(allen, dtype=np.bool)

    for i in xrange(allen-1):
        for j in xrange(i+1, allen):
            muts = (chints[i]!=chints[j])
            transv = (pps[i]!=pps[j])
            anydash = (dashes[i]|dashes[j])
            transitions = muts & ~transv & ~anydash
            transversions = muts & transv & ~anydash
            ti += transitions
            tv += transversions
    return float(sum(ti))/sum(tv)

@cython.boundscheck(False)
def tti(np.ndarray[np.int_t, ndim=2] chints):
    cdef np.ndarray[np.uint8_t, cast=True, ndim=2] pps = ((1 == chints) | (chints == 2))    #0 for pyrimidine, 1 for purine
    cdef np.ndarray[np.uint8_t, cast=True, ndim=2] dashes = chints == 0
    cdef np.ndarray[np.int_t] ti = np.zeros(chints.shape[1], dtype=np.int)
    cdef np.ndarray[np.int_t] tv = np.zeros(chints.shape[1], dtype=np.int)

    cdef int i,j
    cdef int allen = chints.shape[0]

    cdef np.ndarray[np.uint8_t, cast=True, ndim=1] muts = np.zeros(allen, dtype=np.bool)
    cdef np.ndarray[np.uint8_t, cast=True, ndim=1] transv = np.zeros(allen, dtype=np.bool)
    cdef np.ndarray[np.uint8_t, cast=True, ndim=1] anydash = np.zeros(allen, dtype=np.bool)
    cdef np.ndarray[np.uint8_t, cast=True, ndim=1] transitions = np.zeros(allen, dtype=np.bool)
    cdef np.ndarray[np.uint8_t, cast=True, ndim=1] transversions = np.zeros(allen, dtype=np.bool)

    for i in xrange(allen-1):
        for j in xrange(i+1, allen):
            muts = (chints[i]!=chints[j])
            transv = (pps[i]!=pps[j])
            anydash = (dashes[i]|dashes[j])
            transitions = muts & ~transv & ~anydash
            transversions = muts & transv & ~anydash
            ti += transitions
            tv += transversions
    return float(sum(ti))/sum(tv)


def tt1(np.ndarray chars):
    cdef np.ndarray[np.uint8_t,cast=True, ndim=2] pps = ((chars == 'A') | (chars == 'G'))    #0 for pyrimidine, 1 for purine
    return pps

def tt2(np.ndarray chints):
    cdef np.ndarray pps = ((1 == chints) | (chints == 2))
    return pps

#NUCMAP = {'-':0, 'A':1, 'a':1, 'G':2, 'g':2, 'C':3, 'c':3, 'T':4, 't':4}
def ttratio(np.ndarray[np.int_t, ndim=2] chints):
    cdef double ti = 0
    cdef double tv = 0
    cdef int i, j
    cdef int allen = chints.shape[0]
    cdef int seqlen = chints.shape[1]
    cdef int char1, char2

    for i in xrange(allen-1):
        for j in xrange(i+1, allen):
            for k in xrange(seqlen):
                char1 = chints[i,k]
                char2 = chints[j,k]
                if char1 == 0 or char2 == 0:
                    continue
                elif char1 == 1:
                    if char2 == 1: continue
                    elif char2 == 2: ti += 1
                    elif char2 == 3: tv += 1
                    elif char2 == 4: tv += 1
                elif char1 == 2:
                    if char2 == 2: continue
                    elif char2 == 1: ti += 1
                    elif char2 == 3: tv += 1
                    elif char2 == 4: tv += 1
                elif char1 == 3:
                    if char2 == 3: continue
                    elif char2 == 4: ti += 1
                    elif char2 == 1: tv += 1
                    elif char2 == 2: tv += 1
                elif char1 == 4:
                    if char2 == 4: continue
                    elif char2 == 3: ti += 1
                    elif char2 == 1: tv += 1
                    elif char2 == 2: tv += 1
    return ti/tv



"""def tt2(al):
    chars = np.array(al)
    pps = (chars == 'A') | (chars == 'G')    #0 for pyrimidine, 1 for purine
    dashes = chars == '-'
    ti = np.zeros(chars.shape[1])
    tv = np.zeros(chars.shape[1])
    for i in xrange(len(al)-1):
        for j in xrange(i+1, len(al)):
            muts = (chints[i]!=chints[j]).astype('bool')
            transv = (pps[i]!=pps[j]).astype('bool')
            anydash = (dashes[i]|dashes[j]).astype('bool')
            transitions = muts & ~transv & ~anydash
            transversions = muts & transv & ~anydash
            ti += transitions
            tv += transversions
    return float(sum(ti))/sum(tv)"""