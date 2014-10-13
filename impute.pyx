import numpy as np
import sys
import time
cimport numpy as np
cimport cython
np.import_array()

cdef unsigned int NUM_CHARS = 5
ORDERFUNC = np.min

@cython.boundscheck(False)
cpdef np.ndarray[np.uint_t, ndim=2] pdn(np.ndarray[np.int_t, ndim=2] al):
	cdef unsigned int allen = al.shape[0]
	cdef unsigned int seqlen = al.shape[1]
	cdef np.ndarray[np.uint_t, ndim=2] pd = np.zeros((al.shape[0], al.shape[0]), dtype=np.uint)
	cdef unsigned int i,j,k
	for i in xrange(allen):
		for j in xrange(i+1,allen):
			for k in xrange(seqlen):
				if al[i,k]!=al[j,k]: pd[i,j] += 1
	return pd | np.transpose(pd)

cpdef np.ndarray[np.int_t, ndim=2] pssm(np.ndarray[np.int_t, ndim=2] al):
	cdef unsigned int allen = al.shape[0]
	cdef unsigned int seqlen = al.shape[1]
	cdef np.ndarray[np.int_t, ndim=2] mat = np.zeros((NUM_CHARS, seqlen), dtype=np.int)
	cdef unsigned int i,j
	for i in xrange(allen):
		for j in xrange(seqlen):
			mat[al[i,j],j] += 1
	return mat

cpdef np.ndarray[np.uint_t, ndim=1] rangeselect(np.ndarray[np.float_t, ndim=1] arr, unsigned int num):
	cdef np.ndarray[np.float_t, ndim=1]rands = np.random.random(num)
	cdef unsigned int l = rands.shape[0]
	cdef unsigned int i
	cdef np.ndarray[np.uint_t, ndim=1] mat = np.zeros(l, dtype=np.uint)
	for i in xrange(l):
		mat[i] = weightselect(arr, rands[i])
	return mat

cpdef unsigned int weightselect(np.ndarray[np.float_t, ndim=1] arr, float rand):
	cdef unsigned int i
	for i in xrange(arr.shape[0]):
		rand -= arr[i]
		if rand <= 0: return i


cpdef unsigned int wsg(np.ndarray[np.float_t, ndim=1] arr, float rand):
	cdef unsigned int i
	rand *= np.sum(arr)
	for i in xrange(arr.shape[0]):
		rand -= arr[i]
		if rand <= 0: return i

cpdef unsigned int wsw(np.ndarray[np.float_t, ndim=1] arr, float rand, float weight):
	cdef unsigned int i
	rand *= weight
	for i in xrange(arr.shape[0]):
		rand -= arr[i]
		if rand <= 0: return i

cpdef np.ndarray[np.uint_t, ndim=1] wsnr(np.ndarray[np.double_t, ndim=1] arr, unsigned int m):
	cdef np.ndarray[np.double_t, ndim=1] weights = np.copy(arr)
	cdef np.ndarray[np.uint_t, ndim=1] reservoir = np.zeros(m, dtype=np.uint)
	cdef  np.ndarray[np.double_t, ndim=1] rands = np.random.random(m)
	cdef float total = np.sum(weights)
	cdef unsigned int i
	for i in xrange(m):
		reservoir[i] = wsw(weights, rands[i], total)
		total -= weights[reservoir[i]]
		weights[reservoir[i]] = 0
	return reservoir

cdef float random(n):
	pass

cdef double npsum(np.ndarray[np.float_t,ndim=1] arr):
	cdef double s = 0
	cdef unsigned int i
	for i in xrange(arr.shape[0]):
		s += arr[i]
	return s

@cython.boundscheck(False)
def impute(np.ndarray[np.int_t, ndim=2] al, unsigned int imps, orderfunc=ORDERFUNC):
	cdef unsigned int allen = al.shape[0]
	cdef unsigned int seqlen = al.shape[1]
	cdef np.ndarray[np.uint_t, ndim=2] pd = pdn(al)
	pd[np.diag_indices(pd.shape[0])] = sys.maxint
	cdef np.ndarray[np.uint_t, ndim=1] distances = orderfunc(pd, axis=0).astype(np.uint)
	cdef np.ndarray[np.int_t, ndim=2] ps = pssm(al)
	cdef np.ndarray[np.float_t, ndim=1] mutprobs = np.ones(seqlen, dtype=float)*allen
	mutprobs -= np.max(ps, axis=0)
	mutprobs /= np.sum(mutprobs)

	#Time to actually impute
	cdef np.ndarray[np.int_t, ndim=2] new = np.zeros((allen+imps,seqlen), dtype=np.int)
	new[:allen] = al
	cdef unsigned int i, j, dist, ind
	cdef float total
	cdef np.ndarray[np.float_t, ndim=2] rands = np.random.random((2,imps))
	cdef np.ndarray[np.float_t, ndim=2] wsrands = np.random.random((imps, seqlen))
	cdef np.ndarray[np.uint_t, ndim=1] sites = np.zeros(seqlen, dtype=np.uint)
	cdef np.ndarray[np.float_t, ndim=1] siteprobs = np.zeros(NUM_CHARS)
	for i in xrange(imps):
		dist = distances[int(rands[0,i]*allen)]
		ind = int(rands[1,i]*(allen+i))
		new[allen+i] = new[ind]
		sites[:dist] = wsnr(mutprobs, dist)
		for j in xrange(dist):
			siteprobs[:] = ps[:,sites[j]]
			total = allen-siteprobs[new[allen+i,sites[j]]]
			siteprobs[new[allen+i,sites[j]]] = 0
			new[allen+i,sites[j]] = wsw(siteprobs, wsrands[i,j], total)
#		print np.sum(new[allen+i]!=new[ind]), dist
	return new

# Imputation with random decreases in edge distances
@cython.boundscheck(False)
def impute_shrink(np.ndarray[np.int_t, ndim=2] al, unsigned int imps, orderfunc=ORDERFUNC):
	cdef unsigned int allen = al.shape[0]
	cdef unsigned int seqlen = al.shape[1]
	cdef np.ndarray[np.uint_t, ndim=2] pd = pdn(al)
	pd[np.diag_indices(pd.shape[0])] = sys.maxint
	cdef np.ndarray[np.uint_t, ndim=1] distances = orderfunc(pd, axis=0).astype(np.uint)
	cdef np.ndarray[np.int_t, ndim=2] ps = pssm(al)
	cdef np.ndarray[np.float_t, ndim=1] mutprobs = np.ones(seqlen, dtype=float)*allen
	mutprobs -= np.max(ps, axis=0)
	mutprobs /= np.sum(mutprobs)

	#Time to actually impute
	cdef np.ndarray[np.int_t, ndim=2] new = np.zeros((allen+imps,seqlen), dtype=np.int)
	new[:allen] = al
	cdef unsigned int i, j, dist, ind
	cdef float total
	cdef np.ndarray[np.float_t, ndim=2] rands = np.random.random((3,imps))
	cdef np.ndarray[np.float_t, ndim=2] wsrands = np.random.random((imps, seqlen))
	cdef np.ndarray[np.uint_t, ndim=1] sites = np.zeros(seqlen, dtype=np.uint)
	cdef np.ndarray[np.float_t, ndim=1] siteprobs = np.zeros(NUM_CHARS)
	for i in xrange(imps):
		dist = int(rands[2,i]*distances[int(rands[0,i]*allen)])
		ind = int(rands[1,i]*(allen+i))
		new[allen+i] = new[ind]
		sites[:dist] = wsnr(mutprobs, dist)
		for j in xrange(dist):
			siteprobs[:] = ps[:,sites[j]]
			total = allen-siteprobs[new[allen+i,sites[j]]]
			siteprobs[new[allen+i,sites[j]]] = 0
			new[allen+i,sites[j]] = wsw(siteprobs, wsrands[i,j], total)
#		print np.sum(new[allen+i]!=new[ind]), dist
	return new