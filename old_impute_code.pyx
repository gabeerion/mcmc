import numpy as np
import sys
import time
cimport numpy as np
cimport cython
np.import_array()

cdef unsigned int NUM_CHARS = 5


cpdef np.ndarray[np.uint_t, ndim=1] wsnr(np.ndarray[np.double_t, ndim=1] arr, unsigned int m):
	cdef unsigned int i, j
	cdef np.ndarray[np.uint_t, ndim=1] reservoir = np.zeros(m, dtype=np.uint)
	cdef np.ndarray[np.double_t, ndim=1] resweights = np.zeros(m, dtype=np.double)
	cdef np.ndarray[np.double_t, ndim=1] replace = np.ones(m, dtype=np.double)
	# Fill first m entries
	j = 0
	for i in xrange(arr.shape[0]):
		if j >= m: break
		if arr[i]>0:
			reservoir[j] = i
			j += 1
	cdef unsigned int pos = i
	# Running totals
	cdef double run_old = np.sum(arr[:m])
	cdef double run
	
	# Initialize reservoir weights
	for i in xrange(m):
		resweights[i] = m*arr[reservoir[i]]/run_old
	
	# Variables for iterating through rest of list
	cdef np.ndarray[np.double_t, ndim=2] rands = np.random.random((2, arr.shape[0]))	# A bunch of random numbers
	cdef double thresh	# Probability with which we'll choose an element
	cdef double rw_old	# Tracks probability of each reservoir element being sampled
	cdef double tk		# Sum of all reservoir weights thus far
	cdef unsigned int lk 	# Number of reservoir items in categories A & B
	cdef double unif	# Weight for reservoir items in category C

	# Iterate through the rest of the list
	for i in xrange(pos,arr.shape[0]):
		run = run_old + arr[i]
		thresh = m*arr[i]/run
		# Update selection weights for reservoir
		tk = 0
		lk = 0
		for j in xrange(m):
			rw_old = resweights[j]
			resweights[j] *= run_old/run
			if resweights[j] >= 1 and rw_old >= 1:
				replace[j] = 0
				lk += 1
			elif resweights[j] < 1 and rw_old >= 1:
				replace[j] = (1-resweights[j])/thresh
				tk += replace[j]
				lk += 1
		if lk < m:
			unif = (1-tk)/(m-lk)
			for j in xrange(m):
				if replace[j] == 1:
					replace[j] = unif
#		print replace, resweights
		# Sample new element
#		print thresh
		if rands[0,i] <= thresh:
			j = weightselect(replace, rands[1,i])
			reservoir[j] = i
			resweights[j] = thresh
		run_old = run
	return reservoir

cpdef np.ndarray[np.uint_t, ndim=1] wsnr_es(np.ndarray[np.double_t, ndim=1] arr, unsigned int m):
	cdef unsigned int i, j
	cdef np.ndarray[np.uint_t, ndim=1] reservoir = np.zeros(m, dtype=np.uint)
	cdef np.ndarray[np.double_t, ndim=1] resweights = np.zeros(m, dtype=np.double)
	cdef np.ndarray[np.double_t, ndim=1] replace = np.ones(m, dtype=np.double)
	# Fill first m entries
	j = 0
	for i in xrange(arr.shape[0]):
		if j >= m: break
		if arr[i]>0:
			reservoir[j] = i
			j += 1
	cdef unsigned int pos = i
	# Running totals
	cdef double run_old = np.sum(arr[:m])
	cdef double run
	
	# Initialize reservoir weights
	for i in xrange(m):
		resweights[i] = m*arr[reservoir[i]]/run_old
	
	# Variables for iterating through rest of list
	cdef np.ndarray[np.double_t, ndim=2] rands = np.random.random((2, arr.shape[0]))	# A bunch of random numbers
	cdef double thresh	# Probability with which we'll choose an element
	cdef double rw_old	# Tracks probability of each reservoir element being sampled
	cdef double tk		# Sum of all reservoir weights thus far
	cdef unsigned int lk 	# Number of reservoir items in categories A & B
	cdef double unif	# Weight for reservoir items in category C
	cdef int reweight = True

	# Iterate through the rest of the list
	for i in xrange(pos,arr.shape[0]):
		run = run_old + arr[i]
		thresh = m*arr[i]/run
		# Update selection weights for reservoir
		tk = 0
		lk = 0
		if reweight:
			for j in xrange(m):
				rw_old = resweights[j]
				resweights[j] *= run_old/run
				if resweights[j] >= 1 and rw_old >= 1:
					replace[j] = 0
					lk += 1
				elif resweights[j] < 1 and rw_old >= 1:
					replace[j] = (1-resweights[j])/thresh
					tk += replace[j]
					lk += 1
			if lk < m:
				unif = (1-tk)/(m-lk)
				for j in xrange(m):
					if replace[j] == 1:
						replace[j] = unif
			if lk == 0:
				reweight = False
#		print replace, resweights
		# Sample new element
#		print thresh
		if rands[0,i] <= thresh:
			j = weightselect(replace, rands[1,i])
			reservoir[j] = i
			resweights[j] = thresh
		run_old = run
	return reservoir

cpdef np.ndarray[np.uint_t, ndim=1] wsnr_simp(np.ndarray[np.double_t, ndim=1] arr, unsigned int m):
	cdef np.ndarray[np.uint_t, ndim=1] reservoir = np.zeros(m, dtype=np.uint)
	cdef np.ndarray[np.uint_t, ndim=1] inds = np.arange(arr.shape[0], dtype=np.uint)
	cdef np.ndarray[np.double_t, ndim=1] probs = np.copy(arr)
	cdef np.ndarray[np.int_t, ndim=1] remaining = np.array((arr>0), dtype=np.int)
	cdef unsigned int i, ind
	cdef unsigned int left = sum(remaining)
	cdef double val
	for i in xrange(m):
		ind = np.random.choice(inds, p=probs)
		val = arr[ind]*1.
		reservoir[i] = ind
		probs[ind] = 0
		remaining[ind] = 0
		left -= 1
		probs /= (1-val)
		print sum(probs), val, (val/left)*left
	return reservoir