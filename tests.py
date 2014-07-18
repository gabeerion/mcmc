#! /usr/bin/python

import tt, sys, impute, random, math
import numpy as np
import scratch as s
import imp_mcmc as im
import mifunc as mf
from Bio import AlignIO
from scipy.stats import norm

DELS = .4
IMPUTATIONS = 10
THRESHOLD = .1
ALPHA=.05

def clust(arr):
	p = impute.pdn(arr)
	p[np.diag_indices(p.shape[0])] = sys.maxint
	mins = np.min(p, axis=0)
	return float(np.sum(mins<(THRESHOLD*arr.shape[1])))/p.shape[0]

def impute_xval(c, dfrac=DELS):
#	al = AlignIO.read(path, 'fasta')
#	c = s.arint(np.array(al))

	numdel = int(dfrac*c.shape[0])
#	print numdel

	dels = np.array(random.sample(c, c.shape[0]-numdel))
#	d = s.arint(np.array(dels))

	origclust = clust(c)
	delclust = clust(dels)

	imputations = [impute.impute(dels, numdel) for i in xrange(IMPUTATIONS)]
	impclust = np.array(map(clust, imputations))
	withinvars = impclust*(1-impclust)/imputations[0].shape[0]
	wv = np.mean(withinvars)
	bv = np.var(impclust)
	totalvar = wv+((IMPUTATIONS+1)/IMPUTATIONS)*bv
#	print wv, bv, totalvar
	delvar = delclust*(1-delclust)/c.shape[0]

	z = norm.ppf(1-ALPHA/2)
	conf = z*math.sqrt(totalvar)
	delconf = z*math.sqrt(delvar)

	return (origclust, delclust, delconf, np.mean(impclust), conf)