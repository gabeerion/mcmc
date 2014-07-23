#! /usr/bin/python

import tt, sys, impute, random, math, multiprocessing
import numpy as np
import scratch as s
import imp_mcmc as im
import mifunc as mf
from Bio import AlignIO
from scipy.stats import norm, ks_2samp as ks
from mcmc import distmins, clik1, clik2
from matplotlib import pyplot as plt

DELS = .4
IMPUTATIONS = 10
THRESHOLD = .01
ALPHA=.05
CORREPS = 100

def clust(arr):
	p = impute.pdn(arr)
	p[np.diag_indices(p.shape[0])] = sys.maxint
	mins = np.min(p, axis=0)
	return float(np.sum(mins<(THRESHOLD*arr.shape[1])))/p.shape[0]

def ecdf(a):
	s = np.sort(a)
	yvals = np.arange(len(s))/float(len(s))
	return (s,yvals)

def lkcor(a, dels, truth):
	p = multiprocessing.Pool(processes = multiprocessing.cpu_count()-1)
	delclust = clust(a)
	delmins = distmins(a)
	imps = [impute.impute(a,dels) for i in xrange(CORREPS)]
	clusts = map(clust, imps)
	dat = [(i,delclust,a.shape[0]) for i in imps]
	liks = p.map(clik1, dat)
	cerr = np.abs(np.array(clusts)-truth)
#	plt.scatter(liks,cerr)
#	plt.show()
	return (liks,cerr)

def lkcheat((a, origmins)):
	return ks(distmins(a), origmins)[1]


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
