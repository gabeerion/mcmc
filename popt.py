#! /usr/bin/python
# hill-climbing to find best distribution of datasets

import tt, impute, time, multiprocessing, random, sys, math, pdb, csv
import numpy as np
import scratch as s
import mcmc as m
import hc_opt as h
import imp_mcmc as im
import mifunc as mf
from Bio import AlignIO
from collections import defaultdict
from scipy.stats import norm, lognorm, beta, expon, poisson, binom
from scipy.stats import kstest, gaussian_kde as gk
from scipy.misc import comb, logsumexp

from IPython.core import ultratb
sys.excepthook = ultratb.FormattedTB(mode='Verbose',
color_scheme='Linux', call_pdb=1)

ALIGNFILE = 'm14sub.csv'
TFILE = 'popt_target.csv'
CCFILE = 'popt_cc.csv'
CLUSTFILE = 'popt_clust.csv'
KSFILE = 'popt_ks.csv'
IMPS = 100
TBOOT = 1000
RESTARTS = 4
SIZE = 100
STEPS = 10000
MMEAN = 0.01
MSTD = 0.001
SPROB = 0.05
MQS = 1000
THRESHOLD = 0.1

def main(al=np.genfromtxt(ALIGNFILE, delimiter=',').astype(int), imps=IMPS):
	print 'starting...'
	# Basic info about dataset
	allen = al.shape[0]
	implen = allen+imps
	seqlen = al.shape[1]
	delclust = m.clust(al)
	switches = int(SPROB*SIZE)
	if switches < 1: print 'Warning: Edit probability too low; no changes will be made.'

	# Mutation information
	changes = norm(MMEAN*seqlen, MSTD*seqlen)
	pssm = impute.pssm(al).astype(float)/allen
	mutprobs = (1.-np.max(pssm, axis=0)).astype(np.float)
	mutprobs /= np.sum(mutprobs)

	# Target distribution
	tdist = h.build_target(al, TBOOT, TFILE)
	print 'done building target'

	# Parallel opt
	Q, procs, data = multiprocessing.Queue(maxsize=MQS), [], []
	numprocs = multiprocessing.cpu_count()
	reps = -(-RESTARTS/numprocs)
	best = (0,0,0)
	for j in xrange(reps):
		print 'repetition %d' % j
		for i in xrange(numprocs):
			print '\tprocess %d' % i
			p = multiprocessing.Process(target=h.opt, args=(al,imps,tdist), kwargs={'size':SIZE,'steps':STEPS,'sprob':SPROB,'seed':random.randint(0,sys.maxint),'mpqueue':Q})
			procs.append(p)
			p.start()
		for i in xrange(numprocs):
			result = Q.get()
			data.append(result[2])
			if result[2]>best[2]: best = result
			print '\t got process %d result' % i
#	print best
#	print len(data)
#	print [i[2] for i in data]
#	print data[0][0].shape, data[0][1].shape, data[0][2]
	np.savetxt(KSFILE,data)
	np.savetxt(CCFILE,best[0])
	np.savetxt(CLUSTFILE,best[1])

if __name__=='__main__': main()