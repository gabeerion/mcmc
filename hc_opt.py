#! /usr/bin/python
# hill-climbing to find best distribution of datasets

import tt, impute, time, multiprocessing, random, sys, math, pdb, csv
import numpy as np
import scratch as s
import mcmc as m
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
TFILE = 'opt_target.csv'
CCFILE = 'opt_cc.csv'
CLUSTFILE = 'opt_clust.csv'
KSFILE = 'opt_ks.csv'
IMPS = 100
TBOOT = 1000
SIZE = 100
STEPS = 10000
MMEAN = 0.01
MSTD = 0.001
SPROB = 0.05
MQS = 1000
THRESHOLD = 0.1


def alp(old, mutprobs, changes, pssm):
	nucs = np.arange(5,dtype=np.int)
	ret = np.copy(old)
	muts = set([(np.random.randint(old.shape[0]),impute.weightselect(mutprobs,random.random())) for i in xrange(changes)])
	while len(muts) != changes:
		add = (np.random.randint(old.shape[0]),impute.weightselect(mutprobs,random.random()))
		muts.update([add])
	rands = np.random.random(size=len(muts))
	for i, (seq, site) in enumerate(muts):
		ret[seq,site] = impute.weightselect(pssm[:,site],rands[i])
	return ret

def exact_boot(pd,dellen,seqlen):
	binary = pd < (THRESHOLD*seqlen)
	binary[np.diag_indices(binary.shape[0])]=0
	pchoice = float(dellen)/pd.shape[0]
	indicators = [pchoice*(1-(1-pchoice)**np.sum(r)) for r in binary]
	return np.sum(indicators)/dellen

def tclass_v(al, outfile, tboot=TBOOT):
	allen = al.shape[0]
	seqlen = al.shape[1]
	delclust = m.clust(al)
	Q, procs, data = multiprocessing.Queue(maxsize=MQS), [], []
	numprocs = multiprocessing.cpu_count()
	reps = -(-tboot/numprocs)
	def bootclust(al,reps,Q,seed):
		np.random.seed(seed)
		for i in xrange(reps):
			Q.put(m.clust(al[:,np.random.choice(xrange(al.shape[1]),al.shape[1],replace=1)]))
	for i in xrange(numprocs):
		p = multiprocessing.Process(target=bootclust, args=(al,reps,Q,random.randint(0,numprocs**2)))
		procs.append(p)
		p.start()
	old_percent = 0
	for i in xrange(reps*numprocs):
		percent = int(float(i)/(reps*numprocs) * 100)
		if percent > old_percent: 
			print '%d percent' % int(percent)
			old_percent = percent
		data.append(Q.get())
	np.savetxt(outfile, data)
	return np.std(data)

def build_target(al, boot, outfile):
	delclust = m.clust(al)
	# Build target distribution by bootstrapping
	tdist = norm(delclust, tclass_v(al, outfile, tboot=boot))
	return tdist

def opt(al, imps, tdist, size=SIZE, steps=STEPS, sprob=SPROB, seed=random.randint(0,sys.maxint), mpqueue=None):
	# Make sure we're random
	random.seed(seed)
	np.random.seed(seed)
	impute.np.random.seed(seed)

	# Basic info about dataset
	allen = al.shape[0]
	implen = allen+imps
	seqlen = al.shape[1]
	delclust = m.clust(al)
	switches = int(sprob*size)

	# Mutation information
	changes = norm(MMEAN*seqlen, MSTD*seqlen)
	pssm = impute.pssm(al).astype(float)/allen
	mutprobs = (1.-np.max(pssm, axis=0)).astype(np.float)
	mutprobs /= np.sum(mutprobs)

	# Impute first distribution of datsets
	current = np.array([impute.impute(al, imps) for i in xrange(size)])
	current_cc = np.array([exact_boot(impute.pdn(c),allen,seqlen) for c in current])
	current_clust = np.array(map(m.clust,current))
	current_ks = kstest(current_cc,tdist.cdf)[1]

	# Optimize
	for i in xrange(steps):
		change_indices = np.random.choice(xrange(size), size=switches, replace=False)
		prop = np.array([alp(a, mutprobs, int(changes.rvs()), pssm) for a in current[change_indices]])
		prop_clust = map(m.clust,prop)
		prop_cc = np.copy(current_cc)
		prop_cc[change_indices] = [exact_boot(impute.pdn(p), allen, seqlen) for p in prop]
		prop_ks = kstest(prop_cc,tdist.cdf)[1]
		diff = prop_ks-current_ks
		if diff>=0:
			current[change_indices] = prop
			current_clust[change_indices] = prop_clust
			current_cc = prop_cc
			current_ks = prop_ks
	if mpqueue: mpqueue.put((current_cc, current_clust, current_ks))
	return (current_cc, current_clust, current_ks)



def main(al=np.genfromtxt(ALIGNFILE, delimiter=',').astype(int), imps=IMPS):
	# Basic info about dataset
	allen = al.shape[0]
	implen = allen+imps
	seqlen = al.shape[1]
	delclust = m.clust(al)
	switches = int(SPROB*SIZE)

	# Mutation information
	changes = norm(MMEAN*seqlen, MSTD*seqlen)
	pssm = impute.pssm(al).astype(float)/allen
	mutprobs = (1.-np.max(pssm, axis=0)).astype(np.float)
	mutprobs /= np.sum(mutprobs)

	# Build target distribution by bootstrapping
	tdist = norm(delclust, tclass_v(al, TFILE))
#	target = np.genfromtxt('popt_target.csv', delimiter=',')
#	tdist = norm(delclust, np.std(target))

	# Impute first distribution of datsets
	current = np.array([impute.impute(al, imps) for i in xrange(SIZE)])
	current_cc = np.array([exact_boot(impute.pdn(c),allen,seqlen) for c in current])
	current_clust = np.array(map(m.clust,current))
	current_ks = kstest(current_cc,tdist.cdf)[1]

	# Optimize
	ccstates = open(CCFILE, 'w')
	cluststates = open(CLUSTFILE, 'w')
	ks_states = open(KSFILE, 'w')
	ccwriter = csv.writer(ccstates)
	clustwriter = csv.writer(cluststates)
	kswriter = csv.writer(ks_states)

	for i in xrange(STEPS):
		print i, current_ks

		change_indices = np.random.choice(xrange(SIZE), size=switches, replace=False)
		prop = np.array([alp(a, mutprobs, int(changes.rvs()), pssm) for a in current[change_indices]])
		prop_clust = map(m.clust,prop)
		prop_cc = np.copy(current_cc)
		prop_cc[change_indices] = [exact_boot(impute.pdn(p), allen, seqlen) for p in prop]
		prop_ks = kstest(prop_cc,tdist.cdf)[1]
		diff = prop_ks-current_ks
		ccwriter.writerow(current_cc)
		clustwriter.writerow(current_clust)
		kswriter.writerow([current_ks, prop_ks])
		if diff>=0:
#			if diff > 0: print i, prop_ks
#			if diff !=0: pdb.set_trace()
			current[change_indices] = prop
			current_clust[change_indices] = prop_clust
			current_cc = prop_cc
			current_ks = prop_ks
		"""if i%(STEPS/100) == 0:
			print current_ks,
			ccwriter.writerow(current_cc)
			clustwriter.writerow(current_clust)
			print '(saved)'
		else: print current_ks"""
	print current_ks
	return (current_cc, current_clust, current_ks)

if __name__=='__main__': main()