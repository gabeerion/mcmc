#! /usr/bin/python

import tt, impute, multiprocessing, random, sys
import numpy as np
import scratch as s
import imp_mcmc as im
import mifunc as mf
from Bio import AlignIO
from scipy.stats import norm
from scipy.stats import ks_2samp as ks

CCLASS_REPS = 100000
STEPS = 10000
IMPS = 40
BOOTREPS = 100
THRESHOLD = 0.01
OUT_RATIOS = 'mcmc_ratios_avg.csv'
OUT_STATES = 'mcmc_states_avg.csv'
ALIGNFILE = 'mcmc_test_dels.csv'
RDIST = 'brfast.csv'
ORDERFUNC = np.min

def clust(arr):
	p = impute.pdn(arr)
	p[np.diag_indices(p.shape[0])] = sys.maxint
	mins = np.min(p, axis=0)
	return float(np.sum(mins<(THRESHOLD*arr.shape[1])))/p.shape[0]

def distmins(al):
	p = impute.pdn(al)
	p[np.diag_indices(p.shape[0])] = sys.maxint
	return np.min(p,axis=0)

def r(a):
	b = impute.impute(a[0],a[1],orderfunc=ORDERFUNC)
	return tt.ttratio(b)

def k(a):
	al = a[0]
	reps = a[1]
	origmins = a[2]
	b = impute.impute(al, reps, orderfunc=ORDERFUNC)
	stats = []
	for i in xrange(BOOTREPS):
		boot = np.array(random.sample(b,al.shape[0]-reps))
		stats.append(ks(distmins(boot),origmins))
	return np.mean(np.array(stats),axis=0)[1]

# a should be a tuple
def klik1((al,origmins)):
	allen, dellen = al.shape[0], origmins.shape[0]
	stats = []
	for i in xrange(BOOTREPS):
		boot = np.array(random.sample(al,dellen))
		stats.append(ks(distmins(boot),origmins))
	return np.mean(stats, axis=0)[1]

# a should be a tuple
def klik2((al,origmins)):
	allen, dellen = al.shape[0], origmins.shape[0]
	dm = []
	for i in xrange(BOOTREPS):
		boot = np.array(random.sample(al,dellen))
		dm.extend(distmins(boot))
	return ks(origmins,dm)[1]

def clik1((al,origclust,di)):
	allen, dellen = al.shape[0], di
	stats = []
	for i in xrange(BOOTREPS):
		boot = np.array(random.sample(al,dellen))
		stats.append(clust(boot))
	return norm(*norm.fit(stats)).pdf(origclust)

def clik2((al,origclust,di)):
	allen, dellen = al.shape[0], di
	stats = []
	for i in xrange(BOOTREPS):
		boot = np.array(random.sample(al,dellen))
		stats.append(clust(boot))
	return 1/abs(np.mean(stats)-origclust)

def cmm((al,origclust,di)):
	allen, dellen = al.shape[0], di
	stats = []
	boots = []
	for i in xrange(BOOTREPS):
		boot = np.array(random.sample(al,dellen))
		boots.append(boot)
		stats.append(clust(boot))
	cerr = np.array(stats)-origclust
	amin,amax = np.argmin(cerr), np.argmax(cerr)
	return boots[amin], boots[amax]

# a should be a tuple
def kboot((al,origmins)):
	allen, dellen = al.shape[0], origmins.shape[0]
	print allen, dellen
	stats = []
	for i in xrange(BOOTREPS):
		boot = np.array(random.sample(al,dellen))
		stats.append(boot)
	return np.array(stats)



def cclass(al, imps):
	allen = al.shape[0]
	seqlen = al.shape[1]
	numprocs = multiprocessing.cpu_count()
	p = multiprocessing.Pool(processes=numprocs)
	reps = [(al,imps)]*CCLASS_REPS
	ratios = p.map(r,reps)
	np.savetxt(OUT_RATIOS, ratios, delimiter=',')		# Save ratios?
	return norm(*norm.fit(ratios))

def main(al=np.genfromtxt(ALIGNFILE,delimiter=',').astype(np.int), imps=IMPS):
	
	print 'Building likelihood distributions...'
	rdist = np.genfromtxt(RDIST, delimiter=',')
	ldist = norm(*norm.fit(rdist))
	pdist = cclass(al, imps)

	print 'Starting MCMC:'
	print 'Step#\t|New Lik\t|New PropLik\t|Old Lik\t|Old PropLik\t|Accept Prob'
	old = impute.impute(al,imps, orderfunc=ORDERFUNC)
	old_tt = tt.ttratio(old)
	old_lik = ldist.pdf(old_tt)
	old_plik = pdist.pdf(old_tt)

	states = [(clust(old),old_lik,old_plik,old_lik,old_plik,1)]

	for i in xrange(STEPS):
		prop = impute.impute(al,imps, orderfunc=ORDERFUNC)
		prop_tt = tt.ttratio(prop)
		prop_lik = ldist.pdf(prop_tt)
		prop_plik = pdist.pdf(prop_tt)

		a = (prop_lik/old_lik)*(old_plik/prop_plik)
		states.append((clust(old),prop_lik,prop_plik,old_lik,old_plik,a))
		print '%d\t|%2f\t|%2f\t|%2f\t|%2f\t|%e' % (i+1,prop_lik,prop_plik,old_lik,old_plik,a)
		if random.random()<a:
			old, old_tt, old_lik, old_plik = prop, prop_tt, prop_lik, prop_plik

	states.append((clust(old),prop_lik,prop_plik,old_lik,old_plik,a))
	np.savetxt(OUT_STATES, np.array(states), delimiter=',')



if __name__ == '__main__': main()