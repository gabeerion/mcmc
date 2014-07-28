#! /usr/bin/python

import tt, impute, time, multiprocessing, random, sys, math
import numpy as np
import scratch as s
import imp_mcmc as im
import mifunc as mf
from Bio import AlignIO
from scipy.stats import norm
from scipy.stats import ks_2samp as ks, gaussian_kde as gk

CCLASS_REPS = 300
STEPS = 100
IMPS = 29
BOOTREPS = 100
THRESHOLD = 0.01
OUT_RATIOS = 'mcmc_ratios_mp.csv'
OUT_STATES = 'mcmc_states_clust.csv'
ALIGNFILE = 'bwg_del.csv'
RDIST = 'bwgtt.csv'
ORDERFUNC = np.min
LC_DIST = 'mcmc_ratios_clust.csv'
LC_STATES = 'mcmc_states_clust.csv'
MP_DIST = 'mcmc_ratios_mp.csv'
MP_STATES = 'mcmc_states_mp.csv'
TTMP_DIST = 'mcmc_ratios_ttmp.csv'
TTMP_STATES = 'mcmc_states_ttmp.csv'
RAND_OUT = 'randout.csv'


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

def c((al, alclust, imps)):
	allen = al.shape[0]
	seqlen = al.shape[1]
	b = impute.impute(al,imps,orderfunc=ORDERFUNC)
	return clik((b,alclust,allen))

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

def clik((al,origclust,dellen)):
	allen = al.shape[0]
	stats = []
	for i in xrange(BOOTREPS):
		boot = al[np.random.choice(xrange(allen),dellen,replace=0)]
		stats.append(clust(boot))
	return norm(*norm.fit(stats)).pdf(origclust)

def clik1((al,origclust,dellen)):
	allen = al.shape[0]
	stats = []
	for i in xrange(BOOTREPS):
		boot = np.array(random.sample(al,dellen))
		stats.append(clust(boot))
	return norm(*norm.fit(stats)).pdf(origclust)

def clik2((al,origclust,dellen)):
	allen = al.shape[0]
	stats = []
	for i in xrange(BOOTREPS):
		boot = np.array(random.sample(al,dellen))
		stats.append(clust(boot))
	return 1/abs(np.mean(stats)-origclust)


def cbootlik((al, dellen)):
	return clust(al[np.random.choice(xrange(al.shape[0]),dellen,replace=0)])
P = multiprocessing.Pool(processes=multiprocessing.cpu_count())
def mlik((al,origclust,dellen)):
	allen = al.shape[0]
	dat = ((al,dellen) for i in xrange(BOOTREPS))
	stats = P.map(cbootlik, dat)
	return norm(*norm.fit(stats)).pdf(origclust)
def cbl2(al, dellen, reps, Q):
	for i in xrange(reps):
		Q.put(clust(al[np.random.choice(xrange(al.shape[0]),dellen,replace=0)]))
def mlike2((al,origclust,dellen)):
	allen = al.shape[0]
	Q = multiprocessing.Queue()
	numprocs = multiprocessing.cpu_count()
	reps = int(math.ceil(float(BOOTREPS)/numprocs)*numprocs)
	procs = []
	data = []
	for i in xrange(numprocs):
		p = multiprocessing.Process(target = cbl2, args=(al,dellen,reps,Q))
		procs.append(p)
		p.start()
	for i in xrange(reps):
		data.append(Q.get())
	return norm(*norm.fit(data)).pdf(origclust)


def tlik((al,origtt,dellen)):
	allen = al.shape[0]
	stats = []
	for i in xrange(BOOTREPS):
		boot = np.array(random.sample(al,dellen))
		stats.append(tt.ttratio(boot))
	return norm(*norm.fit(stats)).pdf(origtt)

def cmm((al,origclust,dellen)):
	allen = al.shape[0]
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

def mcmc_tt(al=np.genfromtxt(ALIGNFILE,delimiter=',').astype(np.int), imps=IMPS):
	
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

def lclass(al, imps):
	allen = al.shape[0]
	seqlen = al.shape[1]
	delclust = clust(al)
	numprocs = multiprocessing.cpu_count()
	reps = [(al,delclust,imps)]*CCLASS_REPS
	ratios = P.map(c,reps)
	np.savetxt(OUT_RATIOS, ratios, delimiter=',')		# Save ratios?
	return gk(ratios)

def mcmc_clust(al=np.genfromtxt(ALIGNFILE,delimiter=',').astype(np.int), imps=IMPS):
	allen = al.shape[0]
	seqlen = al.shape[1]
	delclust = clust(al)
	
	print 'Building likelihood distributions...'
	try: 
		pdist = gk(np.genfromtxt(LC_DIST, delimiter=','))
	except IOError: 
		print 'Existing distribution not found, building...'
		pdist = lclass(al, imps)

	print 'Starting MCMC:'
	print 'Step#\t|New Lik\t|New PropLik\t|Old Lik\t|Old PropLik\t|Accept Prob'
	old = impute.impute(al,imps, orderfunc=ORDERFUNC)
	old_lik = clik((old,delclust,allen))
	old_plik = pdist(old_lik)

	states = [(clust(old),old_lik,old_plik,old_lik,old_plik,1)]

	for i in xrange(STEPS):
		prop = impute.impute(al,imps, orderfunc=ORDERFUNC)
		prop_lik = clik((prop,delclust,allen))
		prop_plik = pdist(prop_lik)

		a = (prop_lik/old_lik)*(old_plik/prop_plik)
		states.append((clust(old),prop_lik,prop_plik,old_lik,old_plik,a))
		print '%d\t|%2f\t|%2f\t|%2f\t|%2f\t|%e' % (i+1,prop_lik,prop_plik,old_lik,old_plik,a)
		if random.random()<a:
			old, old_lik, old_plik = prop, prop_lik, prop_plik

	states.append((clust(old),prop_lik,prop_plik,old_lik,old_plik,a))
	np.savetxt(LC_STATES, np.array(states), delimiter=',')

#Multithreaded proposal
def genstate(al,imps,reps,Q,seed,pdist=None):
	random.seed(seed)
	impute.np.random.seed(seed)
	allen = al.shape[0]
	delclust = clust(al)
	for i in xrange(reps):
		prop = impute.impute(al,imps)
		prop_lik = clik((prop,delclust,allen))
		prop_clust = clust(prop)
		if pdist: 
			prop_plik = pdist(prop_lik)
			Q.put((prop, prop_lik, prop_plik, prop_clust))
		else: Q.put((prop, prop_lik, prop_clust))

def lclass_mp(al, imps):
	allen = al.shape[0]
	seqlen = al.shape[1]
	delclust = clust(al)
	Q, procs, data = multiprocessing.Queue(), [], []
	numprocs = multiprocessing.cpu_count()
	reps = -(-CCLASS_REPS/numprocs)
	for i in xrange(numprocs):
		p = multiprocessing.Process(target=genstate, args=(al,imps,reps,Q,i))
		procs.append(p)
		p.start()
	old_percent = 0
	for i in xrange(reps*numprocs):
		percent = int(float(i)/(reps*numprocs) * 100)
		if percent > old_percent: 
			print '%d percent' % int(percent)
			old_percent = percent
		prop, prop_lik, prop_clust = Q.get()
		data.append(prop_lik)
	np.savetxt(MP_DIST, data, delimiter=',')		# Save ratios?
	return gk(data)

def mcmc_mp(al=np.genfromtxt(ALIGNFILE,delimiter=',').astype(np.int), imps=IMPS):
	allen = al.shape[0]
	seqlen = al.shape[1]
	delclust = clust(al)
	
	print 'Building likelihood distributions...'
	try: 
		pdist = gk(np.genfromtxt(MP_DIST, delimiter=','))
	except IOError: 
		print 'Existing distribution not found, building...'
		pdist = lclass_mp(al, imps)

	print 'Starting MCMC:'
	print 'Step#\tOld Clust\t|New Lik\t|New PropLik\t|Old Lik\t|Old PropLik\t|Accept Prob'
	old = impute.impute(al,imps, orderfunc=ORDERFUNC)
	old_lik = clik((old,delclust,allen))
	old_plik = pdist(old_lik)
	old_clust = clust(old)

	states = [(old_clust,old_lik,old_plik,old_lik,old_plik,1)]

	Q, procs, data = multiprocessing.Queue(), [], []
	numprocs = multiprocessing.cpu_count()-1
	reps = -(-STEPS/numprocs)
	for i in xrange(numprocs):
		p = multiprocessing.Process(target=genstate, args=(al,imps,reps,Q,i,pdist))
		procs.append(p)
		p.start()
	for i in xrange(reps*numprocs):
		prop, prop_lik, prop_plik, prop_clust = Q.get()
		a = (prop_lik/old_lik)*(old_plik/prop_plik)
		states.append((old_clust,prop_lik,prop_plik,old_lik,old_plik,a))
		print '%d\t|%2f\t|%2f\t|%2f\t|%2f\t|%2f\t|%e' % (i+1,old_clust,prop_lik,prop_plik,old_lik,old_plik,a)
		if random.random()<a:
			old, old_lik, old_plik, old_clust = prop, prop_lik, prop_plik, prop_clust

	states.append((old_clust,prop_lik,prop_plik,old_lik,old_plik,a))
	np.savetxt(MP_STATES, np.array(states), delimiter=',')

#Multithreaded proposal
def gttmp(lik,al,imps,reps,Q,seed,pdist=None):
	random.seed(seed)
	impute.np.random.seed(seed)
	allen = al.shape[0]
	delclust = clust(al)
	for i in xrange(reps):
		prop = impute.impute(al,imps)
		prop_lik = lik(prop)
		prop_clust = clust(prop)
		if pdist: 
			prop_plik = pdist(prop_lik)
			Q.put((prop, prop_lik, prop_plik, prop_clust))
		else: Q.put((prop, prop_lik, prop_clust))
def lclass_ttmp(al, imps, lik):
	allen = al.shape[0]
	seqlen = al.shape[1]
	delclust = clust(al)
	Q, procs, data = multiprocessing.Queue(), [], []
	numprocs = multiprocessing.cpu_count()
	reps = -(-CCLASS_REPS/numprocs)
	for i in xrange(numprocs):
		p = multiprocessing.Process(target=gttmp, args=(lik,al,imps,reps,Q,i))
		procs.append(p)
		p.start()
	old_percent = 0
	for i in xrange(reps*numprocs):
		percent = int(float(i)/(reps*numprocs) * 100)
		if percent > old_percent: 
			print '%d percent' % int(percent)
			old_percent = percent
		prop, prop_lik, prop_clust = Q.get()
		data.append(prop_lik)
	np.savetxt(TTMP_DIST, data, delimiter=',')		# Save ratios?
	return gk(data)

def mcmc_ttmp(al=np.genfromtxt(ALIGNFILE,delimiter=',').astype(np.int), imps=IMPS):
	allen = al.shape[0]
	seqlen = al.shape[1]
	delclust = clust(al)
	
	print 'Building likelihood distributions...'
	ldist = norm(*norm.fit(np.genfromtxt(RDIST, delimiter=',')))
	def lik(al):
		return ldist.pdf(tt.ttratio(al))
	try: 
		pdist = gk(np.genfromtxt(TTMP_DIST, delimiter=','))
	except IOError: 
		print 'Existing distribution not found, building...'
		pdist = lclass_ttmp(al, imps, lik)

	print 'Starting MCMC:'
	print 'Step#\tOld Clust\t|New Lik\t|New PropLik\t|Old Lik\t|Old PropLik\t|Accept Prob'
	old = impute.impute(al,imps, orderfunc=ORDERFUNC)
	old_lik = lik(old)
	old_plik = pdist(old_lik)
	old_clust = clust(old)

	states = [(old_clust,old_lik,old_plik,old_lik,old_plik,1)]

	Q, procs, data = multiprocessing.Queue(), [], []
	numprocs = multiprocessing.cpu_count()-1
	reps = -(-STEPS/numprocs)
	for i in xrange(numprocs):
		p = multiprocessing.Process(target=gttmp, args=(lik,al,imps,reps,Q,i,pdist))
		procs.append(p)
		p.start()
	for i in xrange(reps*numprocs):
		prop, prop_lik, prop_plik, prop_clust = Q.get()
		a = (prop_lik/old_lik)*(old_plik/prop_plik)
		states.append((old_clust,prop_lik,prop_plik,old_lik,old_plik,a))
		print '%d\t|%2f\t|%2f\t|%2f\t|%2f\t|%2f\t|%e' % (i+1,old_clust,prop_lik,prop_plik,old_lik,old_plik,a)
		if random.random()<a:
			old, old_lik, old_plik, old_clust = prop, prop_lik, prop_plik, prop_clust

	states.append((old_clust,prop_lik,prop_plik,old_lik,old_plik,a))
	np.savetxt(TTMP_STATES, np.array(states), delimiter=',')

def unifsamp((allen,seqlen), origclust, dellen):
	def boot((allen,seqlen), origclust, dellen, reps, Q):	
		for i in xrange(reps):
			boot = np.random.random_integers(0,5, size=(allen,seqlen))
			if clust(boot) == 0: Q.put(0.0)
			else: Q.put(clik((boot,origclust,dellen)))
	Q, procs, data = multiprocessing.Queue(), [], []
	numprocs = multiprocessing.cpu_count()-1
	reps = -(-CCLASS_REPS/numprocs)
	for i in xrange(numprocs):
		p = multiprocessing.Process(target = boot, args = ((allen,seqlen), origclust, dellen, reps, Q))
		procs.append(p)
		p.start()
	for i in xrange(reps*numprocs):
		x = Q.get()
		data.append(x)
		print i, x
	np.savetxt(RAND_OUT, data, delimiter=',')


if __name__ == '__main__': mcmc_ttmp()