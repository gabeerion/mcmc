#! /usr/bin/python

import re
import os
import sys
import math
import commands
import random
import pdb
import tempfile
import imp_mcmc as impute
import numpy as np
import itertools
from copy import copy, deepcopy
from operator import mul
from collections import Counter
from scipy.spatial.distance import euclidean
from imp_mcmc import pd2 as pdn
from randalign import rlmp as randliks, rlmark
from scipy.stats import norm, beta
from scipy.stats import ks_2samp as ks
from termcolor import colored
from Bio import AlignIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.Align import MultipleSeqAlignment
from Bio.Align.AlignInfo import SummaryInfo
#from matplotlib import pyplot as plt

DEMOFILE = 'Final_371_Short.csv'
ALIGNFILE = 'mochudi.fasta'

TEMPFILE = 'mcmc_temp'
NIN_CMD = './ninja'
PHY_CMD = 'phyml'
PHYAPPENDS = ('_phyml_tree.txt', '_phyml_stats.txt')
TRANSITIONS = 'dayhoff.csv'
MARGINAL = 'dayhoff_marginal.csv'

ML_TEXT = 'phyml -i %s -d aa -b 0 -m Dayhoff -f m -a .431 -v 0.0 -u %s -o n --no_memory_check'
ML_REGEX = r'Log likelihood of the current tree: (-\d+\.\d+)'

AAS = ['-', 'A', 'C', 'E', 'D', 'G', 'F', 'I', 'H', 'K', 'M', 'L', 'N', 'Q', 'P', 'S', 'R', 'T', 'W', 'V', 'Y', 'X']
AAD = ['A','R','N','D','C','Q','E','G','H','I','L','K','M','F','P','S','T','W','Y','V']
THRESHOLD = 0.22

TFUNC = beta(10,1).pdf

CLEANUP = True

def cleanup(tf):
	if CLEANUP:
		for suffix in ('.phylip', '.fasta', '.newick'):
			os.remove(tf+suffix)
		for suffix in PHYAPPENDS:
			os.remove(tf+'.phylip'+suffix)

def clustering(seqlen, mins, threshold):
	mins = np.array(mins)
	tlen = threshold*seqlen
	clusters = sum(mins<tlen)
	return float(clusters)/len(mins)

def clustlik(alignment, num_imp, ref, threshold, reps):
	pd = pdn(alignment)
	al_len = len(alignment)
	means = []
	for i in xrange(reps):
		r = random.sample(range(al_len), al_len-num_imp)
		boot = pd[r][:,r]		# Resample directly from pairwise distance matrix, instead of from alignment
		mins = np.array([sorted(j)[1] for j in boot])
		means.append(clustering(len(alignment[0]), mins, threshold))
	nloc, nscale = norm.fit(means)
	mmin, mmax = min(means), max(means)
	xr = np.linspace(mmin,mmax)
	plt.hist(means, normed=1, alpha=0.5)
	plt.plot(xr, norm(loc=nloc, scale=nscale).pdf(xr))
	plt.show()
	print nloc, nscale
	return norm(loc=nloc, scale=nscale).pdf(ref)

def distlik(alignment, num_imp, mmeans, reps):
	pd = pdn(alignment)
	means = []
	stds = []
#	boot = MultipleSeqAlignment(random.sample(alignment, len(alignment)-num_imp))
	for i in xrange(reps):
		r = random.sample(range(len(alignment)), len(alignment)-num_imp)
		boot = pd[r][:,r]
		mins = np.array([sorted(i)[1] for i in boot])
		means.append(np.mean(mins))
		stds.append(np.std(mins))
	nloc, nscale = norm.fit(means)
#	mmin, mmax = min(means), max(means)
#	xr = np.linspace(mmin,mmax)
#	plt.hist(means, normed=1, alpha=0.5)
#	plt.plot(xr, norm(loc=nloc, scale=nscale).pdf(xr))
#	plt.show()
	return sum(map(math.log, map(norm(loc=nloc, scale=nscale).pdf, mmeans)))

def dist_cdf(mins, origmins):
	mins, origmins = list(mins), list(origmins)
	r = range(min(mins+origmins), max(mins+origmins)+1)
	orig_ec = Counter(r)
	prop_ec = Counter(r)
	for i in r: orig_ec[i]=0; prop_ec[i]=0
	orig_ec.update(origmins)
	prop_ec.update(mins)
	ov = float(sum(orig_ec.values()))
	pv = float(sum(prop_ec.values()))
	for i in r: orig_ec[i]/=ov; prop_ec[i]/=pv
	ocdf, pcdf = [], []
	for i in r:
		try: ocdf.append(ocdf[-1]+orig_ec[i])
		except IndexError: ocdf.append(orig_ec[i])
		try: pcdf.append(pcdf[-1]+prop_ec[i])
		except IndexError: pcdf.append(prop_ec[i])
	return sum(abs(np.array(ocdf)-np.array(pcdf)))


def dist_ks(pd, num_imp, origmins, reps):
#	pd = pdn(alignment)
#	boot = MultipleSeqAlignment(random.sample(alignment, len(alignment)-num_imp))
	allmins = []
	for i in xrange(reps):
		r = random.sample(range(pd.shape[0]), pd.shape[0]-num_imp)
		boot = pd[r][:,r]
		mins = np.array([sorted(j)[1] for j in boot])
		allmins.extend(mins)
#	plt.hist(allmins, alpha=0.5, normed=True)
#	plt.hist(origmins, alpha=0.5, normed=True)
#	print ks(origmins, allmins)
#	plt.show()
	return math.log(ks(origmins, allmins)[1])
	
def dist_norm(alignment, num_imp, origmins, reps):
	pd = pdn(alignment)
	loglik = 0
	for i in xrange(reps):
		r = random.sample(range(len(alignment)), len(alignment)-num_imp)
		boot = pd[r][:,r]
		mins = np.array([sorted(j)[1] for j in boot])
		loglik += math.log(1./(1+euclidean(sorted(origmins), sorted(mins))))
	return loglik

def full_dist_ks(pd, num_imp, origpd, reps):
#	pd = pdn(alignment)
	allen = pd.shape[0]
	orig_flat = origpd.flatten()
#	boot = MultipleSeqAlignment(random.sample(alignment, len(alignment)-num_imp))
	boots = []
	for i in xrange(reps):
		r = random.sample(range(allen), allen-num_imp)
		boot = pd[r][:,r]
		boot_flat = boot.flatten()
		boots.extend(boot_flat)
	orig_flat = sorted(orig_flat)[allen:]
	boots = sorted(boots)[allen*reps:]
#	plt.hist(orig_flat, alpha=0.5, normed=True)
#	plt.hist(boots, alpha=0.5, normed=True)
#	print ks(orig_flat, boots)
#	plt.show()
#	print sorted(orig_flat)
#	print sorted(boots)
	return math.log(ks(orig_flat, boots)[1])
	
def full_dist_norm(pd, num_imp, origpd, reps):
#	pd = pdn(alignment)
	orig_flat = origpd.flatten()
	loglik = 0
	for i in xrange(reps):
		r = random.sample(range(pd.shape[0]), pd.shape[0]-num_imp)
		boot = pd[r][:,r]
		flat_sub = boot.flatten()
		loglik += math.log(1./(1+euclidean(sorted(orig_flat[pd.shape[0]:]), sorted(flat_sub[pd.shape[0]:]))))
	return loglik
	
def loglik(alignment):
#	tf = TEMPFILE+'_%d' % random.randint(0,100000)
	tf = tempfile.mktemp()
	assert(os.path.exists(NIN_CMD))
	alw = rename(alignment)
	AlignIO.write(alw, tf+'.phylip', 'phylip')
	AlignIO.write(alw, tf+'.fasta', 'fasta')
	ninout = commands.getoutput(NIN_CMD+' %s > %s.newick' % (tf+'.fasta',tf))
	phy_command = ML_TEXT % (tf+'.phylip', tf+'.newick')
	phyout = commands.getoutput(phy_command)
	try: lik = float(re.search(ML_REGEX, phyout).group(1))
	except AttributeError: print phyout, phy_command; exit()
	cleanup(tf)
	return lik

def minoneimp(alignment, num_imp):
	origlist = [i for i in alignment[:num_imp]]
	implist = [i for i in alignment[num_imp:]]
	implist.pop(random.randint(0,len(implist)-1))
	return MultipleSeqAlignment(origlist+implist)

def printmins(list, threshold):
    list = sorted(list)
    lless = [i for i in list if i <= threshold]
    lmore = [i for i in list if i > threshold]
    for i in lless: print colored('%.3d'%i, 'green'),
    for i in lmore: print colored('%.3d'%i, 'red'),
    print

def propose(alignment, num_imp, num_changes, transitions):
	num_changes = int(num_changes)
	record = 0
	origlist = [i for i in alignment[:num_imp]]
	implist = [i for i in alignment[num_imp:]]
	targets = [(random.randint(0,len(implist)-1), random.randint(0, len(implist[0])-1)) for i in xrange(num_changes)]
	for i, seq in enumerate(implist):
		sl = list(seq.seq)
		for j, c in enumerate(sl):
			if (i,j) in targets:
				sl[j] = weightselect(transitions[c])
				record += 1
		implist[i] = SeqRecord(Seq(''.join(sl)), id=seq.id, name=seq.name, description=seq.description, annotations=seq.annotations)
	print '%d AA changes introduced' % record
	return MultipleSeqAlignment(origlist+implist)

def propmat(alignment, num_imp, num_changes, transitions, probs):
	num_changes = int(num_changes)
	orlen = len(alignment)-num_imp
	record = 0
	newpd = copy(alignment.pd)
	newdistarray = copy(alignment.distarray)
	origlist = [i for i in alignment[:orlen]]
	implist = [i for i in alignment[orlen:]]
#	targets = [(random.randint(0,len(implist)-1), random.randint(0,len(alignment[0])-1)) for i in xrange(num_changes)]
	targets = [(random.randint(0,len(implist)-1), wl_one(probs)) for i in xrange(num_changes)]
	for t in targets:
		old = newdistarray[orlen+t[0],t[1]]
		new = weightselect(transitions[old])
#		new = random.choice(AAS)
		newdistarray[orlen+t[0],t[1]] = new
		changes = (newdistarray[:,t[1]]==old).astype(int)-(newdistarray[:,t[1]]==new).astype(int)
#		pdb.set_trace()
		newpd[orlen+t[0]]+=changes
		newpd[:,orlen+t[0]]+=changes
		record += 1
#	newpd = np.tril(newpd,-1)
#	newpd += newpd.transpose()
	np.fill_diagonal(newpd,0)
	inds = Counter([t[0] for t in targets]).keys()
	for ind in inds:
		seq = implist[ind]
		implist[ind] = SeqRecord(Seq(''.join(newdistarray[ind+orlen])), id=seq.id, name=seq.name, description=seq.description, annotations=seq.annotations)
	newalign = MultipleSeqAlignment(origlist+implist)
	newalign.pd, newalign.distarray = newpd, newdistarray
	return record, newalign, targets

def propweight(alignment, num_imp, num_changes, transitions, probs):
	num_changes = int(num_changes)
	orlen = len(alignment)-num_imp
	record = 0
	origlist = [i for i in alignment[:orlen]]
	implist = [i for i in alignment[orlen:]]
	targets = [(random.randint(0,len(implist)-1), wl_one(probs)) for i in xrange(num_changes)]
	for i, seq in enumerate(implist):
		sl = list(seq.seq)
		for j, c in enumerate(sl):
			if (i,j) in targets:
				sl[j] = weightselect(transitions[c])
#				sl[j] = random.choice(AAS)
				record += 1
		implist[i] = SeqRecord(Seq(''.join(sl)), id=seq.id, name=seq.name, description=seq.description, annotations=seq.annotations)
#	print '%d AA changes introduced' % record
	return record, MultipleSeqAlignment(origlist+implist), targets

def rename(alignment):
	cl = deepcopy(alignment)
	name_ints = range(len(cl))
	random.shuffle(name_ints)
	name_strs = map(str, name_ints)
	for i in xrange(len(cl)):
		cl[i].name = name_strs[i]
		cl[i].id = name_strs[i]
	return cl

def transprobs(trans_file, marg_file):
	t = np.genfromtxt(trans_file, delimiter=',')
	m = np.genfromtxt(marg_file, delimiter=',')
	d = {a: {a2:t[i,j] for j, a2 in enumerate(AAD)} for i, a in enumerate(AAD)}
	d['-'] = {a: m[i] for i, a in enumerate(AAD)}
	d['X'] = {a: m[i] for i, a in enumerate(AAD)}
	return d

# Weighted selection of keys from a dictionary where values are weights
def weightselect(d):
    weights = sum([d[i] for i in d])
    t = random.random()*weights
    for i in d:
        t = t - d[i]
        if t <=0: return i

# Weighted selection from a list, returns index
# Just for kicks, hackily assuming weights sum to 1
def wl(a):
	weights = sum(a)
	t = random.random()*weights
	for i, x in enumerate(a):
		t -= x
		if t <= 0: return i
		
# Just for kicks, hackily assuming weights sum to 1
def wl_one(a):
	weights = 1
	t = random.random()*weights
	for i, x in enumerate(a):
		t -= x
		if t <= 0: return i

def biolikplot(alignment, num_imp, dem_ratios, length, threshold):
	acceptances = 0
	seq_len = len(alignment[0])
	al_len = len(alignment)
	clusters, logliks = [], []
	d = transprobs(TRANSITIONS, MARGINAL)
	
	#Get statistics for input alignment
	pd = pdn(alignment)
	mins = np.array(sorted([sorted(i)[1] for i in pd]))
	clusters.append(clustering(seq_len, mins, threshold))
	logliks.append(loglik(alignment))
	print 'Original alignment (len %dx%d) has clustering %.2f and LLH %2f' % (len(alignment), len(alignment[0]), clusters[-1], logliks[-1])
	
	#Delete some sequences so we can re-impute for xval
	alignment = MultipleSeqAlignment(random.sample(alignment,len(alignment)-num_imp))
	#Get statistics for "deletions" alignment
	pd = pdn(alignment)
	mins = np.array(sorted([sorted(i)[1] for i in pd]))
	clusters.append(clustering(seq_len, mins, threshold))
	logliks.append(loglik(alignment))
	print 'Deleted alignment (len %dx%d) has clustering %.2f and LLH %2f' % (len(alignment), len(alignment[0]), clusters[-1], logliks[-1]) 

	pssm = SummaryInfo(alignment).pos_specific_score_matrix()
	probs = 1-np.array([max(pssm[i].values()) for i in xrange(seq_len)])/al_len	#Weight site selection by empirical probability of mutation at that site
	probs /= sum(probs)
	
	# Build first state of Markov chain
	print 'Imputing first alignment...'
	current = impute.imp_align(num_imp, alignment, dem_ratios)
	current.loglik = loglik(current)
	current.distarray = np.array([list(s.seq) for s in current])
	current.pd = pdn(current)
	curmins = np.array(sorted([sorted(i)[1] for i in current.pd]))
	clusters.append(clustering(seq_len, curmins, threshold))
	logliks.append(current.loglik)
	print '\t Log likelihood %2f' % current.loglik
#	if not burnin: AlignIO.write(current, '%s/%d.fasta' % (directory,0), 'fasta')
	# Run chain
	for i in xrange(1,length):
		proposal = propmat(current,num_imp,max(norm(loc=2,scale=1).rvs(),1), d, probs)[1]
		proposal.loglik = loglik(proposal)
		for m,n in itertools.product(range(proposal.pd.shape[0]), range(proposal.pd.shape[1])):
			if (proposal.pd[m][n] < 10) and m!=n: proposal.loglik = -sys.maxint-1; print m,n, proposal.pd[m][n]
		p = proposal.loglik-current.loglik
		print 'Current LLH: %2f; Proposed LLH: %2f; Acceptance probability %e' % (current.loglik, proposal.loglik, math.exp(p))
		if random.random()<math.exp(p):
			current = proposal
			acceptances += 1
			print '\tAccepted'
		else: print '\tNot accepted'
		curmins = np.array(sorted([sorted(i)[1] for i in current.pd]))
		clusters.append(clustering(seq_len, curmins, threshold))
		logliks.append(current.loglik)
#		if i > burnin:
#			AlignIO.write(current, '%s/%d.fasta' % (directory,i-burnin), 'fasta')
	r=random.randint(0,1000000)
	print r
	AlignIO.write(current, '%d.fasta'%r, 'fasta')
	return np.vstack((logliks,clusters))


def mcmc(alignment, num_imp, dem_ratios, directory, length, burnin):
	acceptances = 0
	# Build first state of Markov chain
	print 'Imputing first alignment...'
	current = impute.imp_align(num_imp, alignment, dem_ratios)
	current.loglik = loglik(current)
	print '\t Log likelihood %2f' % current.loglik
	if not burnin: AlignIO.write(current, '%s/%d.fasta' % (directory,0), 'fasta')
	# Run chain
	for i in xrange(1,length+1):
		proposal = impute.imp_align(1, minoneimp(current, num_imp), dem_ratios)
		proposal.loglik = loglik(proposal)
		p = proposal.loglik-current.loglik
		print 'Current LLH: %2f; Proposed LLH: %2f' % (current.loglik, proposal.loglik)
		print '\tAcceptance probability %e' % math.exp(p)
		if p>0:
			current = proposal
			acceptances += 1
			print '\tAccepted'
		elif random.random()<math.exp(p):
			current = proposal
			acceptances += 1
			print '\tAccepted'
		else: print '\tNot accepted'
		if i > burnin:
			AlignIO.write(current, '%s/%d.fasta' % (directory,i-burnin), 'fasta')
	return float(acceptances)/length

def mcmc_simple(alignment, num_imp, dem_ratios, directory, length, burnin):
	acceptances = 0
	d = transprobs(TRANSITIONS, MARGINAL)
	# Build first state of Markov chain
	print 'Imputing first alignment...'
	current = impute.imp_align(num_imp, alignment, dem_ratios)
	current.loglik = loglik(current)
	print '\t Log likelihood %2f' % current.loglik
	if not burnin: AlignIO.write(current, '%s/%d.fasta' % (directory,0), 'fasta')
	# Run chain
	for i in xrange(1,length+1):
		proposal = propose(current,num_imp,max(norm(loc=2,scale=1).rvs(),1), d)
		proposal.loglik = loglik(proposal)
		p = proposal.loglik-current.loglik
		print 'Current LLH: %2f; Proposed LLH: %2f' % (current.loglik, proposal.loglik)
		print '\tAcceptance probability %e' % math.exp(p)
		if random.random()<math.exp(p):
			current = proposal
			acceptances += 1
			print '\tAccepted'
		else: print '\tNot accepted'
		if i > burnin:
			AlignIO.write(current, '%s/%d.fasta' % (directory,i-burnin), 'fasta')
	return float(acceptances)/length

def mcmc_sym_dist(alignment, num_imp, dem_ratios, directory, length, burnin):
	acceptances = 0
	d = transprobs(TRANSITIONS, MARGINAL)
	pd = pdn(alignment)
	mins = np.array([sorted(i) for i in pd])
	nloc, nscale = norm.fit(mins)
	dist = norm(nloc, nscale)
	# Build first state of Markov chain
	print 'Imputing first alignment...'
	current = impute.imp_align(num_imp, alignment, dem_ratios)
	current.loglik = loglik(current)+math.log(distlik(current, num_imp, nloc, 1000))
	print '\t Log likelihood %2f' % current.loglik
	if not burnin: AlignIO.write(current, '%s/%d.fasta' % (directory,0), 'fasta')
	# Run chain
	for i in xrange(1,length+1):
		proposal = propose(current,num_imp,max(norm(loc=2,scale=1).rvs(),1), d)
		l1 = loglik(proposal)
		l2 = math.log(distlik(proposal, num_imp, nloc, 1000))
		proposal.loglik = l1+l2
		p = proposal.loglik-current.loglik
		print 'Current LLH: %2f; Proposed LLH: %2f' % (current.loglik, proposal.loglik)
		print '\tPhylogeny component: %2f; Distance component: %2f' % (l1, l2)
		print '\tAcceptance probability %e' % math.exp(p)
		if random.random()<math.exp(p):
			current = proposal
			acceptances += 1
			print '\tAccepted'
		else: print '\tNot accepted'
		if i > burnin:
			AlignIO.write(current, '%s/%d.fasta' % (directory,i-burnin), 'fasta')
	return float(acceptances)/length

def mcmc_ks(alignment, num_imp, dem_ratios, directory, length, burnin):
	acceptances = 0
	d = transprobs(TRANSITIONS, MARGINAL)
	pd = pdn(alignment)
	mins = np.array([sorted(i)[1] for i in pd])
	# Build first state of Markov chain
	print 'Imputing first alignment...'
	start = impute.imp_align(num_imp, alignment, dem_ratios)
	current = deepcopy(start)
	current.loglik = loglik(current)+math.log(dist_ks(current, num_imp, mins, 1000))
	print '\t Log likelihood %2f' % current.loglik
	if not burnin: AlignIO.write(current, '%s/%d.fasta' % (directory,0), 'fasta')
	# Run chain
	for i in xrange(1,length+1):
		proposal = propose(current,num_imp,max(norm(loc=2,scale=1).rvs(),1), d)
		l1 = loglik(proposal)
		l2 = math.log(dist_ks(proposal, num_imp, mins, 1000))
		proposal.loglik = l1+l2
		p = proposal.loglik-current.loglik
		print 'Current LLH: %2f; Proposed LLH: %2f' % (current.loglik, proposal.loglik)
		print '\tPhylogeny component: %2f; Distance component: %2f' % (l1, l2)
		print '\tAcceptance probability %e' % math.exp(p)
		if random.random()<math.exp(p):
			current = proposal
			acceptances += 1
			print '\tAccepted'
		else: print '\tNot accepted'
		if i > burnin:
			AlignIO.write(current, '%s/%d.fasta' % (directory,i-burnin), 'fasta')
	return float(acceptances)/length, start

#Also incorporates PSSM to weight by site probabilities
def mcmc_clust_ks(alignment, num_imp, dem_ratios, directory, length, burnin, threshold, refpd):
	print num_imp
	refmins = np.array([sorted(i)[1] for i in refpd])
	# Builds PSSM and list of AA frequencies by site: only necessary if we're initializing with random sequences
	int_thresh = int(threshold*len(alignment[0]))
	pssm = SummaryInfo(alignment).pos_specific_score_matrix()
	siteaas = [[k for k in pssm[i].keys() if pssm[i][k]] for i in xrange(len(alignment[0]))]
	al_len = len(alignment)
	seq_len = len(alignment[0])
	probs = 1-np.array([max(pssm[i].values()) for i in xrange(len(alignment[0]))])/al_len	#Weight site selection by empirical probability of mutation at that site
	acceptances = 0
	transitions = transprobs(TRANSITIONS, MARGINAL)	# Build transition probabilities for each site
	pd = pdn(alignment)
	# Statistics for the resampled alignment with "missingness"
	mins = np.array([sorted(i)[1] for i in pd])
	print 'Minimum distances after deletion:'
	printmins(mins, int_thresh)
	minmean = np.mean(mins)
	init_clust = clustering(len(alignment[0]), mins, threshold)
	print 'Initial clustering: %.2f' % init_clust
	likelihoods = []
	# Build first state of Markov chain, by imputing, randomly copying sequences, or generating totally random sequences from empirical AA probabilities
	print 'Imputing first alignment...'
#	start = impute.imp_align(num_imp, alignment, dem_ratios)
	start = MultipleSeqAlignment(list(alignment) + random.sample(alignment, num_imp))
#	start = MultipleSeqAlignment(list(alignment) + [SeqRecord(Seq(''.join([weightselect(pssm[k]) for k in xrange(len(alignment[0]))]))) for _ in xrange(num_imp)])
	current = deepcopy(start)
	current.pd = pdn(current)
#	current.loglik = loglik(current)+math.log(dist_ks(current, num_imp, mins, 1000))+math.log(clustlik(current, num_imp, init_clust, threshold, 1000))	#Contains vestigial cluster likelihood
	fmins = sorted([sorted(j)[1] for j in current.pd])
	current.loglik = math.log(1./(1+euclidean(fmins, refmins)))
	printmins(fmins, int_thresh)
	print '\t Log likelihood %2f' % current.loglik
	if not burnin: AlignIO.write(current, '%s/%d.fasta' % (directory,0), 'fasta')
	print 'Iter\t#AA\tCurrent LLH\tProposed LLH\tDistance Cmpt\tAcceptance Prob'
	targets = []
	clusters = []
	clusters.append(clustering(len(current[0]), np.array([sorted(m)[1] for m in current.pd]), threshold))
	likelihoods.append(current.loglik)
	print clusters[0]
	print euclidean(sorted(current.pd.flatten())[al_len:], sorted(refpd.flatten())[al_len:]), ks(refmins, fmins)
	# Run chain
	for i in xrange(1,length+1):
		likelihoods.append(current.loglik)
		changes, proposal, tapp = propweight(current,num_imp,max(norm(loc=num_imp*2,scale=num_imp/2).rvs(),1), transitions, probs)
		targets.extend(tapp)
		proposal.pd = pdn(proposal)
		fmins = [sorted(n)[1] for n in proposal.pd]
#		printmins(fmins, int_thresh)
#		l1 = loglik(proposal)
#		l2 = math.log(clustlik(proposal, num_imp, init_clust, threshold, 1000))
		l2 = math.log(1./(1+euclidean(fmins, refmins)))
#		l3 = math.log(clustlik(proposal, num_imp, init_clust, threshold, 1000))	#Again, we're not using cluster likelihood anymore
		proposal.loglik = l2
#		print proposal.loglik, len(tapp), sum([str(proposal[i].seq)!=str(current[i].seq) for i in xrange(len(proposal))])
		p = math.exp(proposal.loglik-current.loglik)
	#	if eraselast: print '\r%d\t%d\t%2f\t%2f\t%2f\t%e' % (i,changes, current.loglik, proposal.loglik, l2, p),
#		else: print '\n%d\t%d\t%2f\t%2f\t%2f\t%e' % (i,changes, current.loglik, proposal.loglik, l2, p),
		if 1<float(p):
			current = proposal
			acceptances += 1
			print '%d\t%d\t%2f\t%2f\t%2f\t%e' % (i,changes, current.loglik, proposal.loglik, l2, p)
			fmins = sorted([sorted(n)[1] for n in current.pd])
#			print ''
			printmins(refmins, int_thresh)
			printmins(fmins, int_thresh)
			print clusters[-1], euclidean(sorted(current.pd.flatten())[al_len:], sorted(refpd.flatten())[al_len:]), ks(refmins, fmins)
		else: pass
		clusters.append(clustering(len(current[0]), np.array([sorted(m)[1] for m in current.pd]), threshold))
#		if i > burnin: pass
#			AlignIO.write(current, '%s/%d.fasta' % (directory,i-burnin), 'fasta')
	rn = random.randint(0,1000000)
	AlignIO.write(current, '%s/%d.fasta'%(directory,rn), 'fasta')
#	tx = [i[0] for i in targets]
#	ty = [i[1] for i in targets]
#	h = np.histogram2d(tx,ty,bins=(num_imp,549))
#	plt.imshow(h[0])
#	plt.plot(np.array(likelihoods)/np.mean(likelihoods))
#	plt.plot(np.array(clusters)/np.mean(clusters))
#	plt.show()
#	plt.plot(likelihoods); plt.show()
#	plt.plot(clusters); plt.show()
	clusters = np.array(clusters)
	likelihoods = np.array(likelihoods)
	return float(acceptances)/length, start, likelihoods, clusters, rn

def mcmc_norm(alignment, num_imp, dem_ratios, directory, length, burnin, threshold, refpd):
	refmins = np.array(sorted([sorted(i)[1] for i in refpd]))
	pd = pdn(alignment)
	al_len = len(alignment)
	seq_len = len(alignment[0])
	refclust = clustering(seq_len, refmins, threshold)
	int_thresh = int(threshold*len(alignment[0]))
	acceptances = 0
	# Builds PSSM and list of AA frequencies by site: only necessary if we're initializing with random sequences
	pssm = SummaryInfo(alignment).pos_specific_score_matrix()
	siteaas = [[k for k in pssm[i].keys() if pssm[i][k]] for i in xrange(len(alignment[0]))]
	probs = 1-np.array([max(pssm[i].values()) for i in xrange(len(alignment[0]))])/al_len	#Weight site selection by empirical probability of mutation at that site
	probs /= sum(probs)
	transitions = transprobs(TRANSITIONS, MARGINAL)	# Build transition probabilities for each site
	# Statistics for the resampled alignment with "missingness"
	mins = np.array([sorted(i)[1] for i in pd])
	minmean = np.mean(mins)
	init_clust = clustering(len(alignment[0]), mins, threshold)
	print 'Initial clustering: %.2f' % init_clust
	likelihoods = []
	# Build first state of Markov chain, by imputing, randomly copying sequences, or generating totally random sequences from empirical AA probabilities
	print 'Imputing first alignment...'
	start = impute.imp_align(num_imp, alignment, dem_ratios)
	current = deepcopy(start)
	current.pd = pdn(current)
	current.distarray = np.array([list(s.seq) for s in current])
	fmins = sorted([sorted(j)[1] for j in current.pd])
#	current.loglik = math.log(1./(1+dist_cdf(fmins, refmins)))
#	current.loglik = math.log(1./(1+euclidean(fmins, refmins)))
	current.loglik = math.log(ks(fmins,refmins)[1])
#	current.loglik = 1./(1+(refclust-clustering(seq_len,fmins,threshold))**2)
	printmins(fmins, int_thresh)
	print '\t Log likelihood %2f' % current.loglik
	if not burnin: AlignIO.write(current, '%s/%d.fasta' % (directory,0), 'fasta')
	print 'Iter\t#AA\tCurrent LLH\tProposed LLH\tDistance Cmpt\tAcceptance Prob\tClust'
	targets = []
	clusters = []
	propliks = []
	clusters.append(clustering(len(current[0]), np.array([sorted(m)[1] for m in current.pd]), threshold))
	likelihoods.append(current.loglik)
	propliks.append(current.loglik)
	print clusters[0]
	print euclidean(sorted(current.pd.flatten())[al_len:], sorted(refpd.flatten())[al_len:]), ks(refmins, fmins)
	# Run chain
	for i in xrange(1,length+1):
		likelihoods.append(current.loglik)
		changes, proposal, tapp = propmat(current,num_imp,max(norm(loc=num_imp*2,scale=num_imp/2).rvs(),1), transitions, probs)
		targets.extend(tapp)
#		proposal.pd = pdn(proposal)
		fmins = sorted([sorted(n)[1] for n in proposal.pd])
		proposal.clust = clustering(seq_len,fmins,threshold)
#		l2 = math.log(1./(1+dist_cdf(fmins, refmins)))
#		l2 = math.log(1./(1+euclidean(fmins, refmins)))
		l2 = math.log(ks(fmins,refmins)[1])
#		l2 = 1./(1+(refclust-clustering(seq_len,fmins,threshold))**2)
		proposal.loglik = l2
		propliks.append(proposal.loglik)
		p = math.exp(proposal.loglik-current.loglik)
		if 1<float(p):
			current = proposal
			acceptances += 1
			print colored('%d\t%d\t%2f\t%2f\t%2f\t%e\t%.2f' % (i,changes, current.loglik, proposal.loglik, l2, p, proposal.clust), 'blue')
			printmins(fmins, int_thresh)
#			printmins(refmins, int_thresh)
#			printmins(fmins, int_thresh)
#			print clusters[-1], euclidean(sorted(current.pd.flatten())[al_len:], sorted(refpd.flatten())[al_len:]), ks(refmins, fmins)
#		else: print colored('%d\t%d\t%2f\t%2f\t%2f\t%e\t%.2f' % (i,changes, current.loglik, proposal.loglik, l2, p, proposal.clust), 'grey')
		else: pass
		clusters.append(clustering(len(current[0]), np.array([sorted(m)[1] for m in current.pd]), threshold))
	rn = random.randint(0,1000000)
	AlignIO.write(current, '%s/%d.fasta'%(directory,rn), 'fasta')
	np.savetxt('%s/%dprops.csv'%(directory,rn),propliks, 
delimiter=',')
	clusters = np.array(clusters)
	likelihoods = np.array(likelihoods)
	return float(acceptances)/length, start, likelihoods, clusters, rn

def mcmc_corrected(alignment, num_imp, dem_ratios, directory, length, burnin, threshold, refpd):
	refmins = np.array(sorted([sorted(i)[1] for i in refpd]))
	pd = pdn(alignment)
	al_len = len(alignment)
	seq_len = len(alignment[0])
	refclust = clustering(seq_len, refmins, threshold)
	int_thresh = int(threshold*len(alignment[0]))
	acceptances = 0
	# Builds PSSM and list of AA frequencies by site: only necessary if we're initializing with random sequences
	pssm = SummaryInfo(alignment).pos_specific_score_matrix()
	siteaas = [[k for k in pssm[i].keys() if pssm[i][k]] for i in xrange(len(alignment[0]))]
	probs = 1-np.array([max(pssm[i].values()) for i in xrange(len(alignment[0]))])/al_len	#Weight site selection by empirical probability of mutation at that site
	probs /= sum(probs)
	transitions = transprobs(TRANSITIONS, MARGINAL)	# Build transition probabilities for each site
	# Statistics for the resampled alignment with "missingness"
	mins = np.array([sorted(i)[1] for i in pd])
	minmean = np.mean(mins)
	init_clust = clustering(len(alignment[0]), mins, threshold)
	print 'Initial clustering: %.2f' % init_clust
	likelihoods = []
	# Get the likelihood correction function and target function
	print 'Fitting likelihood correction function...'
	corfunc = rlmark(alignment, pssm, transitions, probs, num_imp, mins, 1000)[0]
#	print 'Using gamma with parameters %.4f, %.4f, %.4f' % cordist.args
#	corfunc = cordist.pdf
	target = TFUNC
	# Build first state of Markov chain, by imputing, randomly copying sequences, or generating totally random sequences from empirical AA probabilities
	print 'Imputing first alignment...'
	start = impute.imp_align(num_imp, alignment, dem_ratios)
	current = deepcopy(start)
	current.pd = pdn(current)
	current.distarray = np.array([list(s.seq) for s in current])
	fmins = sorted([sorted(j)[1] for j in current.pd])
	initlik = math.log(ks(fmins,refmins)[1])
	try: current.loglik = math.log(target(math.exp(initlik)))-math.log(math.exp(corfunc(math.exp(initlik))))
	except ValueError: pdb.set_trace()
	current.clust = clustering(seq_len,fmins,threshold)
	printmins(fmins, int_thresh)
	print 'Iter\t#AA\tCurrent LLH\tProposed LLH\t\tRaw LLH\tAcceptance Prob\tClust'
	print '%d\t%d\t%2f\t%2f\t%2f\t%e\t%.2f' % (0, 0, current.loglik, current.loglik, initlik, 1, current.clust)
	if not burnin: AlignIO.write(current, '%s/%d.fasta' % (directory,0), 'fasta')
#	targets = []
	clusters = []
	propliks = []
	clusters.append(clustering(len(current[0]), np.array([sorted(m)[1] for m in current.pd]), threshold))
	likelihoods.append(current.loglik)
	propliks.append(current.loglik)
	# Run chain
	for i in xrange(1,length+1):
		likelihoods.append(current.loglik)
		changes, proposal, tapp = propmat(current,num_imp,max(norm(loc=num_imp,scale=num_imp/2).rvs(),1), transitions, probs)
#		targets.extend(tapp)
		fmins = sorted([sorted(n)[1] for n in proposal.pd])
		proposal.clust = clustering(seq_len,fmins,threshold)
		initlik = math.log(ks(fmins,refmins)[1])
		try: proposal.loglik = math.log(target(math.exp(initlik)))-math.log(math.exp(corfunc(initlik)))
		except ValueError: pdb.set_trace()
		propliks.append(proposal.loglik)
		p = math.exp(proposal.loglik-current.loglik)
		if random.random()<float(p):
			current = proposal
			acceptances += 1
			print colored('%d\t%d\t%2f\t%2f\t%2f,%e\t%.2f' % (i,changes, current.loglik, proposal.loglik, initlik, p, proposal.clust), 'blue')
			printmins(fmins, int_thresh)
		else: print colored('%d\t%d\t%2f\t%2f\t%2f\t%e\t%.2f' % (i,changes, current.loglik, proposal.loglik, initlik, p, proposal.clust), 'grey')
		clusters.append(current.clust)
	rn = random.randint(0,1000000)
	AlignIO.write(current, '%s/%d.fasta'%(directory,rn), 'fasta')
	np.savetxt('%s/%dprops.csv'%(directory,rn),propliks, delimiter=',')
	return float(acceptances)/length, start, np.array(likelihoods), np.array(clusters), rn

def mcmc_corrected(alignment, num_imp, dem_ratios, directory, length, burnin, threshold, refpd):
	refmins = np.array(sorted([sorted(i)[1] for i in refpd]))
	pd = pdn(alignment)
	al_len = len(alignment)
	seq_len = len(alignment[0])
	refclust = clustering(seq_len, refmins, threshold)
	int_thresh = int(threshold*len(alignment[0]))
	acceptances = 0
	# Builds PSSM and list of AA frequencies by site: only necessary if we're initializing with random sequences
	pssm = SummaryInfo(alignment).pos_specific_score_matrix()
	siteaas = [[k for k in pssm[i].keys() if pssm[i][k]] for i in xrange(len(alignment[0]))]
	probs = 1-np.array([max(pssm[i].values()) for i in xrange(len(alignment[0]))])/al_len	#Weight site selection by empirical probability of mutation at that site
	probs /= sum(probs)
	transitions = transprobs(TRANSITIONS, MARGINAL)	# Build transition probabilities for each site
	# Statistics for the resampled alignment with "missingness"
	mins = np.array([sorted(i)[1] for i in pd])
	minmean = np.mean(mins)
	init_clust = clustering(len(alignment[0]), mins, threshold)
	print 'Initial clustering: %.2f' % init_clust
	likelihoods = []
	# Get the likelihood correction function and target function
	print 'Fitting likelihood correction function...'
	corfunc = rlmark(alignment, pssm, transitions, probs, num_imp, mins, 1000)[0]
	target = TFUNC
	# Build first state of Markov chain, by imputing, randomly copying sequences, or generating totally random sequences from empirical AA probabilities
	print 'Imputing first alignment...'
	start = impute.imp_align(num_imp, alignment, dem_ratios)
	current = deepcopy(start)
	current.pd = pdn(current)
	current.distarray = np.array([list(s.seq) for s in current])
	fmins = sorted([sorted(j)[1] for j in current.pd])
	initlik = math.log(ks(fmins,refmins)[1])
	try: current.loglik = math.log(target(math.exp(initlik)))-math.log(math.exp(corfunc(math.exp(initlik))))
	except ValueError: pdb.set_trace()
	current.clust = clustering(seq_len,fmins,threshold)
	printmins(fmins, int_thresh)
	print 'Iter\t#AA\tCurrent LLH\tProposed LLH\t\tRaw LLH\tAcceptance Prob\tClust'
	print '%d\t%d\t%2f\t%2f\t%2f\t%e\t%.2f' % (0, 0, current.loglik, current.loglik, initlik, 1, current.clust)
	if not burnin: AlignIO.write(current, '%s/%d.fasta' % (directory,0), 'fasta')
	clusters = []
	propliks = []
	clusters.append(clustering(len(current[0]), np.array([sorted(m)[1] for m in current.pd]), threshold))
	likelihoods.append(current.loglik)
	propliks.append(current.loglik)
	# Run chain
	for i in xrange(1,length+1):
		likelihoods.append(current.loglik)
		changes, proposal, tapp = propmat(current,num_imp,max(norm(loc=num_imp,scale=num_imp/2).rvs(),1), transitions, probs)
		fmins = sorted([sorted(n)[1] for n in proposal.pd])
		proposal.clust = clustering(seq_len,fmins,threshold)
		initlik = math.log(ks(fmins,refmins)[1])
		try: proposal.loglik = math.log(target(math.exp(initlik)))-math.log(math.exp(corfunc(initlik)))
		except ValueError: pdb.set_trace()
		propliks.append(proposal.loglik)
		p = math.exp(proposal.loglik-current.loglik)
		if random.random()<float(p):
			current = proposal
			acceptances += 1
			print colored('%d\t%d\t%2f\t%2f\t%2f,%e\t%.2f' % (i,changes, current.loglik, proposal.loglik, initlik, p, proposal.clust), 'blue')
			printmins(fmins, int_thresh)
		else: print colored('%d\t%d\t%2f\t%2f\t%2f\t%e\t%.2f' % (i,changes, current.loglik, proposal.loglik, initlik, p, proposal.clust), 'grey')
		clusters.append(current.clust)
	rn = random.randint(0,1000000)
	AlignIO.write(current, '%s/%d.fasta'%(directory,rn), 'fasta')
	np.savetxt('%s/%dprops.csv'%(directory,rn),propliks, delimiter=',')
	return float(acceptances)/length, start, np.array(likelihoods), np.array(clusters), rn

def main():
	al = impute.load(DEMOFILE, ALIGNFILE)
	
