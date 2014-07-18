#! /usr/bin/python

# imputenew.py -- imputes new phylogenetic sequences in a tree

#-------------------------
#PARAMETERS
#-------------------------
# Default filenames for input/output
DEMOFILE = 'Final_371_Short.csv'
ALIGNFILE = 'mochudi.fasta'

# Miscellaneous settings
VERBOSE = True

# Thresholds for "old" males and females
AGEF = 35
AGEM = 35

ALPHA = 0.05

# Clustering threshold
THRESHOLD = 0.1
THRESHOLDS = [0.1]

# Number of deletions (for testing) and sequences to impute; these are often equal
TESTDELS = 75
IMPUTATIONS = 75

# Number of imputations to include in multiple imputation
MULTIPLES = 10

# Default demographics and weights for demographics
DEMOGRAPHICS = ('OM', 'OF', 'YM', 'YF')
DEMWEIGHTS = {'OM':1, 'OF':10, 'YM':1, 'YF':10}

BOOT = 10

#-------------------------
#IMPORTED PACKAGES
#-------------------------
import copy
import random
import sys
import itertools
import math
import pdb

import numpy as np
from scipy.stats import norm, sem

# import Biopython packages for sequence alignments, reading/writing, and individual sequences
from Bio import Align, AlignIO, Seq, SeqRecord
from Bio.Align import AlignInfo, MultipleSeqAlignment
from Bio.SeqRecord import SeqRecord
from Bio.Alphabet import SingleLetterAlphabet

#-------------------------
#PREDEFINED FUNCTIONS
#-------------------------

def almat(al):
	return np.array([list(i.seq) for i in al])

# Wraps "clustering" function a bit more nicely
def alclust(al, threshold=THRESHOLD):
	pd = pdn(al)
	sl = len(al[0])
	mns = [sorted(i)[1] for i in pd]
	return clustering(sl,mns,threshold)

def arclust(al,thresholds=THRESHOLDS):
	pd = pdn(al)
	sl = len(al[0])
	mns = [sorted(i)[1] for i in pd]
	return np.array([clustering(sl,mns,threshold) for threshold in thresholds])

# Returns percent clustering for a given threshold
def clustering(seqlen, mins, threshold):
	mins = np.array(mins)
	tlen = threshold*seqlen
	clusters = sum(mins<tlen)
	return float(clusters)/len(mins)

# Calculates confidence intervals for a sample from a Binomial distribution using Wilson score interval
def confint(p,n,z=1.96):
	p = float(p)
	n = float(n)
	z = float(z)
	top = (1/(1+(1/n)*z**2))*(p+(1/(2*n))*z**2+z*math.sqrt((1/n)*p*(1-p)+(1/(4*n**2))*z**2))
	bottom = (1/(1+(1/n)*z**2))*(p+(1/(2*n))*z**2-z*math.sqrt((1/n)*p*(1-p)+(1/(4*n**2))*z**2))
	return (bottom, top)

# Deletes a given number of sequences from a given alignment
def delete(al, dels):
	weights = {i: DEMWEIGHTS[seq.annotations['dem']] for i,seq in enumerate(al)}		# Could do this more efficiently with a list, would need a new weightselect function
	remseq = wsnr(weights, dels)
	return Align.MultipleSeqAlignment([seq for i, seq in enumerate(al) if i not in remseq])

# Categorizes as OF,YF,OM,YM
def demcat(sex, age):
	if sex == 'F' and age >= AGEF: return 'OF'
	elif sex == 'F' and age < AGEF: return 'YF'
	elif sex == 'M' and age >= AGEM: return 'OM'
	elif sex == 'M' and age < AGEM: return 'YM'

# Creates a dictionary of pairwise distances for a given demographic pair
def demfilt(dl):
	distdict = {''.join(i):[] for i in itertools.product(DEMOGRAPHICS, repeat=2)}		# Creates a dictionary with entries for each pairing of demographics
#	distdict['OFOM'], distdict['YMOF'], distdict['YMOM'], distdict['YFYM'], distdict['YFOF'], distdict['YFOM'] = distdict['OMOF'], distdict['OFYM'], distdict['OMYM'], distdict['YMYF'], distdict['OFYF'], distdict['OMYF']
	for t in dl:
		distdict[t[1]+t[2]].append(t[0])
	return distdict

# Upweights female demographics, used for deletion or imputation
#def demweight(dem):
#	if dem == 'YF' or dem == 'OF': return 10
#	if dem == 'YM' or dem == 'OM': return 1

# Imputes a new sequence from a reference sequence and distance
def impute(dem, partner, mutweights, dist, pssm):
	s = list(partner.seq)
	sites = wsnr(mutweights, dist)
	for site in sites:
		d = copy.copy(pssm[site])
		d.pop(s[site])
		s[site] = weightselect(d)
	return SeqRecord(Seq.Seq(''.join(s), SingleLetterAlphabet()))

# Loads assigned sequences; returns an alignment file annotated with demographic data, and a pairwise distance matrix
def load(demfile, alfile):
	with open(demfile, 'r') as f:
		demo = [l.split('\t') for l in f.read().split('\r')]		# Splits the tab-separated input csv file
	demdict = {i[0]:i[1:] for i in demo}							# Makes a dictionary of demographic characteristics, keyed by sequence id
	al = AlignIO.read(alfile, 'fasta')
	for seq in al:
		sex, age = demdict[seq.id][0], int(demdict[seq.id][1])
		seq.annotations={'dem':demcat(sex, age)}
	return al

def matal(mat):
	return MultipleSeqAlignment([SeqRecord(Seq.Seq(''.join(r)), annotations={'dem':weightselect(DEMWEIGHTS)}) for r in mat])

# Processes and imputes a given number of new AA sequences
# Note -- do we want to add the new pairwise distance after it's imputed? I'm guessing not, since this probably takes us farther from the original distribution
def multimpute(al, mindict, avgdict, mutweights, pssm, imputations, verbose):
	if verbose: print 'Imputing new alignment:'
	al = copy.deepcopy(al)
	total = imputations+len(al)
#	oldpd = copy.deepcopy(al.pd)
	demalign = {d:[seq for seq in al if seq.annotations['dem']==d] for d in DEMOGRAPHICS}		# Dictionary of sequences by demographic
	for i in xrange(imputations):
		dem = weightselect(DEMWEIGHTS)
		pweights = {pd: len(mindict[dem+pd]) for pd in DEMOGRAPHICS}		# Creates a list of partner demographics weighted by the number of partners in each demographic
		pdem = weightselect(pweights)
		partner = random.choice(demalign[pdem])
		distdict = mindict if probdist(al, total) else avgdict
		dist = int(random.choice(distdict[dem+pdem]))
		new = impute(dem, partner, mutweights, dist, pssm)
		name = 'I%s-%d' % (partner.id[1:], i)
#		new.id, new.name = ['I'+partner.id[2:]+'-'+str(i)]*2
		new.id, new.name = [name]*2
		new.annotations['dem'] = dem
		demalign[dem].append(new)
		al.append(new)
		if verbose: print '\tImputed new sequence %d/%d -- %s' % (i+1, imputations, new.id)
#		print dist, sum([i!=j for i,j in zip(new.seq,partner.seq)]), len(al)
	return al
#		print new
#		print partner

# Returns a list of minimum, average, or maximum pairwise distances, with associated demographics
# -1 indexing is used to count backwards starting with the maximum
# Right now, we'll just use the nth order statistic; using the actual median requires getting a bit more sophisticated
def orderstats(pd, al, order):
	retlist = []
	for i,dl in enumerate(pd):
		s = sorted(dl)[order]
		dem = al[i].annotations['dem']
		ind = np.where(dl==s)[0]
		pti = random.choice(ind)
#		pti = ind[0]
		ptd = al[pti].annotations['dem']
		retlist.append((s,dem,ptd))
	return retlist

# Calculates pairwise distances between all sequences in an alignment
def pdn(al):
	mat = np.array([map(ord, i.seq) for i in al])
	return np.array([[np.sum(i!=j) for i in mat] for j in mat])

def pdd(al):
	allen = len(al)
	pdd = np.zeros((allen, allen))
	pdd[:allen,:allen] = al.opd

# Returns True if we can impute from a minimum distance, else False
# Set to always return True; commented code is our probabilistic distance choice implementation
def probdist(al, total):
#	p = float(len(al))/total
#	return random.random()<p
	return True

# Creates a dictionary with values proportional to the probability of mutation at each site
def pssmweight(pssm):
	maxweight = sum(pssm[0].values())
	mutweights = {i:int(maxweight-max(x.values())) for i,x in enumerate(pssm)}
	return mutweights

def rand_matal(mat):
	x = mat.shape[1]
	return matal(np.vstack([mat[:,random.randint(0,x-1)] for i in xrange(x)]).transpose())

# Bootstraps with an arbitrary function
def rand_pssm(al, func):
	seqlen = len(al[0])
	pssm = AlignInfo.SummaryInfo(al).pos_specific_score_matrix()
	boot = (MultipleSeqAlignment([SeqRecord(Seq.Seq(''.join([weightselect(pssm[k]) for k in xrange(seqlen)])), annotations={'dem':weightselect(DEMWEIGHTS)}) for _ in xrange(len(al))]) for i in xrange(BOOT))
	return [func(b) for b in boot]

# Does one bootstrap a la shao and stiller
def ssboot(args):
	al, imps = args[0], args[1]
	seqlen = len(al[0])
#	ap = AlignInfo.SummaryInfo(al).pos_specific_score_matrix()
#	fullboot = MultipleSeqAlignment([random.choice(al) for i in xrange(len(al))])
	fullboot = rand_matal(almat(al))
#	fullboot = MultipleSeqAlignment([SeqRecord(Seq.Seq(''.join([weightselect(ap[k]) for k in xrange(seqlen)])), annotations={'dem':weightselect(DEMWEIGHTS)}) for _ in xrange(len(al))])
	subs = delete(fullboot, imps)
	pd = pdn(subs)
	minlist, avglist, maxlist = orderstats(pd, subs, 1), orderstats(pd, subs, len(subs)/2), orderstats(pd, subs, -1)		# Using 1 as index for min to skip the leading 0 in each row
	mindict, avgdict, maxdict = demfilt(minlist), demfilt(avglist), demfilt(maxlist)
	pssm = AlignInfo.SummaryInfo(subs).pos_specific_score_matrix()
	mutweights = pssmweight(pssm)
	boots = [multimpute(subs, mindict, avgdict, mutweights, pssm, imps, False) for i in xrange(MULTIPLES)]
#	pdb.set_trace()
	return np.mean(map(alclust, boots))

# Does one bootstrap on the original data a la shao and stiller
def subboot(al):
	return alclust(rand_matal(almat(al)))

def samplewr(pop,k):
    return [random.choice(pop) for i in xrange(k)]

# Weighted selection of keys from a dictionary where values are weights
def weightselect(d):
    weights = sum([d[i] for i in d])
    t = random.random()*weights
    for i in d:
        t = t - d[i]
        if t <=0: return i

def write(al, types, outfile):
	if 'f' in types:
		AlignIO.write(al, outfile, 'fasta')

# Make n weighted selections without replacement
def wsnr(d, n):
    d = copy.copy(d)
    chosen = []
    for _ in xrange(n):
        r = weightselect(d)
        d.pop(r)
        chosen.append(r)
    return chosen

#-------------------------
#MAIN CODE
#-------------------------
# Main function to do the imputation and output the finished alignment
def main(al, imputations, thresh, verbose):
#	if len(sys.argv) > 1: argv = sys.argv[1:]
	try: pd = al.pd
	except AttributeError:
		pd = pdn(al)
		al.pd = pd
	al.opd = copy.copy(al.pd)
	origclust = alclust(al, thresh)
	sorted_dists = np.array([np.sort(i) for i in pd])		#Returns a sorted pairwise distance matrix
	minlist, avglist, maxlist = orderstats(pd, al, 1), orderstats(pd, al, len(al)/2), orderstats(pd, al, -1)		# Using 1 as index for min to skip the leading 0 in each row
	mindict, avgdict, maxdict = demfilt(minlist), demfilt(avglist), demfilt(maxlist)
#	print len(minlist), sum([len(v) for v in mindict.values()]), len(mindict['OFOF'])+len(mindict['OFYF'])+len(mindict['OFOM'])+len(mindict['OFYM'])
#	print len([i for i in al if i.annotations['dem']=='OF'])
#	pdb.set_trace()
	pssm = AlignInfo.SummaryInfo(al).pos_specific_score_matrix()
	mutweights = pssmweight(pssm)
	if verbose: print 'Imputing...'
	seqlen = len(al[0])
	allen = len(al)
	implen = len(al)+imputations
	alignments = [multimpute(al, mindict, avgdict, mutweights, pssm, imputations, verbose) for i in xrange(MULTIPLES)]
	print [len(a) for a in alignments]
	imputed = np.array([clustering(seqlen, [sorted(j)[1] for j in pdn(a)], thresh) for a in alignments])

	avclust = np.mean(imputed)
	withinvars = imputed*(1-imputed)/len(alignments[0])
	wv = np.mean(withinvars)
#	wv = 0
	bv = np.var(imputed)
	totalvar = wv+((MULTIPLES+1.)/MULTIPLES)*bv
#	print wv, bv, totalvar
	stderr = math.sqrt(totalvar)
#	stderr = sem(imputed)
	z = norm.ppf(1-ALPHA/2)
	conf = z*stderr
#	clustmin = max(avclust-conf,0)
#	clustmax = min(avclust+conf,1)
#	pdb.set_trace()

#	subvar = np.var([subboot(al) for _ in xrange(BOOT)])
	subvar = origclust*(1-origclust)/len(al)
	std2 = math.sqrt(subvar)
	conf2 = z*std2

#	if verbose: print 'Final results:\nClustering estimate: %.2f \n95 percent Confidence Interval: %.2f-%.2f' % (avclust, clustmin, clustmax)
	return avclust, conf, conf2

if __name__ == '__main__':
	main()
