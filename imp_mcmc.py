#! /usr/bin/python

# imputenew.py -- imputes new phylogenetic sequences in a tree

#-------------------------
#PARAMETERS
#-------------------------
# Default filenames for input/output
DEMOFILE = 'Final_371_Short.csv'
ALIGNFILE = 'mochudi.fasta'
OUTFILE = 'test2.fasta'

# Miscellaneous settinsg
VERBOSE = True
OUTTYPES = 'f'			# This flag is a bit vestigial; if it's 'f', a file will be saved to the specified OUTFILE.

# Thresholds for "old" males and females
AGEF = 35
AGEM = 35

# Number of deletions (for testing) and imputations; these are often equal
TESTDELS = 75
IMPUTATIONS = 75

# Default demographics and weights for demographics
DEMOGRAPHICS = ('OM', 'OF', 'YM', 'YF')
DEMWEIGHTS = {'OM':1, 'OF':10, 'YM':1, 'YF':10}

#-------------------------
#IMPORTED PACKAGES
#-------------------------
import copy
import random
import sys
import itertools

import numpy as np

# import Biopython packages for sequence alignments, reading/writing, and individual sequences
from Bio import Align, AlignIO, Seq, SeqRecord
from Bio.Align import AlignInfo
from Bio.SeqRecord import SeqRecord

#-------------------------
#PREDEFINED FUNCTIONS
#-------------------------

# Deletes a given number of sequences from a given alignment
#def delete(al, dels):
#	weights = {i: DEMWEIGHTS[seq.annotations['dem']] for i,seq in enumerate(al)}
#	remseq = wsnr(weights, dels)
#	return Align.MultipleSeqAlignment([seq for i, seq in enumerate(al) if i not in remseq])
#
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
	return SeqRecord(Seq.Seq(''.join(s)))

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

# Processes and imputes a given number of new AA sequences
# Note -- do we want to add the new pairwise distance after it's imputed? I'm guessing not, since this probably takes us farther from the original distribution
def multimpute(startal, mindict, avgdict, mutweights, pssm, imputations, demweights):
	al = copy.deepcopy(startal)
	total = imputations+len(al)
	demalign = {d:[seq for seq in al if seq.annotations['dem']==d] for d in DEMOGRAPHICS}		# Dictionary of sequences by demographic
	for i in xrange(imputations):
		dem = weightselect(demweights)
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
#		if verbose: print 'Imputed new sequence %d/%d -- %s' % (i+1, imputations, new.id)
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

# A more efficient pairwise distance algorithm (not actually better)
def pd2(al):
    seqlen, allen = len(al[0]), len(al)
    a = np.array([map(ord, i.seq) for i in al])
    a1 = a.reshape(allen,1,seqlen)
    a2 = a.reshape(1,allen,seqlen)
    x = a1==a2
    return len(al[0])-np.sum(x,axis=2)

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
def imp_align(imputations, al, demweights):
	pd = pdn(al)
	sorted_dists = np.array([np.sort(i) for i in pd])		#Returns a sorted pairwise distance matrix
	minlist, avglist, maxlist = orderstats(pd, al, 1), orderstats(pd, al, len(al)/2), orderstats(pd, al, -1)		# Using 1 as index for min to skip the leading 0 in each row
	mindict, avgdict, maxdict = demfilt(minlist), demfilt(avglist), demfilt(maxlist)
	pssm = AlignInfo.SummaryInfo(al).pos_specific_score_matrix()
	mutweights = pssmweight(pssm)
	imputed = multimpute(al, mindict, avgdict, mutweights, pssm, imputations, demweights)
	return imputed

if __name__ == '__main__':
	main()
