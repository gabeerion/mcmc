import numpy as np

NUCMAP = {'-':0, 'A':1, 'a':1, 'G':2, 'g':2, 'C':3, 'c':3, 'T':4, 't':4}

#Save an alignment to a fasta file
def qfas(l,fname):
    with open(fname, 'w') as w:
        for r in l:
            w.write('>'+r.locus+'\n')
            w.write(r.sequence+'\n')

#Calculates transition/transversion ratio for an alignment
def ttratio(al):
    ti = 0
    tv = 0
    ppd = {'-': 2, 'A': 1, 'C': 0, 'G': 1, 'T': 0}
    for i in xrange(len(al)-1):
        for j in xrange(i+1, len(al)):
            transitions = sum([1 for t in zip(al[i].seq,al[j].seq) if '-' not in t and t[0]!=t[1] and ppd[t[0]]==ppd[t[1]]])
            transversions = sum([1 for t in zip(al[i].seq,al[j].seq) if '-' not in t and t[0]!=t[1] and ppd[t[0]]!=ppd[t[1]]])
            ti += transitions
            tv += transversions
    return float(ti)/tv

#Calculates transition/transversion ratio for an alignment faster w/numpy
def tt2(al):
	chars = np.array(al)
	pps = (chars == 'A') | (chars == 'G')	#0 for pyrimidine, 1 for purine
	dashes = chars == '-'
	chints = np.array([map(ord, r) for r in chars])
	ti = np.zeros(chars.shape[1])
	tv = np.zeros(chars.shape[1])
	for i in xrange(len(al)-1):
		for j in xrange(i+1, len(al)):
			muts = (chints[i]!=chints[j]).astype('bool')
			transv = (pps[i]!=pps[j]).astype('bool')
			anydash = (dashes[i]|dashes[j]).astype('bool')
			transitions = muts & ~transv & ~anydash
			transversions = muts & transv & ~anydash
			ti += transitions
			tv += transversions
	return float(sum(ti))/sum(tv)

def tt3(chars):
	pps = (chars == 'A') | (chars == 'G')	#0 for pyrimidine, 1 for purine
	dashes = chars == '-'
	chints = np.array([map(ord, r) for r in chars])
	ti = np.zeros(chars.shape[1])
	tv = np.zeros(chars.shape[1])
	for i in xrange(chars.shape[0]-1):
		for j in xrange(i+1, chars.shape[0]):
			muts = (chints[i]!=chints[j]).astype('bool')
			transv = (pps[i]!=pps[j]).astype('bool')
			anydash = (dashes[i]|dashes[j]).astype('bool')
			transitions = muts & ~transv & ~anydash
			transversions = muts & transv & ~anydash
			ti += transitions
			tv += transversions
	return float(sum(ti))/sum(tv)

def ttf(al):
        ti = 0
        tv = 0
        purines = ('A','G')
        pyrimidines = ('C','T')
        chars = np.array(al)
        pps = (chars == purines[0]) | (chars == purines[1])     #0 for pyrimidine, 1 for purine
        dashes = chars == '-'
        for i in xrange(len(al)-1):
                for j in xrange(i+1, len(al)):
                        muts = (chars[i]!=chars[j]).astype('bool')
                        transv = (pps[i]!=pps[j]).astype('bool')
                        anydash = (dashes[i]|dashes[j]).astype('bool')
                        transitions = muts & ~transv & ~anydash
                        transversions = muts & transv & ~anydash
                        #print type(muts)
                        #print type(transv)
                        #print type(anydash)
                        ti += sum(transitions)
                        tv += sum(transversions)
        return float(ti)/tv

"""def tt3(arr):
	ti = 0
	tv = 0
	purines = ('A','G')
	pyrimidines = ('C','T')
	chars = arr
	pps = (chars == purines[0]) | (chars == purines[1])	#0 for pyrimidine, 1 for purine
	dashes = chars == '-'
	for i in xrange(arr.shape[0]-1):
		for j in xrange(i+1, arr.shape[0]):
			muts = (chars[i]!=chars[j]).astype('bool')
			transv = (pps[i]!=pps[j]).astype('bool')
			anydash = (dashes[i]|dashes[j]).astype('bool')
			transitions = muts & ~transv & ~anydash
			transversions = muts & transv & ~anydash
			ti += sum(transitions)
			tv += sum(transversions)
	return float(ti)/tv"""

def arint(arr):
	def f(char): return NUCMAP[char]
	return np.array([map(f,r) for r in arr])


def almat(al):
    return np.array([list(i.seq) for i in al])

def matal(mat):
    return MultipleSeqAlignment([SeqRecord.SeqRecord(Seq.Seq(''.join(r))) for r in mat])

def rand_matal(mat):
    x = mat.shape[1]
    return matal(np.vstack([mat[:,random.randint(0,x-1)] for i in xrange(x)]).transpose())