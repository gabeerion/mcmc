#! /usr/bin/python

from Bio import AlignIO, SeqRecord, Seq
from Bio.Align import MultipleSeqAlignment
from multiprocessing import Process, Queue
import numpy as np
import random
from tt import ttratio as tt
from scratch import arint

INFILE = 'hiv-env.fasta'
OUTFILE = 'br_env.csv'
NUMPROCS = 7
REPS = 10003

#Calculates transition/transversion ratio for an alignment faster w/numpy
def tt2(al):
	ti = 0
	tv = 0
	purines = ('A','G')
	pyrimidines = ('C','T')
	chars = np.array(al)
	pps = (chars == purines[0]) | (chars == purines[1])	#0 for pyrimidine, 1 for purine
	dashes = chars == '-'
	for i in xrange(len(al)-1):
		for j in xrange(i+1, len(al)):
			muts = (chars[i]!=chars[j]).astype('bool')
			transv = (pps[i]!=pps[j]).astype('bool')
			anydash = (dashes[i]|dashes[j]).astype('bool')
			transitions = muts & ~transv & ~anydash
			transversions = muts & transv & ~anydash
			ti += sum(transitions)
			tv += sum(transversions)
	return float(ti)/tv

def almat(al):
    return np.array([list(i.seq) for i in al])

def matal(mat):
    return MultipleSeqAlignment([SeqRecord.SeqRecord(Seq.Seq(''.join(r))) for r in mat])

def rand_matal(mat):
    x = mat.shape[1]
    return np.vstack([mat[:,random.randint(0,x-1)] for i in xrange(x)]).transpose()

def boot(al):
	return rand_matal(almat(al))

def fullboot(ar, reps, q):
	for i in xrange(reps):
		q.put(tt(rand_matal(ar)))

def main():
	if REPS%NUMPROCS != 0: print 'Need REPS to be divisible by NUMPROCS'; exit()
	pol = arint(np.array(AlignIO.read(INFILE, 'fasta')))
	Q = Queue()
	procs = []
	data = []
	for i in xrange(NUMPROCS):
		p = Process(target=fullboot, args=(pol,REPS/NUMPROCS,Q))
		procs.append(p)
		p.start()
	for i in xrange(REPS):
		data.append(Q.get())
		print '%d/%d' % (i+1,REPS)
	np.savetxt(OUTFILE,data,delimiter=',')

if __name__ =='__main__': main()
