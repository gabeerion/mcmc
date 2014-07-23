import mcmc as m, tests as t, numpy as np
import random, scratch
from Bio import AlignIO

REPS = 100
DELS = 29
OUTFILE = 'multdels.csv'

t.THRESHOLD=0.01
bwg = np.array([[c for c in s.seq] for s in m.AlignIO.read('pol-global.fasta', 'fasta') if 'BW' in s.id])
bwg = scratch.arint(bwg)
subs = [np.array(random.sample(bwg,bwg.shape[0]-DELS)) for i in xrange(REPS)]
origclust = t.clust(bwg)
liks, cerr = [], []
for s in subs:
	l,c = t.lkcor(s,DELS,origclust)
	liks.extend(l)
	cerr.extend(c)
np.savetxt(OUTFILE,np.vstack((liks,cerr)),delimiter=',')
