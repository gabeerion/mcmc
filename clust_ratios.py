import mcmc as m, tests as t
import numpy as np
import impute, tt
bwg = np.array([[c for c in s.seq] for s in m.AlignIO.read('pol-global.fasta', 'fasta') if 'BW' in s.id])
import scratch as s
bwg = s.arint(bwg)
import random
subs = [np.array(random.sample(bwg,40)) for i in xrange(100)]
subclusts = map(t.clust,subs)
d = subs[np.where(subclusts==np.median(subclusts))[0][0]]
m.lclass(d,29)