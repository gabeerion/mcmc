import mcmc as m
import numpy as np

DFILE = 'd_clust.csv'
IMPS = 29

d = np.genfromtxt(DFILE, delimiter=',').astype(np.int)
m.lclass(d,IMPS)