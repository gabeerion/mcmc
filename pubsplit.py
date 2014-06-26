#! /usr/bin/python

import os
from Bio import GenBank
from collections import Counter

INFILE = 'hiv-db.genbank'
OUTDIR = 'fastapubs'

#Save an alignment to a fasta file
def qfas(l,fname):
    with open(fname, 'w') as w:
        for r in l:
            w.write('>'+r.locus+'\n')
            w.write(r.sequence+'\n')

def main():
	try: os.mkdir(OUTDIR)
	except OSError: print 'Using existing directory %s' % OUTDIR
	with open(INFILE) as handle:
		records = [r for r in GenBank.parse(handle)]
	c = Counter([r.references[0].pubmed_id for r in records])
	del c['']
	pubs = c.keys()
	seqdict = {k:[r for r in records if r.references[0].pubmed_id == k] for k in pubs}
	for pub in pubs:
		qfas(seqdict[pub], OUTDIR+'/'+pub+'.fasta')


if __name__ == '__main__': main()
