#! /bin/bash
#SBATCH -n 1
#SBATCH -p general
#SBATCH --mem-per-cpu=1000
#SBATCH -J pubsplit
#SBATCH -o pubsplit.out
#SBATCH --mail-type=END

python pubsplit.py
