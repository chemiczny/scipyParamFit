#!/bin/env bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --time=1:00:00
#SBATCH --mem-per-cpu=5GB
##### Nazwa kolejki
#SBATCH -p plgrid-testing

module load plgrid/apps/amber/18
module add plgrid/tools/python/3.6.5

python3 scipyParamFit.py &> fitting.log 
