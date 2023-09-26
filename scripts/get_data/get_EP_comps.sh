#!/bin/bash

#SBATCH --partition=compute
#SBATCH --mail-type=NONE
#SBATCH --ntasks=1
#SBATCH --mem=64gb
#SBATCH --time=23:59:00
#SBATCH --job-name EP_comps
#SBATCH --output=log_%j.log

python get_EP_comps.py



