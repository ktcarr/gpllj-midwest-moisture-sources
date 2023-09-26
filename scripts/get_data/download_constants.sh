#!/bin/bash

#SBATCH --partition=compute
#SBATCH --mail-type=NONE
#SBATCH --ntasks=1
#SBATCH --mem=4gb
#SBATCH --time=23:59:00
#SBATCH --job-name get_constants
#SBATCH --output=log_%j.log

## download navy topography data
wget https://iridl.ldeo.columbia.edu/SOURCES/.WORLDBATH/bath/data.nc -O ${DATA_FP}/constants/topo.nc

## Download era5 data
python download_era5_constants.py



