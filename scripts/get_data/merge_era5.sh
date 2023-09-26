#!/bin/bash

#SBATCH --partition=compute
#SBATCH --mail-type=NONE
#SBATCH --ntasks=1
#SBATCH --mem=4gb
#SBATCH --time=23:59:00
#SBATCH --job-name merge
#SBATCH --output=log_%j.log

module load cdo

DATA_FP=$1

### merge data into single file for that year
for varname in uvq_gpllj uvq_wide vq_cross uq_rockies Z Z_500 ivt sm
do
    cdo -b F64 mergetime $DATA_FP/$varname-*.nc $DATA_FP/$varname.nc
    rm $DATA_FP/$varname-19*.nc
    rm $DATA_FP/$varname-20*.nc
done

## Get monthly-mean, single-level data over N. America
MONTHLY_FP="/vortexfs1/share/cmip6/data/era5/reanalysis/single-levels/monthly-means"
for varname in surface_pressure total_column_water_vapour
do
    cdo -b F64 mergetime $MONTHLY_FP/${varname}/*_${varname}.nc $DATA_FP/${varname}_temp.nc
    cdo -sellonlatbox,210,330,60,5 $DATA_FP/${varname}_temp.nc $DATA_FP/${varname}.nc
    rm $DATA_FP/${varname}_temp.nc
done

## rename these files
mv $DATA_FP/surface_pressure.nc $DATA_FP/sp.nc
mv $DATA_FP/total_column_water_vapour.nc $DATA_FP/tcwv.nc
