#!/bin/bash

#SBATCH --partition=compute
#SBATCH --mail-type=NONE
#SBATCH --ntasks=1
#SBATCH --mem=4gb
#SBATCH --time=23:59:00
#SBATCH --job-name trim
#SBATCH --output=log_%j.log

module load cdo

## parse inputs (data filepath and year)
DATA_FP=$1
Y=$2
SM_GRID_FP=${PWD} # save soil moisture grid here

## Filepaths for global ERA-5 data
UVQ_FP="/vortexfs1/share/cmip6/data/era5/reanalysis/pressure-levels/3hr/uvq"
SP_FP="/vortexfs1/share/cmip6/data/era5/reanalysis/single-levels/3hr/sp"
VINT_FP="/vortexfs1/share/cmip6/data/era5/reanalysis/single-levels/3hr/vert-int"
Z_FP="/vortexfs1/share/cmip6/data/era5/reanalysis/pressure-levels/6hr/Z"
SM_FP="/vortexfs1/share/cmip6/data/era5/reanalysis/single-levels/6hr/volumetric_soil_water_layer"

#### Months 1-12
for M in {01..12..01}
do 
    ## uvq_gpllj (GPLLJ region, 3-hourly resolution)
    printf "\nuvq_gpllj (${M}/12)\n"
    cdo -sellonlatbox,257,271,43,25 -sellevel,850,700 $UVQ_FP/uvq-$Y-$M.nc $DATA_FP/uvq_gpllj-$Y-$M.nc
done


#### Months 3-9
for M in {03..09..01}
do
    printf "\nMonth $M/9\n"

    ## vq_cross (GPLLJ region, vertical-cross-section, monthly resolution)
    printf "\nvq_cross\n"
    cdo -sellonlatbox,257,271,30,30 -selvar,"v","q" -monmean $UVQ_FP/uvq-${Y}-$M.nc $DATA_FP/vq_cross-$Y-$M.nc
    
    ## uvq_wide (N. American region, monthly resolution)
    printf "\nuvq_wide\n"
    cdo -sellonlatbox,210,330,60,5 -sellevel,850 -monmean $UVQ_FP/uvq-$Y-$M.nc $DATA_FP/uvq_wide-$Y-$M.nc 
    
    ## IVT (N. Ameican region, daily resolution)
    printf "\nivt\n"
    cdo -selvar,"p71.162","p72.162" -daymean $VINT_FP/vert-int_$Y-$M.nc $DATA_FP/ivt-$Y-$M.nc
    
    ## Z_850, Z_200 (Pacific/N. American region, daily resolution)
    printf "\nZ_850, Z_200\n"
    cdo -sellonlatbox,150,330,60,5 -sellevel,850,200 -daymean $Z_FP/Z_$Y-$M.nc $DATA_FP/Z-$Y-$M.nc

    ## Z_500 (northern hemisphere, for LWA)
    printf "\nZ_500\n"
    cdo -sellonlatbox,0,360,89,0 -seltimestep,2/124/4 -sellevel,500 $Z_FP/Z_$Y-$M.nc $DATA_FP/Z_500-$Y-$M.nc 

    ### sm (Soil moisture)
    printf "\nsm\n"
    cdo -L -daymean -remapbil,${SM_GRID_FP}/sm_grid.txt ${SM_FP}_1/$Y-${M}_volumetric_soil_water_layer_1.nc $DATA_FP/sm-$Y-$M.nc 

    ##### uq_rockies (zonal flux across Rocky mountains)
    printf "\nuq_rockies\n"
    for LAT in 27 32 37
        do
            cdo -daymean -selvar,u,q -sellonlatbox,245,260,$LAT,$LAT $UVQ_FP/uvq-$Y-$M.nc $DATA_FP/uq_rockies-${Y}-${M}_${LAT}.nc
        done
    cdo collgrid $DATA_FP/uq_rockies-${Y}-${M}_*.nc $DATA_FP/uq_rockies-${Y}-${M}.nc
    rm $DATA_FP/uq_rockies-${Y}-${M}_*.nc

done

# merge data into single file for that year
printf "\nmerging...\n"
for varname in uvq_gpllj vq_cross uvq_wide ivt Z Z_500 sm uq_rockies
    do
        cdo -b F64 mergetime $DATA_FP/$varname-$Y-*.nc $DATA_FP/$varname-$Y.nc
        rm $DATA_FP/$varname-$Y-*.nc
    done
printf "Done."

