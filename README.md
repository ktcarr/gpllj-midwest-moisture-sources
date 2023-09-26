# gpllj-midwest-moisture-sources
This repository contains the code used to produce figures in the paper "Impact of atmospheric circulation variability on U.S. Midwest moisture sources".

## File structure
```
-- README.md 
 -- .gitignore
 -- setup.py
|-- src
     -- __init__.py
     -- utils.py
     -- params.py
|-- envs
     -- env_main.yml
     -- env_windspharm.yml
|-- scripts
     -- run_all.sh
    |-- get_data
         -- *.py
         -- *.sh
    |-- preprocess
         -- *.py
|-- tests
     -- test.py
```

## Instructions

### Set up virtual environment environments
1. Navigate to the project's home directory, ```./gpllj-moisture-tracking```
2. Specify location for saving data and results (directories must already exist). E.g.,
```
DATA_FP=/vortexfs1/scratch/kcarr/gpllj-moisture-tracking_data
SAVE_FP=/vortexfs1/home/kcarr/gpllj-moisture-tracking_clean/gpllj-moisture-tracking/results
```
3. Create create skeleton file structure for saving data and results:
```
./scripts/make_folders.sh ${DATA_FP} ${SAVE_FP}
```
4. Specify location for moisture tracking output data (contact authors for data access). E.g.,
```
WAM_FP=/vortexfs1/share/clidex/data/publications/carr_ummenhofer_2023/moisture-tracking-output
```

5a. Install and activate primary virtual environment with
```
mamba create --prefix ./envs/env_main
conda activate ./envs/env_main
mamba env update --file ./envs/env_main.yml
pip install -e .
mamba env config vars set DATA_FP=${DATA_FP} SAVE_FP=${SAVE_FP} WAM_FP=${WAM_FP}
```

5b. Owing to python package conflicts, a second virtual environment is needed to compute the Helmholtz decomposition (of vertically-integrated water vapour transport). To install this environment, use:
```
mamba create --prefix ./envs./env_windspharm
conda activate ./envs/env_windspharm
mamba env update --file env_windspharm.yml
pip install -e .
mamba env config vars set DATA_FP=${DATA_FP} SAVE_FP=${SAVE_FP} WAM_FP=${WAM_FP}
```

5c. Before proceeding, reactivate the primary environment with ```conda activate ./envs/env_main```.

### Obtain data

#### Output from moisture tracking model (available upon request).
| Variable                                | Filename pattern            | Freq. | Res. | Longitude (ºE) | Latitude (ºN) | Month range |
|:----------------------------------------|:----------------------------|:------|:-----|:---------------|:--------------|:------------|
| precipitation, <br> tracked evaporation | ```fluxes_daily_YYYY.nc```  | daily | 1º   | 0 – 359        | -30 – 80      | 1 – 12      |
| tracked moisture                        | ```storage_daily_YYYY.nc``` | daily | 1º   | 0 – 359        | -30 – 80      | 1 – 12      |

Note: data for each year is saved in different file (year indicated by "YYYY"). Location of directory containing these individual files is specified by ```${WAM_FP}```.

#### ERA5 time-varying fields (1979-2020)

| Variable                                    | Filename            | Freq.    | Res.  | Longitude (ºE) | Latitude (ºN) | P-levels (hPa) | Month range |
|:--------------------------------------------|:--------------------|:---------|:------|:---------------|:--------------|:---------------|:------------|
| $u,v,q$                                     | ```uvq_gpllj.nc```  | 3-hrly | 1º    | 257 – 271      | 25 – 43       | 850, 700       | 1 – 12      |
| $v,q$                                       | ```vq_cross.nc```   | monthly  | 1º    | 257 – 271      | 30            | 1000 – 1       | 4 – 8       |
| $u,q$                                       | ```uq_rockies.nc``` | daily    | 1º    | 245 – 260      | 27, 32, 37    | 1000 – 1       | 4 – 8       |
| $u,v,q$                                     | ```uvq_wide.nc```   | monthly  | 1º    | 210 – 330      | 5 – 60        | 850            | 4 – 8       |
| Geopotential                                | ```Z.nc```          | daily            | 1º    | 150 – 330      | 5 – 60        | 850, 200       | 3 – 9       |
| Geopotential                                | ```Z_500.nc```      | daily @ 6 UTC    | 1º    | 0 – 359        | 0 – 90        | 500            | 4 – 8       |
| Surface pressure                            | ```sp.nc```         | monthly  | 0.25º | 210 – 330      | 5 – 60        | N/A            | 3 – 9       |
| Total column <br> water vapour              | ```tcwv.nc```       | monthly  | 0.25º | 210 – 330      | 5 – 60        | N/A            | 4 – 8       |
| Vertical integral of <br> water vapour flux | ```ivt.nc```        | daily    | 1º    | 0 – 359        | -90 – 90      | N/A            | 3 – 9       |
| Volumetric soil water <br> (layer 1)        | ```sm.nc```         | daily    | 1º    | 235 – 295      | 15 – 55       | N/A            | 3 – 9       |

This data should be saved to the 'raw' data subdirectory: ```${DATA_FP}/raw```.

#### ERA5 constants
| Variable             | Filename              | Res.  | Longitude (ºE) | Latitude (ºN) |
|:---------------------|:----------------------|:------|:---------------|:--------------|
| Land-sea mask        | ```lsm.nc```          | 1º    | 0 – 359        | -30 – 80      |
| Lake cover           | ```lake_cover.nc```   | 1º    | 0 – 359        | -30 – 80      |
| Surface geopotential | ```Z_surf_lores.nc``` | 1º    | 210 – 330      | 5 – 60        |
| Surface geopotential | ```Z_surf_hires.nc``` | 0.25º | 210 – 330      | 5 – 60        |

This data should be saved to the 'constants' data subdirectory: ```${DATA_FP}/constants```.

#### Non-ERA5 data
| Variable             | Filename              | Res.  | Longitude (ºE) | Latitude (ºN) | Month range |
|:---------------------|:----------------------|:------|:---------------|:--------------|:------------|
| PRISM precipitation  | ```prism.nc```        | 1/24º | 235 – 294      | 24 – 50       | 3 – 9       |
| GLEAM  evaporation   | ```gleam.nc```        | 0.25º | 210 – 330      | 5 – 60        | 4 – 8       |
| Topography           | ```topo.nc```         | 1/12º | 0 – 359        | -90 – 90      | N/A         |

The PRISM and GLEAM data should be saved to ```${DATA_FP}/raw``` and the topography data should be saved to ```${DATA_FP}/constants```.

#### For Poseidon users at WHOI:
1. Use the following commands to obtain the data (stored locally) for each year in [1979,2020]:
```
cd ./scripts/get_data
./move_era5_parallel.sh ${DATA_FP}/raw
```
When the jobs for each year are complete, merge the data for each year with:
```
sbatch ./merge_era5.sh ${DATA_FP}/raw
```
2. Obtain constants with ```sbatch ./download_constants.sh```
3. Obtain PRISM and GLEAM data with: ```sbatch get_EP_comps.sh```
4. Obtain moisture tracking output data with:
```
cp -r ${WAM_FP} ${DATA_FP}/raw
```

### Data pre-processing
0. Navigate back to project's root directory:
```
cd ~/gpllj-moisture-tracking
```

1. Compute Helmholtz decomposition of IVT:
```
conda activate ./envs/env_windspharm
python ./scripts/preprocess/ivt_helmholtz.py
```

2. Reactivate primary mamba environment and compute LWA:
```
conda activate ./envs/env_main
python ./scripts/preprocess/get_lwa.py
```

3. Aggregate data (sum precip./evap. over the Midwest)
```
python ./scripts/preprocess/postprocess_wam.py
python ./scripts/preprocess/aggregate_EP_comps.py
```

4. Detrend and compute seasonal averages
```
python ./scripts/preprocess/preprocess.py
```

### Reproducing figures
1. Specify season in ```./src/params.py``` (choose one of ```{amj,mjj,jja}```)
2. Run all scripts with:
```
cd ./scripts
./run_all.sh
```

## Notes & acknowledgments  
The ERA-5 data can be downloaded from the Copernicus Climate Change Service's Climate Data Store (\url{https://cds.climate.copernicus.eu}). Navy topography data obtained from https://iridl.ldeo.columbia.edu/SOURCES/.WORLDBATH/. Code for the moisture tracking model was adapted from Ruud Van der Ent's original implementation and can be accessed at \url{https://github.com/ktcarr/WAM2layers_ERA5}.

## Results

### Water vapor and wind
<p float="left">
  <img src="https://github.com/ktcarr/gpllj-moisture-tracking/blob/main/results/fluxes_spatial.png" width="750" />
</p>

### Precipitation covariance with GPLLJ
<p float="left">
  <img src="https://github.com/ktcarr/gpllj-moisture-tracking/blob/main/results/precip_regression.png" width="750" />
</p>

### Moisture flux in the GPLLJ region
<p float="left">
 <img src="https://github.com/ktcarr/gpllj-moisture-tracking/blob/main/results/fluxes_gpllj.png "width="750" />
</p>

### Moisture sources (spatial map, seasonal timescale)
<p float="left">
 <img src="https://github.com/ktcarr/gpllj-moisture-tracking/blob/main/results/moisture_sources_seasonal_spatial.png" width="750" />
</p>

### Moisture sources (aggregated totals)
<p float="left">
  <img src="https://github.com/ktcarr/gpllj-moisture-tracking/blob/main/results/moisture_sources_seasonal_agg.png" width="500" />
</p>

### Synoptic moisture sources
<p float="left">
  <img src="https://github.com/ktcarr/gpllj-moisture-tracking/blob/main/results/synoptic_composite_v850.png" width="750" />
</p>

### Trend over time
<p float="left">
     <img src="https://github.com/ktcarr/gpllj-moisture-tracking/blob/main/results/trends_over_time_v.png" width="750" />
</p>
