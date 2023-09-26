#!/bin/bash

DATA_FP=$1

for i in {1979..2021..1}
    do
        sbatch move-era5_oneyear.sh ${DATA_FP} $i
    done

