DATA_FP=$1
SAVE_FP=$2

mkdir {${DATA_FP}/constants,${DATA_FP}/prepped,${DATA_FP}/raw}

for SEASON in amj mjj jja
do 
    mkdir ${SAVE_FP}/${SEASON}
    mkdir ${SAVE_FP}/${SEASON}/supp
done
