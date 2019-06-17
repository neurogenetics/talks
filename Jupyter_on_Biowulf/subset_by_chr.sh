!/bin/env bash

mkdir PLINK_SUBSET

module load plink 

for chromosome in {1..22};
do
    plink --bfile hapmap1_CR --chr ${chromosome} --make-bed --out ./PLINK_SUBSET/CHR${chromosome}_hapmap1_CR 
done
