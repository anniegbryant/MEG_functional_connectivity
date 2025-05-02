#!/usr/bin/env bash

# The overall repo is the base repo
MEG_base_repo=$(dirname $(dirname $(pwd)))

# Update to where your bids_root directory is
bids_root=/path/to/bids_root

# Use logistic regression as the default classifier
classifier=Logistic_Regression

# n_jobs is set to 1 by default, you can increase this number to speed up the classification
n_jobs=1

# Uncomment the following line to use linear SVM instead
# classifier=Linear_SVM

################################## Call classifiers ###################################

python3 $MEG_base_repo/classification/fit_pyspi_classifiers.py --bids_root $bids_root --n_jobs $n_jobs \
    --SPI_directionality_file $MEG_base_repo/feature_extraction/pyspi_SPI_info.csv \
    --classification_type averaged \
    --classifier $classifier
