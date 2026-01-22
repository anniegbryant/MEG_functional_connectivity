#!/bin/bash

# The overall repo is the base repo
# MEG_base_repo=$(dirname $(dirname $(pwd)))
MEG_base_repo=/taiji1/abry4213/github/MEG_functional_connectivity

# Update to where your bids_root directory is
# bids_root=/path/to/bids_root
bids_root=/taiji1/abry4213/data/Cogitate_MEG

# n_jobs is set to 1 by default, you can increase this number to speed up the classification
n_jobs=16

# Use logistic regression as the default classifier
# classifier=Logistic_Regression

# Uncomment the following line to use linear SVM instead
# classifier=Linear_SVM

################################## Call classifiers ###################################

for classifier in Logistic_Regression Linear_SVM RBF_SVM; do
    export cmd="qsub -o /taiji1/abry4213/github/MEG_functional_connectivity/cluster_output/MEG_FC_classification_with_${classifier}.out \
    -N MEG_FC_classification \
    -l select=1:ncpus=${n_jobs}:mem=10GB:mpiprocs=${n_jobs} \
    -v bids_root=$bids_root,n_jobs=$n_jobs,MEG_base_repo=$MEG_base_repo,classifier=$classifier \
    call_classification.pbs"

    echo "$cmd"
    eval "$cmd"
done