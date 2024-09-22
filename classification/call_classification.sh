#!/usr/bin/env bash

# Define the batch job array command
# input_model_file=/project/hctsa/annie/github/MEG_functional_connectivity/subject_list_Cogitate_MEG.txt
input_model_file=/headnode1/abry4213/github/MEG_functional_connectivity/subject_list_Cogitate_MEG.txt

# github_repo=/project/hctsa/annie/github/MEG_functional_connectivity
github_repo=/headnode1/abry4213/github/MEG_functional_connectivity

# bids_root=/project/hctsa/annie/data/Cogitate_MEG/
bids_root=/headnode1/abry4213/data/Cogitate_MEG/
# input_model_file=/headnode1/abry4213/github/MEG_functional_connectivity/subject_list_Cogitate_MEG.txt

github_repo=/project/hctsa/annie/github/MEG_functional_connectivity
# github_repo=/headnode1/abry4213/github/MEG_functional_connectivity
input_model_file=/project/hctsa/annie/github/MEG_functional_connectivity/subject_list_Cogitate_MEG.txt
# input_model_file=/headnode1/abry4213/github/MEG_functional_connectivity/subject_list_Cogitate_MEG.txt

github_repo=/project/hctsa/annie/github/MEG_functional_connectivity
# github_repo=/headnode1/abry4213/github/MEG_functional_connectivity

###################### Averaged epoch classification ##################
n_jobs=10
cmd="qsub -o $github_repo/cluster_output/Cogitate_MEG_group_averaged_classification.out \
   -N MEG_averaged_classification \
   -l select=1:ncpus=$n_jobs:mem=20GB:mpiprocs=$n_jobs \
   -v input_model_file=$input_model_file,n_jobs=$n_jobs \
   run_averaged_classifiers.pbs"
$cmd

###################### Indivivdual epoch classification ##################
# n_jobs=10
# cmd="qsub -o $github_repo/cluster_output/Cogitate_MEG_individual_epoch_classification_^array_index^.out \
#    -N MEG_individual_classification \
#    -J 1-100 \
#    -l select=1:ncpus=$n_jobs:mem=20GB:mpiprocs=$n_jobs \
#    -v input_model_file=$input_model_file,n_jobs=$n_jobs,github_repo=$github_repo,bids_root=$bids_root \
#    run_individual_classifiers.pbs"
# $cmd