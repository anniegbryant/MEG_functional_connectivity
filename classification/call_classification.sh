#!/usr/bin/env bash

# Define the batch job array command
input_model_file=/project/MEG/github/MEG_functional_connectivity/subject_list_Cogitate_MEG_pyspi.txt
# input_model_file=/headnode1/abry4213/github/MEG_functional_connectivity/subject_list_Cogitate_MEG_pyspi.txt

github_repo=/project/MEG/github/MEG_functional_connectivity
# github_repo=/headnode1/abry4213/github/MEG_functional_connectivity

bids_root=/project/MEG/data/Cogitate_MEG/
# bids_root=/headnode1/abry4213/data/Cogitate_MEG/

# ###################### Averaged epoch classification, catch24 ##################
# n_jobs=10
# for classifier in Linear_SVM Logistic_Regression; do
#    cmd="qsub -o $github_repo/cluster_output/Cogitate_MEG_group_averaged_${classifier}classification.out \
#       -N ${classifier}_MEG_averaged_classification \
#       -l select=1:ncpus=$n_jobs:mem=20GB:mpiprocs=$n_jobs \
#       -v bids_root=$bids_root,github_repo=$github_repo,input_model_file=$input_model_file,n_jobs=$n_jobs,classifier=$classifier \
#       run_averaged_catch24_classifiers.pbs"
#    $cmd
# done

###################### Averaged epoch classification, pyspi ##################
# n_jobs=10
# for classifier in Linear_SVM Logistic_Regression; do
#    cmd="qsub -o $github_repo/cluster_output/Cogitate_MEG_group_averaged_${classifier}classification.out \
#       -N ${classifier}_MEG_averaged_classification \
#       -l select=1:ncpus=$n_jobs:mem=20GB:mpiprocs=$n_jobs \
#       -v bids_root=$bids_root,github_repo=$github_repo,input_model_file=$input_model_file,n_jobs=$n_jobs,classifier=$classifier \
#       run_averaged_pyspi_classifiers.pbs"
#    $cmd
# done

########## Indivivdual epoch classification, pyspi, with subsampling #########
n_jobs=10
# for classification_type in individual individual_subsampled; do
for classification_type in individual_subsampled; do
   for classifier in Linear_SVM Logistic_Regression; do
      cmd="qsub -o $github_repo/cluster_output/Cogitate_MEG_individual_epoch_${classifier}_subsampled_classification_^array_index^.out \
         -N ${classifier}_MEG_individual_classification \
         -J 1-94 \
         -l select=1:ncpus=$n_jobs:mem=20GB:mpiprocs=$n_jobs \
         -v bids_root=$bids_root,github_repo=$github_repo,input_model_file=$input_model_file,n_jobs=$n_jobs,classifier=$classifier,classification_type=$classification_type \
         run_individual_pyspi_classifiers.pbs"
      $cmd
   done
done