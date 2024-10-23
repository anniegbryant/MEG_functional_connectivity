#!/usr/bin/env bash

# Define the batch job array command
github_dir=/headnode1/abry4213/github/MEG_functional_connectivity
input_model_file=$github_dir/subject_list_Cogitate_MEG_pyspi.txt

##################################################################################################
# Running pyspi across subjects, averaged epochs
##################################################################################################

# Averaged epochs
# cmd="qsub -o $github_dir/cluster_output/Cogitate_MEG_pyspi_averaged_^array_index^_fast.out \
#    -J 1-94 \
#    -N fast_pyspi_MEG \
#    -l select=1:ncpus=1:mem=20GB:mpiprocs=1 \
#    -v input_model_file=$input_model_file,SPI_subset=fast \
#    run_pyspi_for_subject_averaged_epochs.pbs"
# $cmd

# # Supplemental SPIs for fast pyspi
# fast_supp_yaml=$github_dir/feature_extraction/fast_supplement.yaml
# cmd="qsub -o $github_dir/cluster_output/Cogitate_MEG_pyspi_averaged_^array_index^_fast_supplement.out \
#    -J 1-94 \
#    -N fast_supp_pyspi_MEG \
#    -l select=1:ncpus=1:mem=20GB:mpiprocs=1 \
#    -v input_model_file=$input_model_file,SPI_subset=$fast_supp_yaml \
#    run_pyspi_for_subject_averaged_epochs.pbs"
# $cmd

# # Merge averaged epochs
# python3 $github_dir/feature_extraction/merge_pyspi_res_averaged_epochs.py

##################################################################################################
# Running pyspi across subjects, individual epochs
##################################################################################################

# # Individual epochs
# cmd="qsub -o $github_dir/cluster_output/pyspi_for_individual_epochs_^array_index^_fast.out \
#    -N pyspi_individual_epochs \
#    -J 1-94 \
#    -l select=1:ncpus=1:mem=20GB:mpiprocs=1 \
#    -v input_model_file=$input_model_file \
#    run_pyspi_for_subject_individual_epochs.pbs"
# $cmd

# Supplemental SPIs for fast pyspi
fast_supp_yaml=$github_dir/feature_extraction/fast_supplement.yaml
cmd="qsub -o $github_dir/cluster_output/pyspi_supplement_for_individual_epochs_^array_index^_fast.out \
   -N pyspi_supp_individual_epochs \
   -J 1-94 \
   -l select=1:ncpus=1:mem=20GB:mpiprocs=1 \
   -v input_model_file=$input_model_file,SPI_subset=$fast_supp_yaml \
   run_pyspi_for_subject_individual_epochs.pbs"
$cmd