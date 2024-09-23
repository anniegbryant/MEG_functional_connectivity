#!/usr/bin/env bash

# Define the batch job array command
input_model_file=/project/hctsa/annie/github/MEG_functional_connectivity/subject_list_Cogitate_MEG_with_all_data.txt
# input_model_file=/headnode1/abry4213/github/MEG_functional_connectivity/subject_list_Cogitate_MEG_with_all_data.txt

# A priori selected regions file
regions_file=/project/hctsa/annie/github/MEG_functional_connectivity/annie_chris_ROIs.json

##################################################################################################
# Preprocessing [Artemis, batch array]
##################################################################################################

# # Step 1+2 together
# cmd="qsub -o /project/hctsa/annie/github/MEG_functional_connectivity/cluster_output/Cogitate_MEG_preproc_^array_index^.out \
#    -N Cogitate_MEG_preproc_1 \
#    -J 1-10 \
#    -l select=1:ncpus=1:mem=40GB:mpiprocs=1 \
#    -v input_model_file=$input_model_file,step=2 \
#    1_preprocess_MEG_subjects.pbs"
# echo $cmd
# $cmd

##################################################################################################
# recon-all [Artemis, individual jobs]
##################################################################################################

# # Define the recon-all command loop
# cat $input_model_file | while read line 
# do
#    subject=$line
#    cmd="qsub -o /project/hctsa/annie/github/MEG_functional_connectivity/cluster_output/recon_all_${subject}.out \
#    -N ${subject}_recon_all \
#    -v subject=$subject \
#    2_recon_all.pbs"

#    # Run the command
#    $cmd
# done

##################################################################################################
# Scalp reconstruction [Physics cluster]
##################################################################################################

# Physics cluster
# cmd="qsub -o /headnode1/abry4213/github/Cogitate_Connectivity_2024/cluster_output/Cogitate_MEG_scalp_recon_^array_index^.out \
#    -N All_scalp_recon \
#    -J 1-94 \
#    -v input_model_file=$input_model_file \
#    3_scalp_recon.pbs"
# $cmd

##################################################################################################
# BEM
##################################################################################################

# # Define the command
# cmd="qsub -o /project/hctsa/annie/github/MEG_functional_connectivity/cluster_output/Cogitate_MEG_BEM_^array_index^.out \
# -N BEM \
# -J 1-94 \
# -v input_model_file=$input_model_file \
# 4_BEM.pbs"

# # Run the command
# $cmd

##################################################################################################
# Extract time series across participants
##################################################################################################

# Define the command
num_cores=10
n_jobs=4
cmd="qsub -o /project/hctsa/annie/github/MEG_functional_connectivity/cluster_output/MEG_extract_time_series_^array_index^.out \
-N MEG_extract_time_series \
-J 1-10 \
-v input_model_file=$input_model_file,regions_file=$regions_file,n_jobs=$n_jobs \
-l select=1:ncpus=$num_cores:mem=120GB:mpiprocs=$num_cores \
5_extract_time_series.pbs"

echo $cmd

# Run the command
$cmd

##################################################################################################
# Combine time series for participant
##################################################################################################

# Define the command
# cmd="qsub -o /project/hctsa/annie/github/MEG_functional_connectivity/cluster_output/MEG_combine_time_series_^array_index^.out \
# -N MEG_combine_time_series \
# -J 1-52 \
# -v input_model_file=$input_model_file \
# -l select=1:ncpus=1:mem=10GB:mpiprocs=1 \
# 6_combine_time_series.pbs"

# echo $cmd

# # Combine all epoch-averaged results into one zipped file
# bids_root=/project/hctsa/annie/data/Cogitate_MEG/
# time_series_file_path=$bids_root/derivatives/MEG_time_series

# # File compression
# cd ${time_series_file_path}
# zip ${time_series_file_path}/all_epoch_averaged_time_series.zip sub-*_ses-*_meg_*_all_time_series.csv