#!/usr/bin/env bash

# Define the batch job array command
# input_model_file=/project/hctsa/annie/github/MEG_functional_connectivity/subject_list_Cogitate_MEG.txt
input_model_file=/headnode1/abry4213/github/MEG_functional_connectivity/subject_list_Cogitate_MEG.txt

##################################################################################################
# Running pyspi across subjects, averaged epochs
##################################################################################################

# # Averaged epochs
# cmd="qsub -o /headnode1/abry4213/github/Cogitate_Connectivity_2024/cluster_output/Cogitate_MEG_pyspi_averaged_^array_index^_fast.out \
#    -J 1-100 \
#    -N fast_pyspi_MEG \
#    -l select=1:ncpus=1:mem=20GB:mpiprocs=1 \
#    -v input_model_file=$input_model_file \
#    run_pyspi_for_subject_averaged_epochs.pbs"
# $cmd

##################################################################################################
# Running pyspi across subjects, individual epochs
##################################################################################################

# Individual epochs
# cmd="qsub -o /headnode1/abry4213/github/Cogitate_Connectivity_2024/cluster_output/pyspi_for_individual_epochs_^array_index^_fast.out \
#    -N pyspi_individual_epochs \
#    -J 1-100 \
#    -l select=1:ncpus=1:mem=20GB:mpiprocs=1 \
#    -v input_model_file=$input_model_file,batch_number=$batch_number \
#    run_pyspi_for_subject_individual_epochs.pbs"
# $cmd

# Line 14
for line_to_read in 14; do 
   cmd="qsub -o /headnode1/abry4213/github/Cogitate_Connectivity_2024/cluster_output/pyspi_for_individual_epochs_^array_index^_fast.out \
   -N pyspi_individual_epochs \
   -l select=1:ncpus=1:mem=20GB:mpiprocs=1 \
   -v line_to_read=$line_to_read,input_model_file=$input_model_file,batch_number=$batch_number \
   run_pyspi_for_subject_individual_epochs.pbs"
   $cmd
done