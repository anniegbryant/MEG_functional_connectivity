#!/usr/bin/env bash

# Define the batch job array command
input_model_file=/project/hctsa/annie/github/MEG_functional_connectivity/metadata/subject_list_Cogitate_MEG_with_all_data.txt

##################################################################################################
# Compute barycenter stats directly [Artemis, batch array]
##################################################################################################

cmd="qsub -o /project/hctsa/annie/github/MEG_functional_connectivity/cluster_output/Cogitate_MEG_barycenters_^array_index^.out \
   -N barycenters \
   -J 1-94 \
   -l select=1:ncpus=1:mem=20GB:mpiprocs=1 \
   -v input_model_file=$input_model_file \
   run_barycenter_max_timing.pbs"
echo $cmd
$cmd
