#!/usr/bin/env bash

# Update to where your bids_root directory is
bids_root=/path/to/bids_root

# Replace with your FreeSurfer installation locations
module load freesurfer/7.1.1
export FREESURFER_HOME=/usr/local/freesurfer/7.1.1
source $FREESURFER_HOME/SetUpFreeSurfer.sh
export SUBJECTS_DIR=${bids_root}/derivatives/fs
export PATH="$FREESURFER_HOME/bin:$PATH"
export PATH="$FREESURFER_HOME/fsfast/bin:$PATH"

# Define derivatives directories
export coreg_dir=${bids_root}/derivatives/coreg
export out_fw=${bids_root}/derivatives/forward

# The ogitate-msp1 github repo should be cloned as a submodule here
COGITATE_MSP_repo=coglib/meeg

# The overall repo is the base repo
MEG_base_repo=$(dirname $(dirname $(pwd)))

# Cogitate subject list
subject_list=$MEG_base_repo/data/metadata/subject_list_Cogitate_MEG_with_all_data.txt

# Using visit 1
visit=1

# Using record run
record=run

######################################################################################
# MNE preprocessing
######################################################################################

# Iterate over subjects in the given data directory
echo "Running MNE preprocessing"

for subject in $(cat $subject_list); do
    echo "Processing subject $subject"
    # Run the preprocessing script for step 1
    python3 $COGITATE_MSP_repo/preprocessing/P99_run_preproc.py \
    --sub $subject --visit $visit --record $record --step 1 \
    --bids_root $bids_root

    # Run the preprocessing script for step 2
    python3 $COGITATE_MSP_repo/preprocessing/P99_run_preproc.py \
    --sub $subject --visit $visit --record $record --step 2 \
    --bids_root $bids_root
done


######################################################################################
# Run FreeSurfer's recon-all pipeline
######################################################################################

for subject in $(cat $subject_list); do
    echo "Running recon-all for subject $subject"
    # Run the recon-all script
    recon-all -all -subjid sub-${subject} -i ${bids_root}/sub-${subject}/ses-1/anat/sub-${subject}_ses-1_T1w.nii.gz -sd ${bids_root}/derivatives/fs
done

######################################################################################
# Run MNE's scalp reconstruction
######################################################################################

for subject in $(cat $subject_list); do
    echo "Running MNE's scalp reconstruction for subject $subject"
    # Run the scalp reconstruction script
    python3 $COGITATE_MSP_repo/source_modelling/S00a_scalp_surfaces.py \
    --sub $subject --visit $visit --bids_root $bids_root --fs_home $FREESURFER_HOME --subjects_dir $SUBJECTS_DIR
done

######################################################################################
# Fit Single-shell Boundary Elements Model
######################################################################################

for subject in $(cat $subject_list); do
    echo "Fitting Single-shell Boundary Elements Model for subject $subject"

    # Run the BEM script
    python3 $COGITATE_MSP_repo/source_modelling/S00b_bem.py \
    --sub $subject --visit $visit --bids_root $bids_root --fs_home $FREESURFER_HOME --subjects_dir $SUBJECTS_DIR

    # Run forward model
    python3 $COGITATE_MSP_repo/source_modelling/S01_forward_model.py \
    --sub $subject --visit $visit --space surface --bids_root $bids_root --subjects_dir $SUBJECTS_DIR \
    --coreg_path $coreg_dir --out_fw $out_fw
done

######################################################################################
# Average across epochs to get a single event-related field (ERF) time series per meta-ROI
######################################################################################

# Use 1 job by default, you can increase as your system allows
n_jobs=1

for subject in $(cat $subject_list); do
    echo "Averaging across ERFs for subject $subject"

    # Run the script to average across epochs
    python3 MEG_preprocessing/extract_time_series_from_MEG.py --sub $subject \
    --bids_root $bids_root --regions brain_regions_a_priori.json --n_jobs $n_jobs
done

# Done :)
echo "All finished with preprocessing. Ready for time-series feature extraction."