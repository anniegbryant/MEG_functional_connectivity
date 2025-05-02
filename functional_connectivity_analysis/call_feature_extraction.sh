#!/usr/bin/env bash

# The overall repo is the base repo
MEG_base_repo=$(dirname $(dirname $(pwd)))

# Update to where your bids_root directory is
bids_root=/path/to/bids_root

# Cogitate subject list
subject_list=$MEG_base_repo/data/metadata/subject_list_Cogitate_MEG_with_all_data.txt

# Always use visid_id 1
visit_id=1
duration='1000ms'

##################################################################################################
# Running pyspi across subjects, averaged epochs
##################################################################################################

# Supplemental SPIs for fast pyspi
FC_measures_yaml=$MEG_base_repo/feature_extraction/functional_connectivity_measures.yaml

for subject in $(cat $subject_list); do
   python3 $MEG_base_repo/feature_extraction/run_pyspi_for_subject_averaged_epochs.py --sub $subject \
      --visit_id $visit_id \
      --bids_root $bids_root \
      --SPI_subset $FC_measures_yaml \
      --duration $duration
done
