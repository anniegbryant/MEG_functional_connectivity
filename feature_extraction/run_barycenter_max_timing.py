import pandas as pd
import numpy as np
from tslearn import barycenters
from scipy.stats import zscore
import os.path as op
from copy import deepcopy
import argparse

parser=argparse.ArgumentParser()
parser.add_argument('--sub',
                    type=str,
                    default='CB040',
                    help='subject_id (e.g. "CB040")')
parser.add_argument('--visit_id',
                    type=str,
                    default='1',
                    help='Visit ID (e.g., 1)')
parser.add_argument('--bids_root',
                    type=str,
                    default='/project/hctsa/annie/data/Cogitate_MEG/',
                    help='Path to the BIDS root directory')
parser.add_argument('--duration',
                    type=str,
                    default='1000ms',
                    help="Trial duration to compute (default is '1000ms')")
opt=parser.parse_args()

subject_id = opt.sub 
bids_root = opt.bids_root
visit_id = opt.visit_id
duration = opt.duration

# Time series output path for this subject
time_series_path = op.join(bids_root, "derivatives", "MEG_time_series")
output_feature_path = op.join(bids_root, "derivatives", "time_series_features/averaged_epochs")

# Define this subject's time-series file
subject_averaged_TS_file = f"{time_series_path}/sub-{subject_id}_ses-{visit_id}_meg_{duration}_all_time_series.csv"

# Define output barycenter stat file
output_barycenter_file = f"{output_feature_path}/sub-{subject_id}_ses-{visit_id}_all_barycenter_stat_{duration}.csv"

if op.isfile(output_barycenter_file):
    print(f"Barycenter stats for sub-{subject_id} already exist. Skipping.")
    exit()

# # Define barycenter methods
# barycenter_method_dict = {'euclidean': barycenters.euclidean_barycenter,
#                           'softdtw': barycenters.softdtw_barycenter,
#                           'dtw': barycenters.dtw_barycenter_averaging,
#                           'sgddtw': barycenters.dtw_barycenter_averaging_subgradient}

# Define barycenter methods
barycenter_method_dict = {'euclidean': barycenters.euclidean_barycenter,
                          'dtw': barycenters.dtw_barycenter_averaging}

def barycenter_helper_stats_function(barycenter_onset, barycenter_offset):
    ## STATS FROM ORIGINAL BARYCENTER ##
    try:
        barycenter_onset_mean = np.mean(barycenter_onset)
        barycenter_onset_max = np.max(barycenter_onset)
        barycenter_onset_max_time = np.argmax(barycenter_onset)
    except:
        barycenter_onset_mean = np.nan
        barycenter_onset_max = np.nan
        barycenter_onset_max_time = np.nan

    try:
        barycenter_first500_mean = np.mean(barycenter_onset[:500])
        barycenter_first500_max = np.max(barycenter_onset[:500])
        barycenter_first500_max_time = np.argmax(barycenter_onset[:500])
    except:
        barycenter_first500_mean = np.nan
        barycenter_first500_max = np.nan
        barycenter_first500_max_time = np.nan

    try:
        barycenter_second500_mean = np.mean(barycenter_onset[500:])
        barycenter_second500_max = np.max(barycenter_onset[500:])
        barycenter_second500_max_time = np.argmax(barycenter_onset[500:])
    except:
        barycenter_second500_mean = np.nan
        barycenter_second500_max = np.nan
        barycenter_second500_max_time = np.nan

    try:
        barycenter_offset_mean = np.mean(barycenter_offset)
        barycenter_offset_max = np.max(barycenter_offset)
        barycenter_offset_max_time = np.argmax(barycenter_offset)
    except:
        barycenter_offset_mean = np.nan
        barycenter_offset_max = np.nan
        barycenter_offset_max_time = np.nan
    
    ## STATS FROM SQUARED BARYCENTER ##
    barycenter_onset_squared = barycenter_onset ** 2
    barycenter_offset_squared = barycenter_offset ** 2

    try:
        barycenter_onset_squared_mean = np.mean(barycenter_onset_squared)
        barycenter_onset_squared_max = np.max(barycenter_onset_squared)
        barycenter_onset_squared_max_time = np.argmax(barycenter_onset_squared)
    except:
        barycenter_onset_squared_mean = np.nan
        barycenter_onset_squared_max = np.nan
        barycenter_onset_squared_max_time = np.nan
    
    try:
        barycenter_first500_squared_mean = np.mean(barycenter_onset_squared[:500])
        barycenter_first500_squared_max = np.max(barycenter_onset_squared[:500])
        barycenter_first500_squared_max_time = np.argmax(barycenter_onset_squared[:500])
    except:
        barycenter_first500_squared_mean = np.nan
        barycenter_first500_squared_max = np.nan
        barycenter_first500_squared_max_time = np.nan

    try:
        barycenter_second500_squared_mean = np.mean(barycenter_onset_squared[500:])
        barycenter_second500_squared_max = np.max(barycenter_onset_squared[500:])
        barycenter_second500_squared_max_time = np.argmax(barycenter_onset_squared[500:])
    except:
        barycenter_second500_squared_mean = np.nan
        barycenter_second500_squared_max = np.nan
        barycenter_second500_squared_max_time = np.nan

    try:
        barycenter_offset_squared_mean = np.mean(barycenter_offset_squared)
        barycenter_offset_squared_max = np.max(barycenter_offset_squared)
        barycenter_offset_squared_max_time = np.argmax(barycenter_offset_squared)
    except:
        barycenter_offset_squared_mean = np.nan
        barycenter_offset_squared_max = np.nan
        barycenter_offset_squared_max_time = np.nan

    # Compile results into a dataframe
    barycenter_stats_df = pd.DataFrame({"Presentation": ["Onset", "Offset", "Onset", "Offset", "Onset_First500", "Onset_Second500", "Onset_First500", "Onset_Second500"],
                                        "Barycenter_Type": ["Original", "Original", "Squared", "Squared", "Original", "Original", "Squared", "Squared"],
                                        "Mean": [barycenter_onset_mean, barycenter_offset_mean, barycenter_onset_squared_mean, barycenter_offset_squared_mean,
                                                barycenter_first500_mean, barycenter_second500_mean, barycenter_first500_squared_mean, barycenter_second500_squared_mean],
                                        "Max": [barycenter_onset_max, barycenter_offset_max, barycenter_onset_squared_max, barycenter_offset_squared_max,
                                               barycenter_first500_max, barycenter_second500_max, barycenter_first500_squared_max, barycenter_second500_squared_max],
                                        "Max_Time": [barycenter_onset_max_time, barycenter_offset_max_time, barycenter_onset_squared_max_time, barycenter_offset_squared_max_time,
                                                    barycenter_first500_max_time, barycenter_second500_max_time, barycenter_first500_squared_max_time, barycenter_second500_squared_max_time]})
    return(barycenter_stats_df)



# Define helper function to compute barycenter methods
def compute_barycenter_stats_for_subject(subject_ID, stim_rel_data, barycenter_method_dict):

    # Filter to specific regions during onset
    CS_onset = zscore(stim_rel_data.query("times >= 0 & times <= 1").Category_Selective.values)
    PFC_onset = zscore(stim_rel_data.query("times >= 0 & times <= 1").Prefrontal_Cortex.values)
    VIS_onset = zscore(stim_rel_data.query("times >= 0 & times <= 1").V1_V2.values)

    # Filter to specific regions during offset
    CS_offset = zscore(stim_rel_data.query("times > 1").Category_Selective.values)
    PFC_offset = zscore(stim_rel_data.query("times > 1").Prefrontal_Cortex.values)
    VIS_offset = zscore(stim_rel_data.query("times > 1").V1_V2.values)

    # Initialize list for results
    all_barycenter_method_res_list = []

    # Iterate over the barycenter methods in barycenter_method_dict
    for method_name, method_func in barycenter_method_dict.items():

        print(f"Computing {method_name} barycenter for {subject_ID}")

        # Compute the time-resolved barycenter for each region--region pair
        try:
            CS_PFC_onset_barycenter = method_func([CS_onset, PFC_onset])
            CS_PFC_offset_barycenter = method_func([CS_offset, PFC_offset])

            # Compute stats for CS_PFC 
            CS_PFC_barycenter_stats_df = (barycenter_helper_stats_function(CS_PFC_onset_barycenter, CS_PFC_offset_barycenter)
                                        .assign(Subject=subject_ID, 
                                                Region="CS_PFC", 
                                                Barycenter_Method=method_name))

            # Append results from this barycenter method to the list
            all_barycenter_method_res_list.append(CS_PFC_barycenter_stats_df)
        except:
            print(f"Error computing {method_name} barycenter for {subject_ID} CS_PFC")

        # Compute stats for CS_VIS
        try:
            CS_VIS_onset_barycenter = method_func([CS_onset, VIS_onset])
            CS_VIS_offset_barycenter = method_func([CS_offset, VIS_offset])
            CS_VIS_barycenter_stats_df = (barycenter_helper_stats_function(CS_VIS_onset_barycenter, CS_VIS_offset_barycenter)
                                        .assign(Subject=subject_ID, 
                                                Region="CS_VIS", 
                                                Barycenter_Method=method_name))
            
            # Append results from this barycenter method to the list
            all_barycenter_method_res_list.append(CS_VIS_barycenter_stats_df)
        except:
            print(f"Error computing {method_name} barycenter for {subject_ID} CS_VIS")

    # Return the dataframe
    all_barycenter_method_res = pd.concat(all_barycenter_method_res_list)
    return all_barycenter_method_res

# Extract subject ID and load in data
subject_ID = op.basename(subject_averaged_TS_file).replace("_ses-1_meg_1000ms_all_time_series.csv", "")
subject_averaged_TS = pd.read_csv(subject_averaged_TS_file)

# Iterate over each stimulus_type and relevance_type combination
subject_barycenter_res_list = []
for this_stim in subject_averaged_TS.stimulus_type.unique():
    for this_rel in subject_averaged_TS.relevance_type.unique():
        # Filter to specific stimulus and relevance type
        stim_rel_data = subject_averaged_TS.query(f"stimulus_type == '{this_stim}' & relevance_type == '{this_rel}'")

        # Compute barycenter stats for this subject, stimulus, and relevance
        stim_rel_barycenter_res = (compute_barycenter_stats_for_subject(subject_ID=subject_ID, 
                                                                       stim_rel_data=stim_rel_data, 
                                                                       barycenter_method_dict=barycenter_method_dict)
                                                                       .assign(Stimulus=this_stim, Relevance=this_rel))
        
        # Append to the list
        subject_barycenter_res_list.append(stim_rel_barycenter_res)

# Concatenate the results into one dataframe
subject_barycenter_res = pd.concat(subject_barycenter_res_list)

# Save results to a CSV
subject_barycenter_res.to_csv(output_barycenter_file, index=False)