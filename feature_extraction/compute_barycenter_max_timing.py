import os
import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt
from glob import glob
from tslearn import barycenters
from scipy.stats import zscore

# Define path for derivatives directory
deriv_dir = "/Users/abry4213/data/Cogitate_MEG/derivatives"
MEG_time_series_dir = f"{deriv_dir}/MEG_time_series"

# Define visit ID and duration
visit_id = "1"
duration = "1000ms"

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
        barycenter_offset_squared_mean = np.mean(barycenter_offset_squared)
        barycenter_offset_squared_max = np.max(barycenter_offset_squared)
        barycenter_offset_squared_max_time = np.argmax(barycenter_offset_squared)
    except:
        barycenter_offset_squared_mean = np.nan
        barycenter_offset_squared_max = np.nan
        barycenter_offset_squared_max_time = np.nan

    # Compile results into a dataframe
    barycenter_stats_df = pd.DataFrame({"Presentation": ["Onset", "Offset", "Onset", "Offset"],
                                        "Barycenter_Type": ["Original", "Original", "Squared", "Squared"],
                                        "Mean": [barycenter_onset_mean, barycenter_offset_mean, barycenter_onset_squared_mean, barycenter_offset_squared_mean],
                                        "Max": [barycenter_onset_max, barycenter_offset_max, barycenter_onset_squared_max, barycenter_offset_squared_max],
                                        "Max_Time": [barycenter_onset_max_time, barycenter_offset_max_time, barycenter_onset_squared_max_time, barycenter_offset_squared_max_time]})
    return(barycenter_stats_df)


# Load in individual subjects' MEG time series one at a time
for subject_averaged_TS_file in glob(f"{MEG_time_series_dir}/*_ses-1_meg_1000ms_all_time_series.csv"):

    # Extract subject ID and load in data
    subject_ID = os.path.basename(subject_averaged_TS_file).replace("_ses-1_meg_1000ms_all_time_series.csv", "")

    # Define subject's output file
    subject_output_file = f"{deriv_dir}/time_series_features/averaged_epochs/{subject_ID}_ses-{visit_id}_all_barycenter_stat_{duration}.csv"

    # Skip if the output file already exists
    if os.path.isfile(subject_output_file):
        print(f"Barycenter stats for {subject_ID} already exist. Skipping.")
        continue

    subject_averaged_TS = pd.read_csv(subject_averaged_TS_file)
    subject_barycenter_res_list = []

    # Iterate over each stimulus_type and relevance_type combination
    for this_stim in subject_averaged_TS.stimulus_type.unique():
        for this_rel in subject_averaged_TS.relevance_type.unique():

            # Filter to specific stimulus and relevance type
            stim_rel_data = subject_averaged_TS.query(f"stimulus_type == '{this_stim}' & relevance_type == '{this_rel}'")

            # Filter to specific regions during onset
            CS_onset = zscore(stim_rel_data.query("times >= 0 & times <= 1").Category_Selective.values)
            PFC_onset = zscore(stim_rel_data.query("times >= 0 & times <= 1").Prefrontal_Cortex.values)
            VIS_onset = zscore(stim_rel_data.query("times >= 0 & times <= 1").V1_V2.values)

            # Filter to specific regions during offset
            CS_offset = zscore(stim_rel_data.query("times > 1").Category_Selective.values)
            PFC_offset = zscore(stim_rel_data.query("times > 1").Prefrontal_Cortex.values)
            VIS_offset = zscore(stim_rel_data.query("times > 1").V1_V2.values)

            # Compute absolute values for each region
            CS_onset_abs = np.abs(CS_onset)
            PFC_onset_abs = np.abs(PFC_onset)
            VIS_onset_abs = np.abs(VIS_onset)

            CS_offset_abs = np.abs(CS_offset)
            PFC_offset_abs = np.abs(PFC_offset)
            VIS_offset_abs = np.abs(VIS_offset)

            # Iterate over the barycenter methods in barycenter_method_dict
            for method_name, method_func in barycenter_method_dict.items():

                print(f"Computing {method_name} barycenter for {subject_ID}, {this_stim}, {this_rel}")

                # Compute the time-resolved barycenter for each region--region pair
                try:
                    CS_PFC_onset_barycenter = method_func([CS_onset, PFC_onset])
                    CS_PFC_offset_barycenter = method_func([CS_offset, PFC_offset])

                    # Compute stats for CS_PFC 
                    CS_PFC_barycenter_stats_df = (barycenter_helper_stats_function(CS_PFC_onset_barycenter, CS_PFC_offset_barycenter)
                                                .assign(Subject=subject_ID, 
                                                        Region="CS_PFC", 
                                                        Relevance=this_rel,
                                                        Stimulus=this_stim,
                                                        Barycenter_Method=method_name))

                    # Append results from this barycenter method to the list
                    subject_barycenter_res_list.append(CS_PFC_barycenter_stats_df)
                except:
                    print(f"Error computing {method_name} barycenter for {subject_ID} CS_PFC")

                        # Compute stats for CS_PFC_abs
                try:
                    CS_PFC_abs_onset_barycenter = method_func([CS_onset_abs, PFC_onset_abs])
                    CS_PFC_abs_offset_barycenter = method_func([CS_offset_abs, PFC_offset_abs])
                    CS_PFC_abs_barycenter_stats_df = (barycenter_helper_stats_function(CS_PFC_abs_onset_barycenter, CS_PFC_abs_offset_barycenter)
                                                .assign(Subject=subject_ID, 
                                                        Region="CS_PFC_abs", 
                                                        Relevance=this_rel,
                                                        Stimulus=this_stim,
                                                        Barycenter_Method=method_name))
                    
                    # Append results from this barycenter method to the list
                    subject_barycenter_res_list.append(CS_PFC_abs_barycenter_stats_df)
                except:
                    print(f"Error computing {method_name} barycenter for {subject_ID} CS_PFC_abs")

                # Compute stats for CS_VIS
                try:
                    CS_VIS_onset_barycenter = method_func([CS_onset, VIS_onset])
                    CS_VIS_offset_barycenter = method_func([CS_offset, VIS_offset])
                    CS_VIS_barycenter_stats_df = (barycenter_helper_stats_function(CS_VIS_onset_barycenter, CS_VIS_offset_barycenter)
                                                .assign(Subject=subject_ID, 
                                                        Region="CS_VIS", 
                                                        Relevance=this_rel,
                                                        Stimulus=this_stim,
                                                        Barycenter_Method=method_name))
                    
                    # Append results from this barycenter method to the list
                    subject_barycenter_res_list.append(CS_VIS_barycenter_stats_df)
                except:
                    print(f"Error computing {method_name} barycenter for {subject_ID} CS_VIS")

                        # Compute stats for CS_VIS_abs
                try:
                    CS_VIS_abs_onset_barycenter = method_func([CS_onset_abs, VIS_onset_abs])
                    CS_VIS_abs_offset_barycenter = method_func([CS_offset_abs, VIS_offset_abs])
                    CS_VIS_abs_barycenter_stats_df = (barycenter_helper_stats_function(CS_VIS_abs_onset_barycenter, CS_VIS_abs_offset_barycenter)
                                                .assign(Subject=subject_ID, 
                                                        Region="CS_VIS_abs", 
                                                        Relevance=this_rel,
                                                        Stimulus=this_stim,
                                                        Barycenter_Method=method_name))
                    
                    # Append results from this barycenter method to the list
                    subject_barycenter_res_list.append(CS_VIS_abs_barycenter_stats_df)
                except:
                    print(f"Error computing {method_name} barycenter for {subject_ID} CS_VIS_abs")
                            
    # Concatenate the barycenter results into one dataframe
    subject_barycenter_res = pd.concat(subject_barycenter_res_list)

    # Save the concatenated dataframe
    subject_barycenter_res.to_csv(subject_output_file, index=False)