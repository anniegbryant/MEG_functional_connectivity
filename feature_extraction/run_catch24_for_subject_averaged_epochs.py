import pandas as pd
import numpy as np
import pycatch22
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

# Define ROI lookup table
ROI_lookup = {"proc-0": "Category_Selective",
                "proc-1": "IPS",
                "proc-2": "Prefrontal_Cortex",
                "proc-3": "V1_V2"}
region_names = list(ROI_lookup.values())
    
if op.isfile(f"{output_feature_path}/sub-{subject_id}_ses-{visit_id}_catch24_results_{duration}.csv"):
    print(f"catch24 results for sub-{subject_id} already exist. Skipping.")
    exit() 

# Iterate over all the time-series files for this subject
sample_TS_data_list = []

sample_TS_data=pd.read_csv(f"{time_series_path}/sub-{subject_id}_ses-{visit_id}_meg_{duration}_all_time_series.csv")
sample_TS_data['duration'] = sample_TS_data['duration'].str.replace('ms', '').astype(int)
sample_TS_data['times'] = np.round(sample_TS_data['times']*1000)
sample_TS_data['times'] = sample_TS_data['times'].astype(int)

# Filter times >= 0
sample_TS_data = sample_TS_data.query('times >= 0')

# Assign stimulus as on if times < duration and off if times >= duration
sample_TS_data['stimulus'] = np.where(sample_TS_data['times'] < sample_TS_data['duration'], 'on', 'off')

# Create list of dataframes for each stimulus_type, relevance_type, duration, and frequency_band
# One list for 'on' (while stimulus is being presented) and another for 'off' (after stimulus is no longer being presented)
sample_TS_data_list = []
for stimulus_type in sample_TS_data['stimulus_type'].unique():
    for relevance_type in sample_TS_data['relevance_type'].unique():
        for duration in sample_TS_data['duration'].unique():
            for stimulus_presentation in ['on', 'off']:
            # for duration in sample_TS_data['duration'].unique():
                this_condition_data = sample_TS_data.query('stimulus_type == @stimulus_type and relevance_type == @relevance_type and duration == @duration and stimulus == @stimulus_presentation')
                if this_condition_data.empty:
                    print(f"Missing data for {stimulus_type}, {relevance_type}, {duration}, {stimulus_presentation}")
                    continue
                sample_TS_data_list.append(this_condition_data)

def run_catch24_for_df(subject_id, df):
        # Pivot so that the columns are meta_ROI and the rows are data
        df_for_catch24 = df.filter(region_names)

        # Compute catch24 features for each column (meta-ROI) of sample_TS_mat
        subject_catch24_res = np.apply_along_axis(pycatch22.catch22_all, 0, df_for_catch24, short_names=True, catch24=True)
        subject_catch24_res_list = []

        # Iterate over each network and extract the catch24 features into a dataframe
        for network_index in range(len(region_names)):
            region_name = region_names[network_index]
            catch24_features = subject_catch24_res[network_index]
            region_catch24_df = (pd.DataFrame(catch24_features)
                                .assign(Meta_ROI=region_name)
                                .rename(columns={"short_names": "catch24_feature", "values": "feature_value"})
                                .drop(columns='names'))

            # Append to list
            subject_catch24_res_list.append(region_catch24_df)

        # Concatenate all results
        catch24_results_df = pd.concat(subject_catch24_res_list).assign(subject_ID = subject_id,
                                                                        stimulus_type = df.stimulus_type.unique()[0],
                                                                        relevance_type = df.relevance_type.unique()[0],
                                                                        duration = df.duration.unique()[0],
                                                                        stimulus = df.stimulus.unique()[0])

        return catch24_results_df
# Initialise an empty list for the results
catch24_res_list = []

# Run for data
for dataframe in sample_TS_data_list:
    dataframe_catch24 = run_catch24_for_df(subject_id, dataframe).assign(stimulus="on")
    catch24_res_list.append(dataframe_catch24)

# Concatenate the results and save to a feather file
all_catch24_res = pd.concat(catch24_res_list).reset_index() 
all_catch24_res.to_csv(f"{output_feature_path}/sub-{subject_id}_ses-{visit_id}_all_catch24_results_{duration}ms.csv", index=False)