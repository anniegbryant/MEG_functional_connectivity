import pandas as pd
import numpy as np
from pyspi.calculator import Calculator
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
parser.add_argument('--SPI_subset',
                    type=str,
                    default='fast',
                    help='Subset of SPIs to compute')
parser.add_argument('--duration',
                    type=str,
                    default='1000ms',
                    help="Trial duration to compute (default is '1000ms')")
opt=parser.parse_args()

subject_id = opt.sub 
bids_root = opt.bids_root
visit_id = opt.visit_id
duration = opt.duration
SPI_subset = opt.SPI_subset

# Get the base name for SPI_subset file
SPI_subset_base = op.basename(SPI_subset).replace(".yaml", "")

# Time series output path for this subject
time_series_path = op.join(bids_root, "derivatives", "MEG_time_series")
output_feature_path = op.join(bids_root, "derivatives", "time_series_features/averaged_epochs")

# Define ROI lookup table
ROI_lookup = {"proc-0": "Category_Selective",
                "proc-1": "IPS",
                "proc-2": "Prefrontal_Cortex",
                "proc-3": "V1_V2"}
region_names = list(ROI_lookup.values())
    
if op.isfile(f"{output_feature_path}/sub-{subject_id}_ses-{visit_id}_all_pyspi_{SPI_subset_base}_results_{duration}.csv"):
    print(f"{SPI_subset_base} SPI results for sub-{subject_id} already exist. Skipping.")
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

def run_pyspi_for_df(subject_id, df, calc):
        # Make deepcopy of calc 
        calc_copy = deepcopy(calc)

        # Pivot so that the columns are meta_ROI and the rows are data
        df_wide = (df.filter(items=['times'] + region_names)
                        .melt(id_vars='times', var_name='meta_ROI', value_name='data')
                        .reset_index()
                        .pivot(index='meta_ROI', columns='times', values='data'))
                
        # Print first 5 rows
        print("First 5 rows of the wide dataframe:")
        print(df_wide.head())

        # Convert to numpy array
        TS_array = df_wide.to_numpy()

        # Load data 
        calc_copy.load_dataset(TS_array)
        calc_copy.compute()

        SPI_res = deepcopy(calc_copy.table)

        # Iterate over each SPI
        SPI_res.columns = SPI_res.columns.to_flat_index()

        SPI_res = SPI_res.rename(columns='__'.join).assign(meta_ROI_from = lambda x: x.index)
        SPI_res_long = SPI_res.melt(id_vars='meta_ROI_from', var_name='SPI__meta_ROI_to', value_name='value')

        SPI_res_long["SPI"] = SPI_res_long["SPI__meta_ROI_to"].str.split("__").str[0]
        SPI_res_long["meta_ROI_to"] = SPI_res_long["SPI__meta_ROI_to"].str.split("__").str[1]

        SPI_res_long = (SPI_res_long
                        .drop(columns='SPI__meta_ROI_to')
                        .query('meta_ROI_from != meta_ROI_to')
                        .assign(meta_ROI_from = lambda x: x['meta_ROI_from'].map(ROI_lookup),
                                meta_ROI_to = lambda x: x['meta_ROI_to'].map(ROI_lookup))
                        .filter(items=['SPI', 'meta_ROI_from', 'meta_ROI_to', 'value'])
                        .assign(stimulus_type = df['stimulus_type'].unique()[0],
                                relevance_type = df['relevance_type'].unique()[0],
                                duration = df['duration'].unique()[0],
                                stimulus_presentation = df['stimulus'].unique()[0],
                                subject_ID = subject_id)
        )

        return SPI_res_long
# Initialise an empty list for the results
pyspi_res_list = []

# Initialise a base calculator
if SPI_subset == "fast":
    calc = Calculator(subset='fast')
else:
    calc = Calculator(configfile=SPI_subset)


# Run for data
for dataframe in sample_TS_data_list:
    dataframe_pyspi = run_pyspi_for_df(subject_id, dataframe, calc)
    pyspi_res_list.append(dataframe_pyspi)

# Concatenate the results and save to a feather file
all_pyspi_res = pd.concat(pyspi_res_list).reset_index() 
all_pyspi_res.to_csv(f"{output_feature_path}/sub-{subject_id}_ses-{visit_id}_all_pyspi_{SPI_subset_base}_results_{duration}ms.csv", index=False)