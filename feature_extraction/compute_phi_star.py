import os
import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt
from glob import glob
from tslearn import barycenters
from scipy.stats import zscore
import jpype
from copy import deepcopy
from pyspi.calculator import Calculator
import os.path as op
from joblib import Parallel, delayed

# Use 4 jobs
num_jobs = 8

# Point to pyspi installation of infodynamics
jarLocation = "/Users/abry4213/github/pyspi/pyspi/lib/jidt/infodynamics.jar"

# Check if a JVM has already been started
# If not, start the JVM (add the "-Xmx" option with say 1024M if you get crashes due to not enough memory space)
if not jpype.isJVMStarted():
    jpype.startJVM(jpype.getDefaultJVMPath(), "-ea", "-Djava.class.path=" + jarLocation)

def run_pyspi_for_df(subject_ID, df, calc, region_names, ROI_lookup):
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
                            subject_ID = subject_ID)
    )

    return SPI_res_long

# Define a function to compute phi star for a subject
def compute_phi_star_for_subj(subject_ID, MEG_time_series_dir, visit_id, duration, base_calc, output_feature_path):

    # Create a copy of the base calculator
    this_calc = deepcopy(base_calc)

    # Define the output file
    output_file = f"{output_feature_path}/sub-{subject_ID}_ses-{visit_id}_all_pyspi_phi_star_results_{duration}ms.csv"
    if op.isfile(output_file):
        print(f"phi-star SPI results for sub-{subject_ID} already exist. Skipping.")
        exit() 

    # Initialise an empty list for the results
    pyspi_res_list = []

    # Load the time series data
    sample_TS_data = pd.read_csv(f"{MEG_time_series_dir}/{subject_ID}_ses-{visit_id}_meg_{duration}ms_all_time_series.csv")
    sample_TS_data['duration'] = sample_TS_data['duration'].str.replace('ms', '').astype(int)
    sample_TS_data['times'] = np.round(sample_TS_data['times']*1000)
    sample_TS_data['times'] = sample_TS_data['times'].astype(int)

    # Filter times >= 0
    sample_TS_data = sample_TS_data.query('times >= 0')

    # Assign stimulus as on if times < duration and off if times >= duration
    sample_TS_data['stimulus'] = np.where(sample_TS_data['times'] < sample_TS_data['duration'], 'on', 'off')

    # Reset the index
    sample_TS_data.reset_index(drop=True, inplace=True)

    # Define ROI lookup table
    ROI_lookup = {"proc-0": "Category_Selective",
                    "proc-1": "IPS",
                    "proc-2": "Prefrontal_Cortex",
                    "proc-3": "V1_V2"}
    region_names = list(ROI_lookup.values())

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
    
    # Run for data
    for dataframe in sample_TS_data_list:
        dataframe_pyspi = run_pyspi_for_df(subject_ID=subject_ID, df=dataframe, calc=this_calc, region_names=region_names, ROI_lookup=ROI_lookup)
        pyspi_res_list.append(dataframe_pyspi)

    # Concatenate the results and save to a feather file
    try:
        all_pyspi_res = pd.concat(pyspi_res_list).reset_index() 
        all_pyspi_res.to_csv(output_file, index=False)
    except Exception as e:
        print(f"Error for {subject_ID}: {e}")

if __name__ == "__main__":
    # Define path for derivatives directory
    deriv_dir = "/Users/abry4213/data/Cogitate_MEG/derivatives"
    MEG_time_series_dir = f"{deriv_dir}/MEG_time_series"

    # Define ROI lookup table
    ROI_lookup = {"proc-0": "Category_Selective",
                    "proc-1": "IPS",
                    "proc-2": "Prefrontal_Cortex",
                    "proc-3": "V1_V2"}
    region_names = list(ROI_lookup.values())

    # Initialise a base calculator
    SPI_subset = "phi_supplement.yaml"
    calc = Calculator(configfile=SPI_subset)

    # Define subject list
    subject_list = pd.read_table("../metadata/subject_list_Cogitate_MEG_with_all_data.txt", header=None)[0].tolist()

    # Iterate over each subject and compute phi star using apply
    # Run in parallel
    Parallel(n_jobs=num_jobs)(delayed(compute_phi_star_for_subj)(subject_ID=f"sub-{subject}", 
                                                                      MEG_time_series_dir=MEG_time_series_dir, 
                                                                      visit_id="1", duration="1000", base_calc=calc, 
                                  output_feature_path=f"{deriv_dir}/time_series_features/averaged_epochs")
                              for subject in subject_list)
    
    # for subject in subject_list:
    #     print(f"Computing phi star for sub-{subject}")
    #     compute_phi_star_for_subj(subject_ID=f"sub-{subject}", MEG_time_series_dir=MEG_time_series_dir, 
    #                               visit_id="1", duration="1000", base_calc=calc, 
    #                               output_feature_path=f"{deriv_dir}/time_series_features/averaged_epochs")
