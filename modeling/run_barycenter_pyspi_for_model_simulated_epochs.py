import pandas as pd
import numpy as np
from pyspi.calculator import Calculator
from pyspi.data import Data
import os.path as op
from copy import deepcopy
import argparse
from joblib import Parallel, delayed
from scipy.signal import detrend

# Define base_repo as one level up from current directory
base_repo = op.abspath(op.join(op.dirname(__file__), op.pardir))

# Input and output directories
simulated_TS_dir = f'{base_repo}/data/model/simulated_data/'
output_barycenter_dir = f'{base_repo}/data/model/barycenter_results/'

# Define the SPI subset files
SPI_subset_euc = f'{base_repo}/functional_connectivity_analysis/barycenter_sq_euclidean.yaml'
SPI_subset = f'{base_repo}/functional_connectivity_analysis/barycenter_sq.yaml'

# We've constructed 1000 simulated for GNWT and IIT in stimulus on and off conditions, respectively
N_sims = 1000

# Use 16 jobs
n_jobs = 16

# Each sim has 1000 timepoints
num_timepoints = 1000

# IIT has 18 parameter settings, GNWT has 22
n_parameters_IIT = 18
n_parameters_GNWT = 22

# We use two ROIs per model
num_ROIs = 2

# Get the base name for SPI_subset file
SPI_subset_base_euc = op.basename(SPI_subset_euc).replace(".yaml", "")
SPI_subset_base = op.basename(SPI_subset).replace(".yaml", "")

# Define ROI lookup tables
GNWT_region_lookup = {"proc-0": "Category_Selective",
                      "proc-1": "Prefrontal_Cortex"}

IIT_region_lookup = {"proc-0": "Category_Selective",
                     "proc-1": "V1_V2"}

region_lookup_dict = {"GNWT_stim_on": GNWT_region_lookup,
                        "GNWT_stim_off": GNWT_region_lookup,
                        "IIT_stim_on": IIT_region_lookup,
                        "IIT_stim_off": IIT_region_lookup}

def run_pyspi_for_arr(pyspi_data, calc, region_lookup):

    # Make deepcopy of calc
    calc_copy = deepcopy(calc)

    # Load data 
    calc_copy.load_dataset(pyspi_data)
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
                    .assign(meta_ROI_from = lambda x: x['meta_ROI_from'].map(region_lookup),
                            meta_ROI_to = lambda x: x['meta_ROI_to'].map(region_lookup))
                    .filter(items=['SPI', 'meta_ROI_from', 'meta_ROI_to', 'value'])
    )

    return SPI_res_long

    
def process_for_sim_array(input_4d_array, array_name, output_barycenter_dir, SPI_subset, SPI_subset_base, region_lookup_dict, measurement_noise=0.5, N_sims=1000, n_parameters=18):
    output_file = f"{output_barycenter_dir}/{array_name}_sims_all_pyspi_{SPI_subset_base}_results_noise_{measurement_noise}_params_{n_parameters}.csv"

    if op.isfile(output_file):
        print(f"{SPI_subset_base} SPI results for {array_name} already exist. Skipping.")
        return() 
    
    # Get the region lookup
    region_lookup = region_lookup_dict[array_name]
    
    # Make a copy of calc and compute
    base_calc = Calculator(configfile=SPI_subset)

    # Create a list to store the results
    all_sim_pyspi_results = []

    # Iterate over each simulation
    for sim_num in range(N_sims):

        # Iterate over each parameter setting 
        for param_number in range(n_parameters):
            # Get the time series for this simulation and parameter setting
            time_series = input_4d_array[:,:,sim_num, param_number]

            # Create a pyspi Data object, no detrending or normalisation
            time_series_Data = Data(time_series, normalise=False)

            # Get the results
            this_sim_results = (run_pyspi_for_arr(pyspi_data=time_series_Data, calc=base_calc, region_lookup=region_lookup)
                                .assign(Data_Type = 'Raw',
                                        Noise = measurement_noise,
                                        sim_num = sim_num,
                                        param_number = param_number,
                                        sim_context = array_name)
                                )

            # Add the results to the list
            all_sim_pyspi_results.append(this_sim_results)

            # Also take absolute value 
            time_series_abs = np.abs(time_series)
            time_series_abs_Data = Data(time_series_abs, normalise=False)

            # Get the results
            this_sim_results_abs = (run_pyspi_for_arr(pyspi_data=time_series_abs_Data, calc=base_calc, region_lookup=region_lookup)
                                    .assign(Data_Type = 'Abs',
                                            Noise = measurement_noise,
                                            sim_num = sim_num,
                                            param_number = param_number,
                                            sim_context = array_name)
                                    )
            
            # Add the results to the list
            all_sim_pyspi_results.append(this_sim_results_abs)

    # Concatenate the results and save
    all_sim_pyspi_results_df = pd.concat(all_sim_pyspi_results)
    all_sim_pyspi_results_df.to_csv(output_file, index=False)
    
# Use 4 jobs with parallel processing
n_jobs=4

########### Part 1: Sweeping across parameter settings at noise = 1.0 ###########
measurement_noise=1.0
time_series_GNWT_stim_on = np.load(f'{simulated_TS_dir}/GNWT_stim_on_{N_sims}_sims_noise_{measurement_noise}_all_parameters.npy')
time_series_GNWT_stim_off = np.load(f'{simulated_TS_dir}/GNWT_stim_off_{N_sims}_sims_noise_{measurement_noise}_all_parameters.npy')
time_series_IIT_stim_on = np.load(f'{simulated_TS_dir}/IIT_stim_on_{N_sims}_sims_noise_{measurement_noise}_all_parameters.npy')
time_series_IIT_stim_off = np.load(f'{simulated_TS_dir}/IIT_stim_off_{N_sims}_sims_noise_{measurement_noise}_all_parameters.npy')

# Reshape for barycenter computation
time_series_GNWT_stim_on = np.reshape(time_series_GNWT_stim_on, (num_ROIs, num_timepoints, N_sims, n_parameters_GNWT))
time_series_GNWT_stim_off = np.reshape(time_series_GNWT_stim_off, (num_ROIs, num_timepoints, N_sims, n_parameters_GNWT))
time_series_IIT_stim_on = np.reshape(time_series_IIT_stim_on, (num_ROIs, num_timepoints, N_sims, n_parameters_IIT))
time_series_IIT_stim_off = np.reshape(time_series_IIT_stim_off, (num_ROIs, num_timepoints, N_sims, n_parameters_IIT))

# Define array name dictionary
noise_array_name_dict = {
    "GNWT_stim_on": (time_series_GNWT_stim_on, n_parameters_GNWT),
    "GNWT_stim_off": (time_series_GNWT_stim_off, n_parameters_GNWT),
    "IIT_stim_on": (time_series_IIT_stim_on, n_parameters_IIT),
    "IIT_stim_off": (time_series_IIT_stim_off, n_parameters_IIT),
}

Parallel(n_jobs=int(n_jobs))(delayed(process_for_sim_array)(input_4d_array, 
                                                            array_name, 
                                                            output_barycenter_dir, 
                                                            SPI_subset_euc, 
                                                            SPI_subset_base_euc, 
                                                            region_lookup_dict,
                                                            measurement_noise,
                                                            n_parameters=n_params,
    )
    for array_name, (input_4d_array, n_params) in noise_array_name_dict.items()
)

########## Part 2: Sweeping across measurement noise levels ###########
for measurement_noise in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
    GNWT_param_number = 15
    IIT_param_number = 13
    n_parameters=1

    time_series_GNWT_stim_on = np.load(f'{simulated_TS_dir}/GNWT_stim_on_{N_sims}_sims_noise_{measurement_noise}_param_{GNWT_param_number}.npy')
    time_series_GNWT_stim_off = np.load(f'{simulated_TS_dir}/GNWT_stim_off_{N_sims}_sims_noise_{measurement_noise}_param_{GNWT_param_number}.npy')
    time_series_IIT_stim_on = np.load(f'{simulated_TS_dir}/IIT_stim_on_{N_sims}_sims_noise_{measurement_noise}_param_{IIT_param_number}.npy')
    time_series_IIT_stim_off = np.load(f'{simulated_TS_dir}/IIT_stim_off_{N_sims}_sims_noise_{measurement_noise}_param_{IIT_param_number}.npy')

    # Reshape for barycenter computation
    time_series_GNWT_stim_on = np.reshape(time_series_GNWT_stim_on, (num_ROIs, num_timepoints, N_sims, n_parameters))
    time_series_GNWT_stim_off = np.reshape(time_series_GNWT_stim_off, (num_ROIs, num_timepoints, N_sims, n_parameters))
    time_series_IIT_stim_on = np.reshape(time_series_IIT_stim_on, (num_ROIs, num_timepoints, N_sims, n_parameters))
    time_series_IIT_stim_off = np.reshape(time_series_IIT_stim_off, (num_ROIs, num_timepoints, N_sims, n_parameters))

    # Define array name dictionary
    noise_array_name_dict = {"GNWT_stim_on": time_series_GNWT_stim_on,
                             "GNWT_stim_off": time_series_GNWT_stim_off,
                             "IIT_stim_on": time_series_IIT_stim_on,
                             "IIT_stim_off": time_series_IIT_stim_off}

    Parallel(n_jobs=int(n_jobs))(delayed(process_for_sim_array)(input_4d_array, 
                                                                array_name, 
                                                                output_barycenter_dir, 
                                                                SPI_subset_euc, 
                                                                SPI_subset_base_euc, 
                                                                region_lookup_dict,
                                                                measurement_noise,
                                                                n_parameters=n_parameters)
                        for array_name, input_4d_array in noise_array_name_dict.items()
                        )

########## Part 3: all barycenter geometries with noise=1.0 ###########

# Define parameters
measurement_noise=1.0
GNWT_param_number = 15
IIT_param_number = 13
n_parameters=1

# Load the time series
time_series_GNWT_stim_on = np.load(f'{simulated_TS_dir}/GNWT_stim_on_{N_sims}_sims_noise_{measurement_noise}_param_{GNWT_param_number}.npy')
time_series_GNWT_stim_off = np.load(f'{simulated_TS_dir}/GNWT_stim_off_{N_sims}_sims_noise_{measurement_noise}_param_{GNWT_param_number}.npy')
time_series_IIT_stim_on = np.load(f'{simulated_TS_dir}/IIT_stim_on_{N_sims}_sims_noise_{measurement_noise}_param_{IIT_param_number}.npy')
time_series_IIT_stim_off = np.load(f'{simulated_TS_dir}/IIT_stim_off_{N_sims}_sims_noise_{measurement_noise}_param_{IIT_param_number}.npy')

# Reshape for barycenter computation
time_series_GNWT_stim_on = np.reshape(time_series_GNWT_stim_on, (num_ROIs, num_timepoints, N_sims, n_parameters))
time_series_GNWT_stim_off = np.reshape(time_series_GNWT_stim_off, (num_ROIs, num_timepoints, N_sims, n_parameters))
time_series_IIT_stim_on = np.reshape(time_series_IIT_stim_on, (num_ROIs, num_timepoints, N_sims, n_parameters))
time_series_IIT_stim_off = np.reshape(time_series_IIT_stim_off, (num_ROIs, num_timepoints, N_sims, n_parameters))

# Define array name dictionary
noise_array_name_dict = {"GNWT_stim_on": time_series_GNWT_stim_on,
                            "GNWT_stim_off": time_series_GNWT_stim_off,
                            "IIT_stim_on": time_series_IIT_stim_on,
                            "IIT_stim_off": time_series_IIT_stim_off}

Parallel(n_jobs=int(n_jobs))(delayed(process_for_sim_array)(input_4d_array, 
                                                            array_name, 
                                                            output_barycenter_dir, 
                                                            SPI_subset, 
                                                            SPI_subset_base, 
                                                            region_lookup_dict,
                                                            measurement_noise,
                                                            n_parameters=n_parameters)
                    for array_name, input_4d_array in noise_array_name_dict.items()
                    )
