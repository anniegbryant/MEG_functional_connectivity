import pandas as pd
import numpy as np
from pyspi.calculator import Calculator
from pyspi.data import Data
import os.path as op
from copy import deepcopy
import argparse
from joblib import Parallel, delayed
from scipy.signal import detrend

simulated_TS_dir = 'simulated_data/'
output_barycenter_dir = 'barycenter_results/'
SPI_subset = '../barycenter_robustness/barycenter_sq.yaml'
SPI_subset_euclidean = '../barycenter_robustness/barycenter_sq_euclidean.yaml'
N_sims = 1000

# Get the base name for SPI_subset file
SPI_subset_base = op.basename(SPI_subset).replace(".yaml", "")
SPI_subset_euclidean_base = op.basename(SPI_subset_euclidean).replace(".yaml", "")



# Define ROI lookup tables
GNWT_region_lookup = {"proc-0": "Category_Selective",
                      "proc-1": "Prefrontal_Cortex"}

IIT_region_lookup = {"proc-0": "Category_Selective",
                     "proc-1": "V1_V2"}

region_lookup_dict = {"GNWT_stim_on": GNWT_region_lookup,
                        "GNWT_stim_off": GNWT_region_lookup,
                        "IIT_stim_on": IIT_region_lookup,
                        "IIT_stim_off": IIT_region_lookup}

def run_pyspi_for_arr(pyspi_Data, calc, region_lookup):

    # Make deepcopy of calc
    calc_copy = deepcopy(calc)

    # Load data 
    calc_copy.load_dataset(pyspi_Data)
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

    
def process_for_sim_array(input_3d_array, array_name, output_barycenter_dir, SPI_subset, SPI_subset_base, region_lookup_dict, measurement_noise):
    if op.isfile(f"{output_barycenter_dir}/{array_name}_sims_all_pyspi_{SPI_subset_base}_results_noise_{measurement_noise}.csv"):
        print(f"{SPI_subset_base} SPI results for {array_name} already exist. Skipping.")
        return() 
    
    # Get the region lookup
    region_lookup = region_lookup_dict[array_name]
    
    # Make a copy of calc and compute
    base_calc = Calculator(configfile=SPI_subset)

    # Find size of third dimension in input_3d_array
    num_sims_in_array = input_3d_array.shape[2]

    # Create a list to store the results
    all_sim_pyspi_results = []

    # Iterate over each simulation
    for sim_num in range(num_sims_in_array):
        # Get the time series for this simulation
        time_series = input_3d_array[:,:,sim_num]

        # Create a pyspi Data object, no detrending or normalisation
        time_series_Data = Data(time_series, normalise=False)

        # Get the results
        this_sim_results = (run_pyspi_for_arr(pyspi_Data=time_series_Data, calc=base_calc, region_lookup=region_lookup)
                            .assign(Data_Type = 'Raw',
                                    Noise = measurement_noise,
                                    sim_num = sim_num,
                                    sim_context = array_name)
                            )

        # Add the results to the list
        all_sim_pyspi_results.append(this_sim_results)

        # Also take absolute value 
        time_series_abs = np.abs(time_series)
        time_series_abs_Data = Data(time_series_abs, normalise=False)

        # Get the results
        this_sim_results_abs = (run_pyspi_for_arr(pyspi_Data=time_series_abs_Data, calc=base_calc, region_lookup=region_lookup)
                                .assign(Data_Type = 'Abs',
                                        Noise = measurement_noise,
                                        sim_num = sim_num,
                                        sim_context = array_name)
                                )
        
        # Add the results to the list
        all_sim_pyspi_results.append(this_sim_results_abs)

    # Concatenate the results and save
    all_sim_pyspi_results_df = pd.concat(all_sim_pyspi_results)
    all_sim_pyspi_results_df.to_csv(f"{output_barycenter_dir}/{array_name}_sims_all_pyspi_{SPI_subset_base}_results_noise_{measurement_noise}.csv", index=False)
    

n_jobs=4
for measurement_noise in [0.5, 0.6, 0.7, 0.8, 0.9, 1]:
    time_series_GNWT_stim_on = np.load(f'{simulated_TS_dir}/GNWT_stim_on_{N_sims}_sims_noise_{measurement_noise}.npy')
    time_series_GNWT_stim_off = np.load(f'{simulated_TS_dir}/GNWT_stim_off_{N_sims}_sims_noise_{measurement_noise}.npy')
    time_series_IIT_stim_on = np.load(f'{simulated_TS_dir}/IIT_stim_on_{N_sims}_sims_noise_{measurement_noise}.npy')
    time_series_IIT_stim_off = np.load(f'{simulated_TS_dir}/IIT_stim_off_{N_sims}_sims_noise_{measurement_noise}.npy')

    # Define array name dictionary
    noise_array_name_dict = {"GNWT_stim_on": time_series_GNWT_stim_on,
                             "GNWT_stim_off": time_series_GNWT_stim_off,
                             "IIT_stim_on": time_series_IIT_stim_on,
                             "IIT_stim_off": time_series_IIT_stim_off}

    Parallel(n_jobs=int(n_jobs))(delayed(process_for_sim_array)(input_3d_array, 
                                                                array_name, 
                                                                output_barycenter_dir, 
                                                                SPI_subset_euclidean, 
                                                                SPI_subset_euclidean_base, 
                                                                region_lookup_dict,
                                                                measurement_noise)
                        for array_name, input_3d_array in noise_array_name_dict.items()
                        )

# for measurement_noise in [1]:
#     time_series_GNWT_stim_on = np.load(f'{simulated_TS_dir}/GNWT_stim_on_{N_sims}_sims_noise_{measurement_noise}.npy')
#     time_series_GNWT_stim_off = np.load(f'{simulated_TS_dir}/GNWT_stim_off_{N_sims}_sims_noise_{measurement_noise}.npy')
#     time_series_IIT_stim_on = np.load(f'{simulated_TS_dir}/IIT_stim_on_{N_sims}_sims_noise_{measurement_noise}.npy')
#     time_series_IIT_stim_off = np.load(f'{simulated_TS_dir}/IIT_stim_off_{N_sims}_sims_noise_{measurement_noise}.npy')

#     # Define array name dictionary
#     noise_array_name_dict = {"GNWT_stim_on": time_series_GNWT_stim_on,
#                         "GNWT_stim_off": time_series_GNWT_stim_off,
#                         "IIT_stim_on": time_series_IIT_stim_on,
#                         "IIT_stim_off": time_series_IIT_stim_off}
    
#     Parallel(n_jobs=int(n_jobs))(delayed(process_for_sim_array)(input_3d_array, 
#                                                                 array_name, 
#                                                                 output_barycenter_dir, 
#                                                                 SPI_subset, 
#                                                                 SPI_subset_base, 
#                                                                 region_lookup_dict,
#                                                                 1)
#                         for array_name, input_3d_array in noise_array_name_dict.items()
#                         )