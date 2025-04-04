import os
import os.path as op
import numpy as np
import matplotlib.pyplot as plt
import argparse
import itertools
import json
import pandas as pd
import pickle
from joblib import Parallel, delayed

import mne
import mne_bids
from mne.minimum_norm import apply_inverse,apply_inverse_epochs

parser=argparse.ArgumentParser()
parser.add_argument('--sub',
                    type=str,
                    default='CB040',
                    help='site_id + subject_id (e.g. "CB040")')
parser.add_argument('--bids_root',
                    type=str,
                    default='/project/hctsa/annie/data/Cogitate_MEG/',
                    help='Path to the BIDS root directory')
parser.add_argument('--regions',
                    type=str,
                    default='all',
                    help='config file with region names, or "all"')
parser.add_argument('--n_jobs',
                    type=int,
                    default='all',
                    help='Number of concurrent processing jobs')
opt=parser.parse_args()


# Set params
visit_id = "1" # Using the first visit for this project
sfreq = 100 # Setting sampling frequency to 100Hz

subject_id = opt.sub
regions = opt.regions
bids_root = opt.bids_root
n_jobs = opt.n_jobs

debug = False

factor = ['Category', 'Task_relevance', "Duration"]
# conditions = [['face', 'object', 'letter', 'false'],
#               ['Relevant target', 'Relevant non-target', 'Irrelevant'],
#               ['500ms', '1000ms', '1500ms']]
conditions = [['face', 'object', 'letter', 'false'],
              ['Relevant target', 'Relevant non-target', 'Irrelevant'],
              ['1000ms']]

# Helper function to create a dictionary of ROI labels depending on the type of region subset requested
def compute_ROI_labels(labels_atlas, regions):
    # Create dictionary to store labels and vertices
    labels_dict = {}
    if regions == "all":
        for label in labels_atlas: 
            label_name = label.name
            labels_dict[label_name] =  np.sum([label])
    else:
        # Read GNW and IIT ROI list
        f = open(regions)
        regions_dict = json.load(f)

        # Iterate over regions in the config file
        for region_name in regions_dict.keys():
            print(f"Now processing {region_name} with regions:")
            # Iterate over individual ROIs listed under region_name

            # If there is only one ROI listed, we can directly add the vertices from that ROI
            if len(regions_dict[region_name]) == 1:
                roi_name = regions_dict[region_name][0]

                # Find the label object that matches the ROI name
                labels_dict[region_name] = np.sum([l for l in labels_atlas if roi_name == l.name])

            # If there are multiple ROIs listed, we need to sum the vertices from each ROI into one meta-ROI
            else:
                region_dict = {}
                for roi_name in regions_dict[region_name]:
                    print(roi_name)
                    # Find the label object that matches the ROI name
                    region_dict[roi_name] = np.sum([l for l in labels_atlas if roi_name == l.name])
                region_label = np.sum([region_dict[roi_name] for roi_name in region_dict.keys()])
                labels_dict[region_name] = region_label

    return labels_dict

# Helper function to compute covariance matrices and inverse solution 
def fit_cov_and_inverse(subject_id, visit_id, factor, conditions, bids_root, downsample=True, tmin=-0.5, tmax=1.99):
    # Set directory paths
    prep_deriv_root = op.join(bids_root, "derivatives", "preprocessing")
    fwd_deriv_root = op.join(bids_root, "derivatives", "forward")
    source_deriv_root = op.join(bids_root, "derivatives", "source_dur_ERF")

    if not op.exists(source_deriv_root):
        os.makedirs(source_deriv_root, exist_ok=True)

    source_figure_root =  op.join(source_deriv_root,
                                f"sub-{subject_id}",f"ses-{visit_id}","meg",
                                "figures")
    if not op.exists(source_figure_root):
        os.makedirs(source_figure_root)

    # Set task
    bids_task = 'dur'
    
    # Read epoched data
    bids_path_epo = mne_bids.BIDSPath(
        root=prep_deriv_root, 
        subject=subject_id,  
        datatype='meg',  
        task=bids_task,
        session=visit_id, 
        suffix='epo',
        extension='.fif',
        check=False)
    
    bids_path_epo_rs = mne_bids.BIDSPath(
        root=prep_deriv_root, 
        subject=subject_id,  
        datatype='meg',  
        task=bids_task,
        session=visit_id, 
        suffix='epo_rs',
        extension='.fif',
        check=False)
    
    print("Loading epochs data")
    epochs_all = mne.read_epochs(bids_path_epo.fpath, preload=True)

    if not downsample:
        epochs_final = epochs_all

    # If downsampling is requested
    else:
        print("Applying downsampling")
        if os.path.exists(bids_path_epo_rs.fpath):
            epochs_rs = mne.read_epochs(bids_path_epo_rs.fpath,
                                    preload=True)
        else:
            epochs_all = mne.read_epochs(bids_path_epo.fpath,
                                    preload=True)
            resample_epochs(epochs_all, sfreq, bids_path_epo_rs, tmin=tmin, tmax=tmax)
            epochs_rs = mne.read_epochs(bids_path_epo_rs.fpath, preload=True)
        epochs_final = epochs_rs

    # Run baseline correction
    print("Running baseline correction")
    b_tmin = tmin
    b_tmax = 0.
    baseline = (b_tmin, b_tmax)
    epochs_final.apply_baseline(baseline=baseline)

    # Compute rank
    print("Computing the rank")
    if os.path.isfile(f"{fwd_deriv_root}/sub-{subject_id}_ses-{visit_id}_task-{bids_task}_rank.pkl"):
        with open(f"{fwd_deriv_root}/sub-{subject_id}_ses-{visit_id}_task-{bids_task}_rank.pkl", 'rb') as f:
            rank = pickle.load(f)
    else: 
        rank = mne.compute_rank(epochs_final, 
                                tol=1e-6, 
                                tol_kind='relative')
        with open(f"{fwd_deriv_root}/sub-{subject_id}_ses-{visit_id}_task-{bids_task}_rank.pkl", 'wb') as f:
            pickle.dump(rank, f)

    # Read forward model
    print("Reading forward model")
    bids_path_fwd = bids_path_epo.copy().update(
            root=fwd_deriv_root,
            task=bids_task,
            suffix="surface_fwd",
            extension='.fif',
            check=False)
    fwd = mne.read_forward_solution(bids_path_fwd.fpath)

    # Compute covariance matrices
    print("Computing covariance matrices")
    if os.path.isfile(f"{fwd_deriv_root}/sub-{subject_id}_ses-{visit_id}_task-{bids_task}_common_cov.pkl"): 
        with open(f"{fwd_deriv_root}/sub-{subject_id}_ses-{visit_id}_task-{bids_task}_common_cov.pkl", 'rb') as f:
            common_cov = pickle.load(f)
    else:
        base_cov = mne.compute_covariance(epochs_final, 
                                        tmin=-0.5, 
                                        tmax=0, 
                                        n_jobs=n_jobs,
                                        method='empirical', 
                                        rank=rank)

        active_cov = mne.compute_covariance(epochs_final, 
                                        tmin=0,
                                        tmax=None,
                                        n_jobs=n_jobs,
                                        method='empirical', 
                                        rank=rank)
        common_cov = base_cov + active_cov

        with open(f"{fwd_deriv_root}/sub-{subject_id}_ses-{visit_id}_task-{bids_task}_common_cov.pkl", 'wb') as f:
            pickle.dump(common_cov, f)

    # Make inverse operator
    print("Computing inverse operator")
    if os.path.isfile(f"{fwd_deriv_root}/sub-{subject_id}_ses-{visit_id}_task-{bids_task}_inverse_operator.pkl"):
        with open(f"{fwd_deriv_root}/sub-{subject_id}_ses-{visit_id}_task-{bids_task}_inverse_operator.pkl", 'rb') as f:
            inverse_operator = pickle.load(f)
    else:
        inverse_operator = mne.minimum_norm.make_inverse_operator(
            epochs_final.info,
            fwd, 
            common_cov,
            loose=.2,
            depth=.8,
            fixed=False,
            rank=rank,
            use_cps=True)
        
        with open(f"{fwd_deriv_root}/sub-{subject_id}_ses-{visit_id}_task-{bids_task}_inverse_operator.pkl", 'wb') as f:
            pickle.dump(inverse_operator, f)

    # Find all combinations between variables' levels
    if len(factor) == 1:
        cond_combs = list(itertools.product(conditions[0]))
    if len(factor) == 2:
        cond_combs = list(itertools.product(conditions[0],
                                            conditions[1]))
    if len(factor) == 3:
        cond_combs = list(itertools.product(conditions[0],
                                            conditions[1],
                                            conditions[2]))
        
    print("Done finding final epochs and inverse operator.")
        
    return epochs_final, inverse_operator, cond_combs
# Helper function to process condition combination 
def cond_comb_helper_process_by_epoch(cond_comb, epochs_final, inverse_operator, labels_dict, subject_time_series_output_path):
    print("\nAnalyzing %s: %s" % (factor, cond_comb))

    # Take subset of epochs corresponding to this condition combination
    cond_epochs = epochs_final['%s == "%s" and %s == "%s" and %s == "%s"' % (
        factor[0], cond_comb[0], 
        factor[1], cond_comb[1], 
        factor[2], cond_comb[2])]
    fname_base = f"{cond_comb[0]}_{cond_comb[1]}_{cond_comb[2]}".replace(" ","-")

    # Compute inverse solution for each epoch
    snr = 3.0
    lambda2 = 1.0 / snr ** 2
    stcs = apply_inverse_epochs(cond_epochs, inverse_operator,
                                lambda2=lambda2, verbose=False,
                                method="dSPM", pick_ori="normal")

    # Extract time course for each stc
    for i in range(len(stcs)):

        # Find epoch number
        epoch_number = i+1
        print(f"Extracting epoch number {epoch_number}")

        # Find stc
        stc = stcs[i]

        # Loop over labels        
        for label_name, label in labels_dict.items():

            # Select data in label
            stc_in = stc.in_label(label)

            # Extract time course data, averaged across channels within ROI
            times = stc_in.times
            data = stc_in.data.mean(axis=0)

            # Concatenate into dataframe
            epoch_df = pd.DataFrame({
                'epoch_number': epoch_number,
                'stimulus_type': cond_comb[0], 
                'relevance_type': cond_comb[1],
                'duration': cond_comb[2],
                'times': times,
                'meta_ROI': label_name,
                'data': data})
            
            # Write this epoch to a CSV file
            output_CSV_file = op.join(subject_time_series_output_path, f"{fname_base}_epoch{epoch_number}_{label_name}.csv")
            epoch_df.to_csv(output_CSV_file, index=False)
                
# Extract all epoch time series
def extract_all_epoch_TS(subject_id, visit_id, regions, factor, conditions):

    fs_deriv_root = op.join(bids_root, "derivatives", "fs")
    rois_deriv_root = op.join(bids_root, "derivatives", "roilabel")
    prep_deriv_root = op.join(bids_root, "derivatives", "preprocessing")

    time_series_output_path = op.join(bids_root, "derivatives", "MEG_time_series")
    if not op.exists(time_series_output_path):
        os.makedirs(time_series_output_path, exist_ok=True)

    # Time series output path for this subject
    subject_time_series_output_path = op.join(time_series_output_path, f"sub-{subject_id}", f"ses-{visit_id}", "meg")
    if not op.exists(subject_time_series_output_path):
        os.makedirs(subject_time_series_output_path, exist_ok=True)
        
    # Set task
    bids_task = 'dur'

    # Read epoched data
    bids_path_epo = mne_bids.BIDSPath(
        root=prep_deriv_root, 
        subject=subject_id,  
        datatype='meg',  
        task=bids_task,
        session=visit_id, 
        suffix='epo',
        extension='.fif',
        check=False)
 
    # Use subject-transferred Glasser atlas to compute dictionary of labels
    labels_atlas = mne.read_labels_from_annot(
        "sub-"+subject_id, 
        parc=f'Schaefer100_7Networks_in_sub-{subject_id}',
        subjects_dir=fs_deriv_root)
    labels_dict = compute_ROI_labels(labels_atlas, regions)

    # Save label names
    bids_path_label_names = bids_path_epo.copy().update(
                    root=time_series_output_path,
                    suffix="desc-labels",
                    extension='.txt',
                    check=False)

    # Find epochs_rs, inverse_operator, cond_combs
    print("Now finding inverse operator")
    epochs_final, inverse_operator, cond_combs = fit_cov_and_inverse(subject_id, visit_id, factor, conditions, bids_root, downsample=False)

    # Loop over conditions of interest
    print("Now looping over task conditions")
    Parallel(n_jobs=int(n_jobs))(delayed(cond_comb_helper_process_by_epoch)(cond_comb=cond_comb, 
                                                                epochs_final=epochs_final, 
                                                                inverse_operator=inverse_operator, 
                                                                labels_dict=labels_dict, 
                                                                subject_time_series_output_path=subject_time_series_output_path)
                                                for cond_comb in cond_combs)
    
    if not os.path.isfile(bids_path_label_names):
        with open(bids_path_label_names.fpath, "w") as output:
            output.write(str(list(labels_dict.keys())))

if __name__ == '__main__':
    extract_all_epoch_TS(subject_id, visit_id, regions, factor, conditions)