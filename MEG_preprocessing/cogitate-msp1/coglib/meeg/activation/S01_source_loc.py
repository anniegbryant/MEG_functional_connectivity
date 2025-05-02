"""
================
S02. Source localization of frequency-band-specific activity
================

Compute LCMV and DICS beamforming.

@author: Oscar Ferrante oscfer88@gmail.com

"""

import os
import os.path as op
# import numpy as np
# import matplotlib.pyplot as plt
import argparse

import mne
from mne.cov import compute_covariance
# from mne.beamformer import make_lcmv, apply_lcmv_cov, make_dics, apply_dics_csd
# from mne.time_frequency import csd_multitaper
import mne_bids

import sys
sys.path.insert(1, op.dirname(op.dirname(os.path.abspath(__file__))))


parser=argparse.ArgumentParser()
parser.add_argument('--sub',
                    type=str,
                    default='SA113',
                    help='site_id + subject_id (e.g. "SA101")')
parser.add_argument('--visit',
                    type=str,
                    default='V2',
                    help='visit_id (e.g. "V1")')
parser.add_argument('--method',
                    type=str,
                    default='dspm',
                    help='method used for the inverse solution ("lcmv", "dics", "dspm")')
parser.add_argument('--bids_root',
                    type=str,
                    default='/data/MEG_data/BIDS',
                    help='BIDS root directory')

opt=parser.parse_args()


# Set params
subject_id = opt.sub
visit_id = opt.visit
inv_method = opt.method
bids_root = opt.bids_root

debug = True
use_rs_noise = True

def run_sourcerecon(subject_id, visit_id):
    # Set path to preprocessing derivatives and create the related folders
    prep_deriv_root = op.join(bids_root, "derivatives", "preprocessing")
    fwd_deriv_root = op.join(bids_root, "derivatives", "forward")
    fs_deriv_root = op.join(bids_root, "derivatives", "fs")
    
    stfr_deriv_root = op.join(bids_root, "derivatives", "source_loc")
    if not op.exists(stfr_deriv_root):
        os.makedirs(stfr_deriv_root)
    stfr_figure_root =  op.join(stfr_deriv_root,
                                f"sub-{subject_id}",f"ses-{visit_id}","meg",
                                "figures")
    if not op.exists(stfr_figure_root):
        os.makedirs(stfr_figure_root)
    
    print("Processing subject: %s" % subject_id)
    
    # Set task
    if visit_id in [1, "1", "01"]:
        bids_task = 'dur'
    elif visit_id in [2, "2", "02"]:
        bids_task = 'vg'
    # elif visit_id == "V2":  #find a better way to set the task in V2
    #     bids_task = 'replay'
    else:
        raise ValueError("Error: could not set the task")
    
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
    
    epochs = mne.read_epochs(
        bids_path_epo.fpath,
        preload=False)
    
    # Read resting-state data
    if visit_id in [2, "2", "02"] and use_rs_noise:
        bids_path_rs = bids_path_epo.copy().update(
            task="rest",
            check=False)
        
        epochs_rs = mne.read_epochs(
            bids_path_rs.fpath,
            preload=False)
    
    # Pick trials
    if visit_id in [1, "1", "01"]:
        epochs = epochs['Task_relevance in ["Relevant non-target", "Irrelevant"]']
    if debug:
        epochs = epochs[0:100]
    
    # Select sensor type
    epochs.load_data().pick('meg')
    if visit_id in [2, "2", "02"] and use_rs_noise:
        epochs_rs.load_data().pick('meg')
    
    # Baseline correction
    baseline_win = (-0.5, 0.)
    active_win = (.0, 1.5)
    if visit_id in [1, "1", "01"] or not use_rs_noise:
        epochs.apply_baseline(baseline=baseline_win)
    
    # Compute rank
    rank = mne.compute_rank(epochs, 
                            tol=1e-6, 
                            tol_kind='relative')
    
    # Read forward model
    if inv_method == 'dspm':
        space = "surface"
    else:
        space = "volume"
        
    if visit_id in [1, "1", "01"]:
        task = "dur"
    elif visit_id in [2, "2", "02"]:
        task = "vg"
    
    bids_path_fwd = bids_path_epo.copy().update(
            root=fwd_deriv_root,
            task=task,
            suffix=space+"_fwd",
            extension=".fif",
            check=False)
    fwd = mne.read_forward_solution(bids_path_fwd.fpath)
    
    # Loop iver frequency bands
    for fr_band in ['alpha', 'beta', 'gamma']:
    
        # Filter data
        if fr_band == "alpha":
            fmin = 8
            fmax = 13
        elif fr_band == "beta":
            fmin = 13
            fmax = 30
        elif fr_band == "gamma":
            fmin = 60
            fmax = 90
        else:
            raise ValueError("Error: 'band' value not valid")
        
        epochs_band = epochs.copy().filter(fmin, fmax)
        if visit_id in [2, "2", "02"] and use_rs_noise:
            epochs_rs_band = epochs_rs.copy().filter(fmin, fmax)
                    
        # Compute covariance matrices
        if visit_id in [1, "1", "01"] or not use_rs_noise:
            noise_cov = compute_covariance(epochs_band, 
                                           tmin=baseline_win[0], 
                                           tmax=baseline_win[1], 
                                           method='empirical', 
                                           rank=rank)
        elif visit_id in [2, "2", "02"]:
            noise_cov = compute_covariance(epochs_rs_band, 
                                           method='empirical', 
                                           rank=rank)
        
        active_cov = compute_covariance(epochs_band, 
                                     tmin=active_win[0], 
                                     tmax=active_win[1],
                                     method='empirical', 
                                     rank=rank)
        common_cov = noise_cov + active_cov
        
        # Make inverse operator
        filters = mne.minimum_norm.make_inverse_operator(
            epochs_band.info,
            fwd, 
            common_cov,
            loose=.2,
            depth=.8,
            fixed=False,
            rank=rank,
            use_cps=True)
        
        for condition in range(1,3):
            
            # Pick condition
            if visit_id in [1, "1", "01"]:
                if condition == 1:
                    epochs_cond = epochs_band['Task_relevance == "Relevant non-target"'].copy()
                    cond_name = "relevant non-target"
                elif condition == 2:
                    epochs_cond = epochs_band['Task_relevance == "Irrelevant"'].copy()
                    cond_name = "irrelevant"
                else:
                    raise ValueError("Condition %s does not exists" % condition)
            elif visit_id in [2, "2", "02"]:
                if condition == 1:
                    epochs_cond = epochs_band.copy()
                    if use_rs_noise:
                        cond_name = "all"
                    else:
                        cond_name = "all_vgbase"
                else:
                    continue
            
            print(f"\n\n\n### Running on task {cond_name} ###\n\n")
                            
            # Compute covariance matrices
            act_cov_cond = compute_covariance(epochs_cond, 
                                              tmin=active_win[0], 
                                              tmax=active_win[1],
                                              method='empirical', 
                                              rank=rank)
            if visit_id in [1, "1", "01"] or not use_rs_noise:
                noise_cov_cond = compute_covariance(epochs_cond, 
                                                    tmin=baseline_win[0], 
                                                    tmax=baseline_win[1],
                                                    method='empirical', 
                                                    rank=rank)
            elif visit_id in [2, "2", "02"]:
                noise_cov_cond = compute_covariance(epochs_rs_band, 
                                                    method='empirical', 
                                                    rank=rank)
            
            # Apply dSPM filter
            stc_act = mne.minimum_norm.apply_inverse_cov(act_cov_cond, 
                                        epochs_cond.info, 
                                        filters,
                                        method='dSPM', 
                                        pick_ori=None,
                                        verbose=True)
            if visit_id in [1, "1", "01"] or not use_rs_noise:
                stc_base = mne.minimum_norm.apply_inverse_cov(noise_cov_cond, 
                                                              epochs_cond.info, 
                                                              filters,
                                                              method='dSPM', 
                                                              pick_ori=None,
                                                              verbose=True)
            elif visit_id in [2, "2", "02"]:
                stc_base = mne.minimum_norm.apply_inverse_cov(noise_cov_cond, 
                                                              epochs_rs_band.info, 
                                                              filters,
                                                              method='dSPM', 
                                                              pick_ori=None,
                                                              verbose=True)

            # Compute baseline correction
            stc_act /= stc_base
            
            # Save source estimates
            bids_path_con = bids_path_epo.copy().update(
                root=stfr_deriv_root,
                suffix=f"stfr_beam-{inv_method}_band-{fr_band}_c-{cond_name}",
                extension=None,
                check=False)
            
            stc_act.save(bids_path_con, overwrite=True)
            
            # Morph to fsaverage  #not needed if morphing the forward solution
            if inv_method in ["lcmv", "dics"]:
                fname_fs_src = fs_deriv_root + '/fsaverage/bem/fsaverage-vol-5-src.fif'
            elif inv_method == "dspm":
                fname_fs_src = fs_deriv_root + '/fsaverage/bem/fsaverage-ico-5-src.fif'
            
            src_fs = mne.read_source_spaces(fname_fs_src)
            
            morph = mne.compute_source_morph(
                fwd['src'], 
                subject_from="sub-"+subject_id, 
                subject_to='fsaverage', 
                src_to=src_fs, 
                subjects_dir=fs_deriv_root,
                verbose=True)
            
            stc_fs = morph.apply(stc_act)
            
            # Save morphed source estimates
            bids_path_sou = bids_path_epo.copy().update(
                root=stfr_deriv_root,
                suffix=f"stfr_beam-{inv_method}_band-{fr_band}_c-{cond_name}_morph",
                extension=None,
                check=False)
            
            stc_fs.save(bids_path_sou, overwrite=True)


if __name__ == '__main__':
    run_sourcerecon(subject_id, visit_id)
    