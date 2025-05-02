"""
=================
S00. BEM (and coregistration)
=================

Perform the automated coregistration:

Step 1 - Visualize Freesurfer parcellation
Step 2 - MNE-python scalp surface reconstruction
Step 3 - Boundary Element Model (BEM) reconstruction
Step 4 - Get Boundary Element Model (BEM)
(Step 5 - Coregistration)

@author: Oscar Ferrante oscfer88@gmail.com

"""

import os
import os.path as op
# import numpy as np
import argparse

import mne
from mne import bem as mnebem


parser=argparse.ArgumentParser()
parser.add_argument('--sub',
                    type=str,
                    default='SA101',
                    help='site_id + subject_id (e.g. "SA101")')
parser.add_argument('--visit',
                    type=str,
                    default='V1',
                    help='visit_id (e.g. "V1")')
parser.add_argument('--bids_root',
                    type=str,
                    default='/mnt/beegfs/XNAT/COGITATE/MEG/phase_2/processed/bids',
                    help='Path to the BIDS root directory')
parser.add_argument('--fs_home',
                    type=str,
                    default='/mnt/beegfs/XNAT/COGITATE/MEG/phase_2/processed/bids/derivatives/fs',
                    help='Path to the FreeSurfer directory')
parser.add_argument('--subjects_dir',
                    type=str,
                    default='/mnt/beegfs/XNAT/COGITATE/MEG/phase_2/processed/bids/derivatives/fs',
                    help='Path to the FreeSurfer directory')
opt=parser.parse_args()


# Set params
subject = "sub-"+opt.sub
visit = opt.visit
subjects_dir = opt.subjects_dir
FREESURFER_HOME = opt.fs_home

if visit in [1, "1", "01"]:
    fname_raw = op.join(opt.bids_root, subject, "ses-"+visit, "meg", subject+"_ses-1_task-dur_run-01_meg.fif")
elif visit == "V2":
    fname_raw = op.join(opt.bids_root, subject, "ses-"+visit, "meg", subject+"_ses-2_task-vg_run-01_meg.fif")  #TODO: to be tested
coreg_deriv_root = op.join(opt.bids_root, "derivatives", "coreg")
if not op.exists(coreg_deriv_root):
    os.makedirs(coreg_deriv_root, exist_ok=True)
coreg_figure_root =  op.join(coreg_deriv_root,
                            f"sub-{opt.sub}",f"ses-{visit}","meg",
                            "figures")
if not op.exists(coreg_figure_root):
    os.makedirs(coreg_figure_root)

# Step 1 - Boundary Element Model (BEM) reconstruction
def make_bem():
    '''
    Boundary Element Model (BEM)
    
    To create the BEM, either use this function or run the following command
    in a terminal (requires FreeSurfer):
    > mne watershed_bem --overwrite --subject ${file}
    
    '''
    mnebem.make_watershed_bem(subject, 
                               subjects_dir=subjects_dir, 
                               overwrite=True, 
                               verbose=True)
    
    
# Step 2 - Get Boundary Element Model (BEM) solution
def get_bem():
    '''
    Make Boundary Element Model (BEM) solution
    
    Computing the BEM surfaces requires FreeSurfer and is done using the 
    following command:
    > mne watershed_bem --overwrite --subject SA101
    
    Once the BEM surfaces are read, create the BEM model
    
    '''
    # Create BEM model
    conductivity = (0.3,)  # for single layer
    # conductivity = (0.3, 0.006, 0.3)  # for three layers
    model = mne.make_bem_model(subject,
                               ico=4,
                               conductivity=conductivity,
                               subjects_dir=subjects_dir)
    
    # Finally, the BEM solution is derived from the BEM model
    bem = mne.make_bem_solution(model)
    
    # Save data
    fname_bem = op.join(subjects_dir, subject, subject+"_ses-"+visit+"_bem-sol.fif")
    mne.write_bem_solution(fname_bem,
                           bem,
                           overwrite=True)
    # Visualize the BEM
    fig = mne.viz.plot_bem(subject=subject,
                           subjects_dir=subjects_dir,
                           #brain_surfaces='white',
                           orientation='coronal')
    
    # Save figure
    fname_figure = op.join(subjects_dir, subject, "bem-sol.png")
    fig.savefig(fname_figure)
    
    return bem


if __name__ == "__main__":
    print("Getting BEM now")
    make_bem()
    bem = get_bem()
    # coreg()
    
