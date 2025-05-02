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

# Step 1 - Freesurfer recontruction (only on Linux/MACos)
def viz_fs_recon():
    '''
    Freesurfer recontruction (only on Linux/MACos)
    
    Run the following command in a terminal:
    > recon-all -i SA101.nii -s SA101 -all
    For more info, go to https://surfer.nmr.mgh.harvard.edu/fswiki/recon-all/
    
    To convert DICOM to NIFTI, use MRIcron
    
    '''
    # Visualize reconstruction:
    Brain = mne.viz.get_brain_class()
    brain = Brain(subject, 
                  hemi='lh', 
                  surf='pial',
                  subjects_dir=subjects_dir, 
                  size=(800, 600))
    brain.add_annotation('aparc', borders=False)  #aparc.a2009s

    # Save figure
    fname_figure = op.join(subjects_dir, "fs_aparc.png")
    brain.save_image(fname_figure)
    
    
# Step 2 - Scalp surface reconstruction
def make_scalp_surf():
    '''
    Scalp surface reconstruction
    
    Either use this function ot run the following commands in a terminal:
    > mne make_scalp_surfaces --overwrite --subject SA101 --force
    
    
    '''
    mnebem.make_scalp_surfaces(subject, 
                                subjects_dir=subjects_dir, 
                                force=True, 
                                overwrite=True, 
                                verbose=True)

    

if __name__ == "__main__":
    # viz_fs_recon()  #TODO: 3d plots don't work on the HPC
    make_scalp_surf()
    # coreg()
    
