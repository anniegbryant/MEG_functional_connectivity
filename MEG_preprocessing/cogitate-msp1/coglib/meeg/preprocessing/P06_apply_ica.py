"""
===============
06. Apply ICA
===============

This relies on the ICAs computed in P05-run_ica.py
    
@author: Oscar Ferrante oscfer88@gmail.com
    
"""

import os.path as op
import os
import matplotlib.pyplot as plt
import shutil
import json

from fpdf import FPDF
from mne.preprocessing import read_ica
import mne_bids

import sys
sys.path.insert(1, op.dirname(op.dirname(os.path.abspath(__file__))))

def apply_ica(subject_id, visit_id, bids_root, record="run", has_eeg=False):
    
    # Prepare PDF report
    pdf = FPDF(orientation="P", unit="mm", format="A4")
    
    # Set path to preprocessing derivatives
    prep_deriv_root = op.join(bids_root, "derivatives", "preprocessing")
    prep_figure_root =  op.join(prep_deriv_root,
                                f"sub-{subject_id}",f"ses-{visit_id}","meg",
                                "figures")
    prep_report_root =  op.join(prep_deriv_root,
                                f"sub-{subject_id}",f"ses-{visit_id}","meg",
                                "reports")
    prep_code_root = op.join(prep_deriv_root,
                             f"sub-{subject_id}",f"ses-{visit_id}","meg",
                             "codes")
    
    # Read what component to reject from the JSON file
    if os.path.isfile(op.join(prep_deriv_root, 'P05_rej_comp.json')):
        with open(op.join(prep_deriv_root, 'P05_rej_comp.json'), 'r') as openfile:
            rej_comp_json = json.load(openfile)
        
        meg_ica_eog = rej_comp_json[subject_id][visit_id].get('meg_ica_eog')
        meg_ica_ecg = rej_comp_json[subject_id][visit_id].get('meg_ica_ecg')
        eeg_ica_eog = rej_comp_json[subject_id][visit_id].get('eeg_ica_eog')
        eeg_ica_ecg = rej_comp_json[subject_id][visit_id].get('eeg_ica_ecg')
    else:
        meg_ica_eog = []
        meg_ica_ecg = []
        eeg_ica_eog = []
        eeg_ica_ecg = []
        
    if meg_ica_eog + meg_ica_ecg != []:
        # Read ICA mixing matrices
        bids_path_meg_ica = mne_bids.BIDSPath(
            root=prep_deriv_root, 
            subject=subject_id,  
            datatype='meg',  
            session=visit_id, 
            suffix='meg_ica',
            extension='.fif',
            check=False)
        
        ica_meg = read_ica(bids_path_meg_ica.fpath)
        
        # Select EOG- and ECG-related components for exclusion
        ica_meg.exclude.extend(meg_ica_eog + meg_ica_ecg)
        
    if eeg_ica_eog + eeg_ica_ecg != []:
        # Read ICA mixing matrices
        bids_path_eeg_ica = mne_bids.BIDSPath(
            root=prep_deriv_root, 
            subject=subject_id,  
            datatype='meg',  
            session=visit_id, 
            suffix='eeg_ica',
            extension='.fif',
            check=False)
        
        ica_eeg = read_ica(bids_path_eeg_ica.fpath)
        
        # Select EOG- and ECG-related components for exclusion
        ica_eeg.exclude.extend(eeg_ica_eog + eeg_ica_ecg)
        
    print("Processing subject: %s" % subject_id)
    
    # Loop over runs
    data_path = os.path.join(bids_root,f"sub-{subject_id}",f"ses-{visit_id}","meg")
    for fname in sorted(os.listdir(data_path)):
        if fname.endswith(".json") and record in fname:
            
            # Set run
            if "run" in fname:
                run = fname.split("run-")[1].split("_")[0]
            elif "rest" in fname:
                run = None
            print("  Run: %s" % run)
        
            # Set task
            if 'dur' in fname:
                bids_task = 'dur'
            elif 'vg' in fname:
                bids_task = 'vg'
            elif 'replay' in fname:
                bids_task = 'replay'
            elif "rest" in fname:
                bids_task = "rest"
            else:
                raise ValueError("Error: could not find the task for %s" % fname)
        
            # Set BIDS path
            bids_path_annot = mne_bids.BIDSPath(
                root=prep_deriv_root, 
                subject=subject_id,  
                datatype='meg',  
                task=bids_task,
                run=run,
                session=visit_id, 
                suffix='annot',
                extension='.fif',
                check=False)
            
            # Read raw data
            raw = mne_bids.read_raw_bids(bids_path_annot).load_data()
            
            # Fix EOG001 channel name (required for SA only)
            if 'EOG004' in raw.ch_names:
                raw.rename_channels({'EOG004': 'EOG001'})
            
            # Show original signal
            if has_eeg:
                chs = ['MEG0311', 'MEG0121', 'MEG1211', 'MEG1411', 'EEG001','EEG002', 'EOG001','EOG002']
            else:
                chs = ['MEG0311', 'MEG0121', 'MEG1211', 'MEG1411', 'EOG001','EOG002'] 

            # Filter chs
            chs = [ch for ch in chs if ch in raw.ch_names]
            print(chs)
            
            chan_idxs = [raw.ch_names.index(ch) for ch in chs]
            fig1 = raw.plot(order=chan_idxs,
                           duration=20,
                           start=100)        
            fname_fig1 = op.join(prep_figure_root,
                                '06_%sr%s_ica_raw0.png' % (bids_task,run))
            fig1.savefig(fname_fig1)
            plt.close()
            
            # Add figure to report
            pdf.add_page()
            pdf.set_font('helvetica', 'B', 16)
            pdf.cell(0, 10, fname[:-8])
            pdf.ln(20)
            pdf.set_font('helvetica', 'B', 12)
            pdf.cell(0, 10, 'Timecourse of input data', 'B', ln=1)
            pdf.image(fname_fig1, 0, 45, pdf.epw)
            
            # Remove component from MEG signal
            if meg_ica_eog + meg_ica_ecg != []:
                ica_meg.apply(raw)
            
            # Remove component from EEG signal
            if eeg_ica_eog + eeg_ica_ecg != []:
                ica_eeg.apply(raw)
            
            # Save filtered data
            bids_path_filt = bids_path_annot.copy().update(
                root=prep_deriv_root,
                suffix="filt",
                check=False)
    
            raw.save(bids_path_filt, overwrite=True)
                 
            # Show cleaned signal
            fig_ica = raw.plot(order=chan_idxs,
                                   duration=20,
                                   start=100)        
            fname_fig_ica = op.join(prep_figure_root,
                                    '06_%sr%s_ica_rawICA.png' % (bids_task,run))
            fig_ica.savefig(fname_fig_ica)
            plt.close()
            
            # Add figures to report
            pdf.ln(120)
            pdf.cell(0, 10, 'Timecourse of output data', 'B', ln=1)
            pdf.image(fname_fig_ica, 0, 175, pdf.epw)
        
    # Save code
    shutil.copy(__file__, prep_code_root)
    
    # Add note about removed ICs to report
    pdf.add_page()
    pdf.set_font('helvetica', 'B', 16)
    pdf.cell(0, 10, "Excluded indipendent components:")
    pdf.ln(20)
    pdf.set_font('helvetica', 'B', 12)
    pdf.cell(0, 10, 'MEG eog: %s' % meg_ica_eog, 'B', ln=1)
    pdf.cell(0, 10, 'MEG ecg: %s' % meg_ica_ecg, 'B', ln=1)
    pdf.ln(20)
    pdf.cell(0, 10, 'EEG eog: %s' % eeg_ica_eog, 'B', ln=1)
    pdf.cell(0, 10, 'EEG ecg: %s' % eeg_ica_ecg, 'B', ln=1)
    
    # Save report
    if record == "rest":
        pdf.output(op.join(prep_report_root,
                       os.path.basename(__file__) + '-report_rest.pdf'))
    else:
        pdf.output(op.join(prep_report_root,
                       os.path.basename(__file__) + '-report.pdf'))

def input_bool(message):
    value = input(message)
    if value == "True":
        return True
    if value == "False":
        return False


if __name__ == '__main__':
    subject_id = input("Type the subject ID (e.g., SA101)\n>>> ")
    visit_id = input("Type the visit ID (V1 or V2)\n>>> ")
    has_eeg = input_bool("Has this recording EEG data? (True or False)\n>>> ")
    apply_ica(subject_id, visit_id, has_eeg)
    