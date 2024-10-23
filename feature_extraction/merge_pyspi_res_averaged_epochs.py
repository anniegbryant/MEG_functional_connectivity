import pandas as pd
import os.path as op
from glob import glob

visit_id = "1"
duration = "1000ms"
bids_root = "/headnode1/abry4213/data/Cogitate_MEG/"

# Time series output path for this subject
pyspi_path = op.join(bids_root, "derivatives", "time_series_features/averaged_epochs")

# Iterate over the files in the directory
for file in glob(pyspi_path + "/*fast_supplement*"):
    # Get the base name for the file
    subject_ID = op.basename(file).replace(".csv", "").split("_")[0]

    if not op.exists(f"{pyspi_path}/{subject_ID}_ses-{visit_id}_all_pyspi_results_{duration}.csv"):

        # Read in subject's fast results
        subject_fast_results = pd.read_csv(f"{pyspi_path}/{subject_ID}_ses-{visit_id}_all_pyspi_fast_results_{duration}.csv")

        # Read in fast supplement results
        subject_fast_supp_results = pd.read_csv(file)

        # Merge the two dataframes
        all_subject_results = pd.concat([subject_fast_results, subject_fast_supp_results], axis=0).reset_index(drop=True)

        # Save the merged dataframe
        all_subject_results.to_csv(f"{pyspi_path}/{subject_ID}_ses-{visit_id}_all_pyspi_results_{duration}.csv", index=False)