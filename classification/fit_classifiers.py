import os
import numpy as np
import pandas as pd
import sys
from sklearn import svm
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedGroupKFold, cross_validate, StratifiedKFold
import numpy as np
from sklearn.linear_model import LogisticRegression
from copy import deepcopy
import itertools
import argparse
from os import path as op
from glob import glob

# add path to classification analysis functions
from mixed_sigmoid_normalisation import MixedSigmoidScaler

parser=argparse.ArgumentParser()
parser.add_argument('--bids_root',
                    type=str,
                    default='/project/hctsa/annie/data/Cogitate_Batch1/MEG_Data/',
                    help='Path to the BIDS root directory')
parser.add_argument('--n_jobs',
                    type=int,
                    default=1,
                    help='Number of concurrent processing jobs')
parser.add_argument('--SPI_directionality_file',
                    type=str,
                    default='/headnode1/abry4213/github/Cogitate_Connectivity_2024/feature_extraction/pyspi_SPI_info.csv',
                    help='CSV file with SPI directionality info')
parser.add_argument('--subject_ID',
                    type=str,
                    default=None,
                    help='Subject for intra-subject classification [optional]')
parser.add_argument('--classification_type',
                    type=str,
                    default='all',
                    help='Whether to perform average and/or individual classification; default is all')
parser.add_argument('--classifier',
                    type=str,
                    default='Linear_SVM',
                    help='Which type of classifier to use')
opt=parser.parse_args()

bids_root = opt.bids_root
n_jobs = opt.n_jobs
subject_ID = opt.subject_ID
SPI_directionality_file = opt.SPI_directionality_file
classification_type = opt.classification_type
classifier = opt.classifier

# Read in SPI directionality info
SPI_directionality_info = pd.read_csv(SPI_directionality_file)

# Load data paths
pyspi_res_path = f"{bids_root}/derivatives/time_series_features"
pyspi_res_path_averaged = f"{pyspi_res_path}/averaged_epochs"
pyspi_res_path_individual = f"{pyspi_res_path}/individual_epochs"

classification_res_path = f"{bids_root}/derivatives/classification_results"
classification_res_path_averaged = f"{classification_res_path}/across_participants"
classification_res_path_individual = f"{classification_res_path}/within_participants"

# Make classification result directories
os.makedirs(classification_res_path_averaged, exist_ok=True)
os.makedirs(classification_res_path_individual, exist_ok=True)

# Define classifier
if classifier == "Linear_SVM":
    model = svm.SVC(C=1, class_weight='balanced', kernel='linear', random_state=127)
else:
    model = LogisticRegression(penalty='l1', C=1, solver='liblinear', class_weight='balanced', random_state=127)

pipe = Pipeline([('scaler', MixedSigmoidScaler(unit_variance=True)), 
                            ('model', model)])

# Define scoring type
scoring = {'accuracy': 'accuracy',
           'balanced_accuracy': 'balanced_accuracy'}

#################################################################################################
# Classification across participants with averaged epochs
#################################################################################################

if classification_type == "averaged":

    # Load in pyspi results
    all_pyspi_res_list = []
    # for pyspi_res_file in os.listdir(pyspi_res_path_averaged):
    for pyspi_res_file in glob(f"{pyspi_res_path_averaged}/*all_pyspi_results_1000ms.csv"):
        pyspi_res = pd.read_csv(pyspi_res_file)
        # Reset index
        pyspi_res.reset_index(inplace=True, drop=True)
        pyspi_res['stimulus_type'] = pyspi_res['stimulus_type'].replace(False, 'false').replace('False', 'false')
        pyspi_res['relevance_type'] = pyspi_res['relevance_type'].replace("Relevant non-target", "Relevant-non-target")
        # Rename stimulus to stimulus_presentation if it is present
        if 'stimulus' in pyspi_res.columns:
            if 'stimulus_presentation' in pyspi_res.columns:
                pyspi_res.drop(columns=['stimulus'], inplace=True)
            else:
                pyspi_res = pyspi_res.rename(columns={'stimulus': 'stimulus_presentation'})

        all_pyspi_res_list.append(pyspi_res)
    all_pyspi_res = pd.concat(all_pyspi_res_list)

    # Define comparisons

    # meta-ROI comparisons
    meta_ROIs = ["Category_Selective", "IPS", "Prefrontal_Cortex", "V1_V2"]
    meta_roi_comparisons = list(itertools.permutations(meta_ROIs, 2))

    # Relevance type comparisons
    relevance_type_comparisons = ["Relevant-non-target", "Irrelevant"]

    # Stimulus presentation comparisons
    stimulus_presentation_comparisons = ["on", "off"]

    # Stimulus type comparisons
    stimulus_types = all_pyspi_res.stimulus_type.unique().tolist()
    stimulus_type_comparisons = list(itertools.combinations(stimulus_types, 2))

    # Also add in face vs. non-face
    stimulus_type_comparisons.append(("face", "non-face"))

    # Comparing between stimulus types
    if not os.path.isfile(f"{classification_res_path_averaged}/comparing_between_stimulus_types_{classifier}_classification_results.csv"):
        # All comparisons list
        comparing_between_stimulus_types_classification_results_list = []

        for relevance_type in relevance_type_comparisons:
            print("Relevance type:" + str(relevance_type))
            for stimulus_presentation in stimulus_presentation_comparisons:
                print("Stimulus presentation:" + str(stimulus_presentation))
                for SPI in all_pyspi_res.SPI.unique():
                    # First, look at each meta-ROI pair separately
                    for meta_roi_comparison in meta_roi_comparisons:
                        print("ROI Comparison:" + str(meta_roi_comparison))
                        ROI_from, ROI_to = meta_roi_comparison
                        # Finally, we get to the final dataset
                        roi_pair_wise_dataset_for_classification = (all_pyspi_res.query("meta_ROI_from == @ROI_from & meta_ROI_to == @ROI_to & relevance_type == @relevance_type & stimulus_presentation == @stimulus_presentation")
                                                                    .reset_index(drop=True)
                                                                    .drop(columns=['index']))

                        # Extract this SPI
                        this_SPI_data = roi_pair_wise_dataset_for_classification.query(f"SPI == '{SPI}'")

                        # Find overall number of rows
                        num_rows = this_SPI_data.shape[0]

                        # Extract SPI values
                        this_column_data = this_SPI_data["value"]

                        # Find number of NaN in this column 
                        num_NaN = this_column_data.isna().sum()
                        prop_NaN = num_NaN / num_rows

                        # Find mode and SD
                        column_mode_max = this_column_data.value_counts().max()
                        column_SD = this_column_data.std()

                        # If 0% < num_NaN < 10%, impute by the mean of each component
                        if 0 < prop_NaN < 0.1:
                            values_imputed = (this_column_data
                                                .transform(lambda x: x.fillna(x.mean())))

                            this_column_data = values_imputed
                            print(f"Imputing column values for {SPI}")
                            this_SPI_data["value"] = this_column_data

                        # If there are: 
                        # - more than 10% NaN values;
                        # - more than 90% of the values are the same; OR
                        # - the standard deviation is less than 1*10**(-10)
                        # then remove the column
                        if prop_NaN > 0.1 or column_mode_max / num_rows > 0.9 or column_SD < 1*10**(-10):
                            print(f"{SPI} has low SD: {column_SD}, and/or too many mode occurences: {column_mode_max} out of {num_rows}, and/or {100*prop_NaN}% NaN")
                            continue
                        
                        # Start an empty list for the classification results
                        SPI_combo_res_list = []
                    
                        # Iterate over stimulus combos
                        for this_combo in stimulus_type_comparisons:

                            # Subset data to the corresponding stimulus pairs
                            if this_combo == ("face", "non-face"):
                                final_dataset_for_classification_this_combo = this_SPI_data.assign(stimulus_type = lambda x: np.where(x.stimulus_type == "face", "face", "non-face"))
                            else:
                                final_dataset_for_classification_this_combo = this_SPI_data.query(f"stimulus_type in {this_combo}")

                            # Fit classifier
                            X = final_dataset_for_classification_this_combo.value.to_numpy().reshape(-1, 1)
                            y = final_dataset_for_classification_this_combo.stimulus_type.to_numpy().reshape(-1, 1)
                            groups = final_dataset_for_classification_this_combo.subject_ID.to_numpy().reshape(-1, 1)
                            groups_flat = np.array([str(item[0]) for item in groups])

                            group_stratified_CV = StratifiedGroupKFold(n_splits = 10, shuffle = True, random_state=127)

                            # Make a deepcopy of the pipeline
                            this_iter_pipe = deepcopy(pipe)
                            this_classifier_res = cross_validate(this_iter_pipe, X, y, groups=groups_flat, cv=group_stratified_CV, scoring=scoring, n_jobs=n_jobs, 
                                                                        return_estimator=False, return_train_score=False)
                            
                            this_SPI_combo_df = pd.DataFrame({"SPI": [SPI], 
                                    "classifier": [classifier],
                                    "meta_ROI_from": [ROI_from],
                                    "meta_ROI_to": [ROI_to],
                                    "relevance_type": [relevance_type],
                                    "stimulus_presentation": [stimulus_presentation],
                                    "stimulus_combo": [this_combo], 
                                    "accuracy": [this_classifier_res['test_accuracy'].mean()],
                                    "balanced_accuracy": [this_classifier_res['test_balanced_accuracy'].mean()]})
                            
                            # Append to growing results list
                            comparing_between_stimulus_types_classification_results_list.append(this_SPI_combo_df)

        comparing_between_stimulus_types_classification_results = pd.concat(comparing_between_stimulus_types_classification_results_list).reset_index(drop=True)
        comparing_between_stimulus_types_classification_results.to_csv(f"{classification_res_path_averaged}/comparing_between_stimulus_types_{classifier}_classification_results.csv", index=False)

    # Comparing between relevance types
    if not os.path.isfile(f"{classification_res_path_averaged}/comparing_between_relevance_types_{classifier}_classification_results.csv"):
        # All comparisons list
        comparing_between_relevance_types_classification_results_list = []

        for meta_roi_comparison in meta_roi_comparisons:
            print("ROI Comparison:" + str(meta_roi_comparison))
            ROI_from, ROI_to = meta_roi_comparison
            for stimulus_presentation in stimulus_presentation_comparisons:
                print("Stimulus presentation:" + str(stimulus_presentation))
                # Finally, we get to the final dataset
                final_dataset_for_classification = all_pyspi_res.query("meta_ROI_from == @ROI_from & relevance_type in @relevance_type_comparisons and meta_ROI_to == @ROI_to & stimulus_presentation == @stimulus_presentation").reset_index(drop=True).drop(columns=['index'])

                for SPI in final_dataset_for_classification.SPI.unique():

                    # Extract this SPI
                    this_SPI_data = final_dataset_for_classification.query(f"SPI == '{SPI}'")

                    # Find overall number of rows
                    num_rows = this_SPI_data.shape[0]

                    # Extract SPI values
                    this_column_data = this_SPI_data["value"]

                    # Find number of NaN in this column 
                    num_NaN = this_column_data.isna().sum()
                    prop_NaN = num_NaN / num_rows

                    # Find mode and SD
                    column_mode_max = this_column_data.value_counts().max()
                    column_SD = this_column_data.std()

                    # If 0% < num_NaN < 10%, impute by the mean of each component
                    if 0 < prop_NaN < 0.1:
                        values_imputed = (this_column_data
                                            .transform(lambda x: x.fillna(x.mean())))

                        this_column_data = values_imputed
                        print(f"Imputing column values for {SPI}")
                        this_SPI_data["value"] = this_column_data

                    # If there are: 
                    # - more than 10% NaN values;
                    # - more than 90% of the values are the same; OR
                    # - the standard deviation is less than 1*10**(-10)
                    # then remove the column
                    if prop_NaN > 0.1 or column_mode_max / num_rows > 0.9 or column_SD < 1*10**(-10):
                        print(f"{SPI} has low SD: {column_SD}, and/or too many mode occurences: {column_mode_max} out of {num_rows}, and/or {100*prop_NaN}% NaN")
                        continue

                    # Start an empty list for the classification results
                    SPI_combo_res_list = []

                    # Fit classifier
                    X = this_SPI_data.value.to_numpy().reshape(-1, 1)
                    y = this_SPI_data.relevance_type.to_numpy().reshape(-1, 1)
                    groups = this_SPI_data.subject_ID.to_numpy().reshape(-1, 1)
                    groups_flat = np.array([str(item[0]) for item in groups])

                    group_stratified_CV = StratifiedGroupKFold(n_splits = 10, shuffle = True, random_state=127)

                    # Make a deepcopy of the pipeline
                    this_iter_pipe = deepcopy(pipe)

                    this_classifier_res = cross_validate(this_iter_pipe, X, y, groups=groups_flat, cv=group_stratified_CV, scoring="accuracy", n_jobs=n_jobs, 
                                                                return_estimator=False, return_train_score=False)
                    
                    this_SPI_relevance_results_df = pd.DataFrame({"SPI": [SPI], 
                                                        "meta_ROI_from": [ROI_from],
                                                        "meta_ROI_to": [ROI_to],
                                                        "stimulus_presentation": [stimulus_presentation],
                                                        "comparison": ["Relevant non-target vs. Irrelevant"], 
                                                        "accuracy": [this_classifier_res['test_accuracy'].mean()],
                                                        "balanced_accuracy": [this_classifier_res['test_balanced_accuracy'].mean()]})
                    
                    # Append to growing results list
                    comparing_between_relevance_types_classification_results_list.append(this_SPI_relevance_results_df)

        comparing_between_relevance_types_classification_results = pd.concat(comparing_between_relevance_types_classification_results_list).reset_index(drop=True)
        comparing_between_relevance_types_classification_results.to_csv(f"{classification_res_path_averaged }/comparing_between_relevance_types_{classifier}_classification_results.csv", index=False)

#################################################################################################
# Classification across participants with averaged epochs
#################################################################################################

if classification_type == "individual":
    # meta-ROI comparisons
    meta_ROIs = ["Category_Selective", "IPS", "Prefrontal_Cortex", "V1_V2"]
    meta_roi_comparisons = list(itertools.permutations(meta_ROIs, 2))

    # BY STIMULUS TYPE 
    # Load in this subject's pyspi results
    if not op.isfile(f"{classification_res_path_individual}/sub-{subject_ID}_comparing_between_stimulus_types_{classifier}_classification_results.csv"):

        # Load in results
        individual_subject_pyspi_res = pd.read_csv(f"{pyspi_res_path_individual}/sub-{subject_ID}_ses-1_all_pyspi_results_individual_epochs_1000ms.csv")

        # Fix stimulus_type where False to 'false' 
        individual_subject_pyspi_res['stimulus_type'] = individual_subject_pyspi_res['stimulus_type'].replace(False, 'false')
        individual_subject_pyspi_res['relevance_type'] = individual_subject_pyspi_res['relevance_type'].replace("Relevant non-target", "Relevant-non-target")

        # Relevance type comparisons
        relevance_type_comparisons = ["Relevant-non-target", "Irrelevant"]

        # Stimulus presentation comparisons
        stimulus_presentation_comparisons = individual_subject_pyspi_res.stimulus_presentation.unique().tolist()

        # Stimulus type comparisons
        stimulus_types = individual_subject_pyspi_res.stimulus_type.unique().tolist()
        stimulus_type_comparisons = list(itertools.combinations(stimulus_types, 2))

        # Also add in face vs. non-face
        stimulus_type_comparisons.append(("face", "non-face"))

        # All comparisons list
        comparing_between_stimulus_types_classification_results_list = []

        for meta_roi_comparison in meta_roi_comparisons:
            print("ROI Comparison:" + str(meta_roi_comparison))
            ROI_from, ROI_to = meta_roi_comparison
            for relevance_type in relevance_type_comparisons:
                print("Relevance type:" + str(relevance_type))
                for stimulus_presentation in stimulus_presentation_comparisons:
                    print("Stimulus presentation:" + str(stimulus_presentation))
                    # Finally, we get to the final dataset
                    final_dataset_for_classification = individual_subject_pyspi_res.query("meta_ROI_from == @ROI_from & meta_ROI_to == @ROI_to & relevance_type == @relevance_type & stimulus_presentation == @stimulus_presentation").reset_index(drop=True).drop(columns=['index'])

                    for SPI in final_dataset_for_classification.SPI.unique():

                        # Extract this SPI
                        this_SPI_data = final_dataset_for_classification.query(f"SPI == '{SPI}'")

                        # Find overall number of rows
                        num_rows = this_SPI_data.shape[0]

                        # Extract SPI values
                        this_column_data = this_SPI_data["value"]

                        # Find number of NaN in this column 
                        num_NaN = this_column_data.isna().sum()
                        prop_NaN = num_NaN / num_rows

                        # Find mode and SD
                        column_mode_max = this_column_data.value_counts().max()
                        column_SD = this_column_data.std()

                        # If 0% < num_NaN < 10%, impute by the mean of each component
                        if 0 < prop_NaN < 0.1:
                            values_imputed = (this_column_data
                                                .transform(lambda x: x.fillna(x.mean())))

                            this_column_data = values_imputed
                            print(f"Imputing column values for {SPI}")
                            this_SPI_data["value"] = this_column_data

                        # If there are: 
                        # - more than 10% NaN values;
                        # - more than 90% of the values are the same; OR
                        # - the standard deviation is less than 1*10**(-10)
                        # then remove the column
                        if prop_NaN > 0.1 or column_mode_max / num_rows > 0.9 or column_SD < 1*10**(-10):
                            print(f"{SPI} has low SD: {column_SD}, and/or too many mode occurences: {column_mode_max} out of {num_rows}, and/or {100*prop_NaN}% NaN")
                            continue
                        
                        # Start an empty list for the classification results
                        SPI_combo_res_list = []
                    
                        # Iterate over stimulus combos
                        for this_combo in stimulus_type_comparisons:

                            # Subset data to the corresponding stimulus pairs
                            if this_combo == ("face", "non-face"):
                                final_dataset_for_classification_this_combo = this_SPI_data.assign(stimulus_type = lambda x: np.where(x.stimulus_type == "face", "face", "non-face"))
                            else:
                                final_dataset_for_classification_this_combo = this_SPI_data.query(f"stimulus_type in {this_combo}")


                            # Fit classifier
                            X = final_dataset_for_classification_this_combo.value.to_numpy().reshape(-1, 1)
                            y = final_dataset_for_classification_this_combo.stimulus_type.to_numpy().reshape(-1, 1)

                            stimulus_stratified_CV = StratifiedKFold(n_splits = 10, shuffle = True, random_state=127)

                            # Make a deepcopy of the pipeline
                            this_iter_pipe = deepcopy(pipe)
                            this_classifier_res = cross_validate(this_iter_pipe, X, y, cv=stimulus_stratified_CV, scoring=scoring, n_jobs=n_jobs, 
                                                                        return_estimator=False, return_train_score=False)
                            
                            this_SPI_combo_df = pd.DataFrame({subject_ID: ["sub-" + subject_ID],
                                                                "SPI": [SPI], 
                                                                "meta_ROI_from": [ROI_from],
                                                                "meta_ROI_to": [ROI_to],
                                                                "relevance_type": [relevance_type],
                                                                "stimulus_presentation": [stimulus_presentation],
                                                                "stimulus_combo": [this_combo], 
                                                                "accuracy": [this_classifier_res['test_accuracy'].mean()],
                                                                "balanced_accuracy": [this_classifier_res['test_balanced_accuracy'].mean()]})
                            
                            # Append to growing results list
                            comparing_between_stimulus_types_classification_results_list.append(this_SPI_combo_df)

        comparing_between_stimulus_types_classification_results = pd.concat(comparing_between_stimulus_types_classification_results_list).reset_index(drop=True)
        comparing_between_stimulus_types_classification_results.to_csv(f"{classification_res_path_individual}/sub-{subject_ID}_comparing_between_stimulus_types_{classifier}_classification_results.csv", index=False)
    
    # Comparing between relevance types
    # BY RELEVANCE TYPE
    if not os.path.isfile(f"{classification_res_path_individual}/sub-{subject_ID}_comparing_between_relevance_types_{classifier}_classification_results.csv"):
        
        # Load in results
        individual_subject_pyspi_res = pd.read_csv(f"{pyspi_res_path_individual}/sub-{subject_ID}_ses-1_all_pyspi_results_individual_epochs_1000ms.csv")

        # Fix stimulus_type where False to 'false' 
        individual_subject_pyspi_res['stimulus_type'] = individual_subject_pyspi_res['stimulus_type'].replace(False, 'false')

        # Relevance type comparisons
        relevance_type_comparisons = ["Relevant-non-target", "Irrelevant"]

        # Stimulus presentation comparisons
        stimulus_presentation_comparisons = individual_subject_pyspi_res.stimulus_presentation.unique().tolist()

        # All comparisons list
        comparing_between_relevance_types_classification_results_list = []

        for meta_roi_comparison in meta_roi_comparisons:
            print("ROI Comparison:" + str(meta_roi_comparison))
            ROI_from, ROI_to = meta_roi_comparison
            for stimulus_presentation in stimulus_presentation_comparisons:
                print("Stimulus presentation:" + str(stimulus_presentation))
                # Finally, we get to the final dataset
                final_dataset_for_classification = individual_subject_pyspi_res.query("meta_ROI_from == @ROI_from & relevance_type in @relevance_type_comparisons and meta_ROI_to == @ROI_to & stimulus_presentation == @stimulus_presentation").reset_index(drop=True).drop(columns=['index'])

                for SPI in final_dataset_for_classification.SPI.unique():

                    # Extract this SPI
                    this_SPI_data = final_dataset_for_classification.query(f"SPI == '{SPI}'")

                    # Find overall number of rows
                    num_rows = this_SPI_data.shape[0]

                    # Extract SPI values
                    this_column_data = this_SPI_data["value"]

                    # Find number of NaN in this column 
                    num_NaN = this_column_data.isna().sum()
                    prop_NaN = num_NaN / num_rows

                    # Find mode and SD
                    column_mode_max = this_column_data.value_counts().max()
                    column_SD = this_column_data.std()

                    # If 0% < num_NaN < 10%, impute by the mean of each component
                    if 0 < prop_NaN < 0.1:
                        values_imputed = (this_column_data
                                            .transform(lambda x: x.fillna(x.mean())))

                        this_column_data = values_imputed
                        print(f"Imputing column values for {SPI}")
                        this_SPI_data["value"] = this_column_data

                    # If there are: 
                    # - more than 10% NaN values;
                    # - more than 90% of the values are the same; OR
                    # - the standard deviation is less than 1*10**(-10)
                    # then remove the column
                    if prop_NaN > 0.1 or column_mode_max / num_rows > 0.9 or column_SD < 1*10**(-10):
                        print(f"{SPI} has low SD: {column_SD}, and/or too many mode occurences: {column_mode_max} out of {num_rows}, and/or {100*prop_NaN}% NaN")
                        continue

                    # Start an empty list for the classification results
                    SPI_combo_res_list = []

                    # Fit classifier
                    X = this_SPI_data.value.to_numpy().reshape(-1, 1)
                    y = this_SPI_data.relevance_type.to_numpy().reshape(-1, 1)

                    stimulus_stratified_CV = StratifiedKFold(n_splits = 10, shuffle = True, random_state=127)

                    # Make a deepcopy of the pipeline
                    this_iter_pipe = deepcopy(pipe)
                    this_classifier_res = cross_validate(this_iter_pipe, X, y, cv=stimulus_stratified_CV, scoring=scoring, n_jobs=n_jobs, 
                                            return_estimator=False, return_train_score=False)
                    
                    this_SPI_relevance_results_df = pd.DataFrame({subject_ID: ["sub-" + subject_ID],
                                                        "SPI": [SPI], 
                                                        "meta_ROI_from": [ROI_from],
                                                        "meta_ROI_to": [ROI_to],
                                                        "stimulus_presentation": [stimulus_presentation],
                                                        "comparison": ["Relevant non-target vs. Irrelevant"], 
                                                        "accuracy": [this_classifier_res['test_accuracy'].mean()],
                                                        "balanced_accuracy": [this_classifier_res['test_balanced_accuracy'].mean()]})
                    
                    # Append to growing results list
                    comparing_between_relevance_types_classification_results_list.append(this_SPI_relevance_results_df)

        comparing_between_relevance_types_classification_results = pd.concat(comparing_between_relevance_types_classification_results_list).reset_index(drop=True)
        comparing_between_relevance_types_classification_results.to_csv(f"{classification_res_path_individual}/sub-{subject_ID}_comparing_between_relevance_types_{classifier}_classification_results.csv", index=False)
