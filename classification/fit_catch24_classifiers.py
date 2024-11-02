from copy import deepcopy
from glob import glob
import os
from os import path as op
import numpy as np
import pandas as pd
import sys
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, roc_auc_score, accuracy_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedGroupKFold, cross_validate, StratifiedKFold, LeaveOneOut, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.utils import resample
import itertools
import argparse

# add path to classification analysis functions
from mixed_sigmoid_normalisation import MixedSigmoidScaler

parser=argparse.ArgumentParser()
parser.add_argument('--bids_root',
                    type=str,
                    default='/project/hctsa/annie/data/Cogitate_MEG/',
                    help='Path to the BIDS root directory')
parser.add_argument('--n_jobs',
                    type=int,
                    default=1,
                    help='Number of concurrent processing jobs')
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
classification_type = opt.classification_type
classifier = opt.classifier

# Load data paths
catch24_res_path = f"{bids_root}/derivatives/time_series_features"
catch24_res_path_averaged = f"{catch24_res_path}/averaged_epochs"
catch24_res_path_individual = f"{catch24_res_path}/individual_epochs"

classification_res_path = f"{bids_root}/derivatives/classification_results"
classification_res_path_averaged = f"{classification_res_path}/across_participants"
classification_res_path_individual = f"{classification_res_path}/within_participants"

# Make classification result directories
os.makedirs(classification_res_path_averaged, exist_ok=True)
os.makedirs(classification_res_path_individual, exist_ok=True)

# Define classifier
if classifier == "Linear_SVM":
    model = svm.SVC(C=1, class_weight='balanced', kernel='linear', random_state=127, probability=True)
else:
    model = LogisticRegression(penalty='l1', C=1, solver='liblinear', class_weight='balanced', random_state=127)

pipe = Pipeline([('scaler', MixedSigmoidScaler(unit_variance=True)), 
                            ('model', model)])

# Define scoring type
scoring = {'accuracy': 'accuracy',
           'balanced_accuracy': 'balanced_accuracy',
           'AUC': make_scorer(roc_auc_score, needs_proba=True)}

# Defiene cross-validators
LOOCV = LeaveOneOut()
SKF = StratifiedKFold(n_splits=5, shuffle=True, random_state=127)

#################################################################################################
# Classification across participants with averaged epochs
#################################################################################################

if classification_type == "averaged":

    # Load in catch24 results
    all_catch24_res_list = []
    for catch24_res_file in glob(f"{catch24_res_path_averaged}/*all_catch24_results_1000ms.csv"):
        catch24_res = pd.read_csv(catch24_res_file)
        # Reset index
        catch24_res.reset_index(inplace=True, drop=True)
        catch24_res['stimulus_type'] = catch24_res['stimulus_type'].replace(False, 'false').replace('False', 'false')
        catch24_res['relevance_type'] = catch24_res['relevance_type'].replace("Relevant non-target", "Relevant-non-target")
        # Rename stimulus to stimulus_presentation if it is present
        if 'stimulus' in catch24_res.columns:
            if 'stimulus_presentation' in catch24_res.columns:
                catch24_res.drop(columns=['stimulus'], inplace=True)
            else:
                catch24_res = catch24_res.rename(columns={'stimulus': 'stimulus_presentation'})

        all_catch24_res_list.append(catch24_res)
    all_catch24_res = pd.concat(all_catch24_res_list)

    # Define comparisons

    # meta-ROI comparisons
    Meta_ROIs = ["Category_Selective", "IPS", "Prefrontal_Cortex", "V1_V2"]

    # Relevance type comparisons
    relevance_type_comparisons = ["Relevant-non-target", "Irrelevant"]

    # Stimulus presentation comparisons
    stimulus_presentation_comparisons = ["on", "off"]

    # Stimulus type comparisons
    stimulus_types = all_catch24_res.stimulus_type.unique().tolist()
    stimulus_type_comparisons = list(itertools.combinations(stimulus_types, 2))

    # Also add in face vs. non-face
    stimulus_type_comparisons.append(("face", "non-face"))

    # Comparing between stimulus types
    if not os.path.isfile(f"{classification_res_path_averaged}/comparing_between_stimulus_types_catch24_{classifier}_classification_results.csv"):
        # All comparisons list
        comparing_between_stimulus_types_classification_results_list = []

        for relevance_type in relevance_type_comparisons:
            print("Relevance type:" + str(relevance_type))
            for stimulus_presentation in stimulus_presentation_comparisons:
                print("Stimulus presentation:" + str(stimulus_presentation))
                for catch24_feature in all_catch24_res.catch24_feature.unique():
                    # First, look at each meta-ROI pair separately
                    for Meta_ROI in Meta_ROIs:
                        print("ROI:" + Meta_ROI)
                        # Finally, we get to the final dataset
                        roi_dataset_for_classification = (all_catch24_res.query("Meta_ROI == @Meta_ROI & relevance_type == @relevance_type & stimulus_presentation == @stimulus_presentation")
                                                                    .reset_index(drop=True)
                                                                    .drop(columns=['index']))

                        # Extract this catch24_feature
                        this_catch24_feature_data = roi_dataset_for_classification.query(f"catch24_feature == '{catch24_feature}'")

                        # Print first five rows
                        print(this_catch24_feature_data.head())

                        # Find overall number of rows
                        num_rows = this_catch24_feature_data.shape[0]

                        # Extract catch24_feature values
                        this_column_data = this_catch24_feature_data["feature_value"]

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
                            print(f"Imputing column values for {catch24_feature}")
                            this_catch24_feature_data["feature_value"] = this_column_data

                        # If there are: 
                        # - more than 10% NaN values;
                        # - more than 90% of the values are the same; OR
                        # - the standard deviation is less than 1*10**(-10)
                        # then remove the column
                        if prop_NaN > 0.1 or column_mode_max / num_rows > 0.9 or column_SD < 1*10**(-10):
                            print(f"{catch24_feature} has low SD: {column_SD}, and/or too many mode occurences: {column_mode_max} out of {num_rows}, and/or {100*prop_NaN}% NaN")
                            continue
                        
                        # Start an empty list for the classification results
                        catch24_feature_combo_res_list = []
                    
                        # Iterate over stimulus combos
                        for this_combo in stimulus_type_comparisons:

                            # Subset data to the corresponding stimulus pairs
                            if this_combo == ("face", "non-face"):
                                final_dataset_for_classification_this_combo = this_catch24_feature_data.assign(stimulus_type = lambda x: np.where(x.stimulus_type == "face", "face", "non-face"))
                            else:
                                final_dataset_for_classification_this_combo = this_catch24_feature_data.query(f"stimulus_type in {this_combo}")

                            # Fit classifier
                            X = final_dataset_for_classification_this_combo.feature_value.to_numpy().reshape(-1, 1)
                            y = final_dataset_for_classification_this_combo.stimulus_type.to_numpy().reshape(-1, 1)
                            groups = final_dataset_for_classification_this_combo.subject_ID.to_numpy().reshape(-1, 1)
                            groups_flat = np.array([str(item[0]) for item in groups])

                            group_stratified_CV = StratifiedGroupKFold(n_splits = 10, shuffle = True, random_state=127)

                            # Make a deepcopy of the pipeline
                            this_iter_pipe = deepcopy(pipe)
                            this_classifier_res = cross_validate(this_iter_pipe, X, y, groups=groups_flat, cv=group_stratified_CV, scoring=scoring, n_jobs=n_jobs, 
                                                                        return_estimator=False, return_train_score=False)
                            
                            this_catch24_feature_combo_df = pd.DataFrame({"catch24_feature": [catch24_feature], 
                                    "classifier": [classifier],
                                    "Meta_ROI": [Meta_ROI],
                                    "relevance_type": [relevance_type],
                                    "stimulus_presentation": [stimulus_presentation],
                                    "stimulus_combo": [this_combo], 
                                    "accuracy": [this_classifier_res['test_accuracy'].mean()],
                                    "balanced_accuracy": [this_classifier_res['test_balanced_accuracy'].mean()]})
                            
                            # Append to growing results list
                            comparing_between_stimulus_types_classification_results_list.append(this_catch24_feature_combo_df)

        comparing_between_stimulus_types_classification_results = pd.concat(comparing_between_stimulus_types_classification_results_list).reset_index(drop=True)
        comparing_between_stimulus_types_classification_results.to_csv(f"{classification_res_path_averaged}/comparing_between_stimulus_types_catch24_{classifier}_classification_results.csv", index=False)

    # Comparing between relevance types
    if not os.path.isfile(f"{classification_res_path_averaged}/comparing_between_relevance_types_catch24_{classifier}_classification_results.csv"):
        # All comparisons list
        comparing_between_relevance_types_classification_results_list = []

        for Meta_ROI in Meta_ROIs:
            print("ROI:" + Meta_ROI)
            for stimulus_presentation in stimulus_presentation_comparisons:
                print("Stimulus presentation:" + str(stimulus_presentation))
                # Finally, we get to the final dataset
                final_dataset_for_classification = all_catch24_res.query("Meta_ROI == @Meta_ROI & relevance_type in @relevance_type_comparisons & stimulus_presentation == @stimulus_presentation").reset_index(drop=True).drop(columns=['index'])

                for catch24_feature in final_dataset_for_classification.catch24_feature.unique():

                    # Extract this catch24_feature
                    this_catch24_feature_data = final_dataset_for_classification.query(f"catch24_feature == '{catch24_feature}'")

                    # PRint first five rows
                    print(this_catch24_feature_data.head())

                    # Find overall number of rows
                    num_rows = this_catch24_feature_data.shape[0]

                    # Extract catch24_feature values
                    this_column_data = this_catch24_feature_data["feature_value"]

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
                        print(f"Imputing column values for {catch24_feature}")
                        this_catch24_feature_data["feature_value"] = this_column_data

                    # If there are: 
                    # - more than 10% NaN values;
                    # - more than 90% of the values are the same; OR
                    # - the standard deviation is less than 1*10**(-10)
                    # then remove the column
                    if prop_NaN > 0.1 or column_mode_max / num_rows > 0.9 or column_SD < 1*10**(-10):
                        print(f"{catch24_feature} has low SD: {column_SD}, and/or too many mode occurences: {column_mode_max} out of {num_rows}, and/or {100*prop_NaN}% NaN")
                        continue

                    # Start an empty list for the classification results
                    catch24_feature_combo_res_list = []

                    # Fit classifier
                    X = this_catch24_feature_data.feature_value.to_numpy().reshape(-1, 1)
                    y = this_catch24_feature_data.relevance_type.to_numpy().reshape(-1, 1)
                    groups = this_catch24_feature_data.subject_ID.to_numpy().reshape(-1, 1)
                    groups_flat = np.array([str(item[0]) for item in groups])

                    group_stratified_CV = StratifiedGroupKFold(n_splits = 10, shuffle = True, random_state=127)

                    # Make a deepcopy of the pipeline
                    this_iter_pipe = deepcopy(pipe)

                    this_classifier_res = cross_validate(this_iter_pipe, X, y, groups=groups_flat, cv=group_stratified_CV, scoring="accuracy", n_jobs=n_jobs, 
                                                                return_estimator=False, return_train_score=False)
                    
                    this_catch24_feature_relevance_results_df = pd.DataFrame({"catch24_feature": [catch24_feature], 
                                                        "Meta_ROI_from": [Meta_ROI],
                                                        "stimulus_presentation": [stimulus_presentation],
                                                        "comparison": ["Relevant non-target vs. Irrelevant"], 
                                                        "accuracy": [this_classifier_res['test_accuracy'].mean()],
                                                        "balanced_accuracy": [this_classifier_res['test_balanced_accuracy'].mean()]})
                    
                    # Append to growing results list
                    comparing_between_relevance_types_classification_results_list.append(this_catch24_feature_relevance_results_df)

        comparing_between_relevance_types_classification_results = pd.concat(comparing_between_relevance_types_classification_results_list).reset_index(drop=True)
        comparing_between_relevance_types_classification_results.to_csv(f"{classification_res_path_averaged }/comparing_between_relevance_types_catch24_{classifier}_classification_results.csv", index=False)
