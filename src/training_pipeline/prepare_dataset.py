"""
Given a new dataset, preprocess it for classification

1 - Load emg env
2 - run_AM_process
3 - parse_segment_dat_list
4 - generate_final_features
5 - parse with event_code_dict

NOTE: this is pre-existing exploratory/scratch code carried over from
NBT_EMB_Classifier_Analyses (not written cleanly to run top-to-bottom as a
script — e.g. `model_save_dir`/`fin_segment_frame`/`basename` are used
before being defined). Only the hardcoded repo-relative paths below have
been updated for this repo; the ordering issues are pre-existing and left
as-is.
"""

from generate_training_dataset import (
    run_AM_process, parse_segment_dat_list, generate_final_features)
import xgboost as xgb
import os
import sys
from glob import glob
import numpy as np
import pandas as pd
import json
from tqdm import tqdm

this_dir = os.path.dirname(os.path.abspath(__file__))  # src/training_pipeline
sys.path.append(this_dir)

artifact_dir = os.path.join(this_dir, 'artifacts')


def load_env_file(env_path):
    """
    Load env file and remove nans

    Inputs:
        env_path: str, path to env file
    Outputs:
        env: np.array, env file with nans removed
            - Expected shape: (n_tastes, n_trials, n_timepoints)
    """
    env = np.load(env_path)
    # If nans are present
    non_nan_trials = ~np.isnan(env).any(axis=(0, 2))
    env = env[:, non_nan_trials, :]
    return env


############################################################
# Find and load raw data
############################################################
data_dir = '/media/storage/ABU_GC-EMG_Data/emg_process_only'
emg_output_dirs = sorted(glob(os.path.join(data_dir, '*', 'emg_output')))

additional_data_dir = '/media/fastdata/Natasha_classifier_data'
additional_emg_output_dirs = sorted(
    glob(os.path.join(additional_data_dir, '*', '*', '*', 'emg_output')))
emg_output_dirs.extend(additional_emg_output_dirs)

# For each day of experiment, load env and table files
data_subdirs = [glob(os.path.join(x, '*')) for x in emg_output_dirs]
data_subdirs = [item for sublist in data_subdirs for item in sublist]
# Make sure that path is a directory
data_subdirs = [subdir for subdir in data_subdirs if os.path.isdir(subdir)]
data_subdirs = [x for x in data_subdirs if 'emg' in os.path.basename(x)]
# Make sure that subdirs are in order
subdir_basenames = [subdir.split('/')[-3].lower() for subdir in data_subdirs]
emg_name = [os.path.basename(x) for x in data_subdirs]

emg_path_df = pd.DataFrame({'path': data_subdirs,
                            'basename': subdir_basenames,
                            'emg_name': emg_name,
                            'emg': True})

# Only keep emgad
emg_path_df['emg_name'] = [x.lower() for x in emg_path_df.emg_name]
emg_path_df = emg_path_df[emg_path_df.emg_name == 'emgad']

# Load env files
env_paths = [glob(os.path.join(x, '*env.npy'))[0] for x in emg_path_df.path]
emg_path_df['env_path'] = env_paths

############################################################
# Necessary information to process a single file
# 1) Env file path
# 2)


class ClassifierHandler():
    """
    Class to handle all classifier operations
    """

    def __init__(
            self,
            model_dir,
            output_dir,
            env_path,
    ):
        """
        Initialize classifier handler

        Inputs:
            model_dir: str, path to model directory
            output_dir: str, path to output directory
            env_path: str, path to EMG env file
        """
        self.model_dir = model_dir
        self.output_dir = output_dir
        self.env_path = env_path
        self.run_AM_process = run_AM_process
        self.parse_segment_dat_list = parse_segment_dat_list
        self.generate_final_features = generate_final_features

    def load_env_file(self):
        """
        Load env file and remove nans

        Outputs:
            env: np.array, env file with nans removed
                - Expected shape: (n_tastes, n_trials, n_timepoints)
        """
        env = np.load(self.env_path)
        # If nans are present
        non_nan_trials = ~np.isnan(env).any(axis=(0, 2))
        env = env[:, non_nan_trials, :]
        return env

    def run_pre_process(self):
        """
        Run the entire process
        """
        env = self.load_env_file()
        segment_dat_list, feature_names, inds = self.run_AM_process(env)
        segment_frame = self.parse_segment_dat_list(segment_dat_list, inds)
        all_features = np.stack(segment_frame.features.values)
        scaled_segments = np.stack(segment_frame.segment_norm_interp.values)
        all_features, feature_names, scaled_features = \
            self.generate_final_features(all_features, feature_names, scaled_segments,
                                         artifact_dir=self.output_dir)
        self.feature_names = feature_names
        return all_features, feature_names, scaled_features, segment_frame

    def load_model(self):
        """
        Load the model
        """
        clf = xgb.XGBClassifier()
        clf.load_model(os.path.join(self.model_dir, 'xgb_model.json'))
        return clf

    def predict(self, X):
        """
        Predict on X
        """
        clf = self.load_model()
        y_pred = clf.predict(X)
        return y_pred

    def parse_and_predict(self):
        """
        Run the entire process
        """
        all_features, feature_names, scaled_features, segment_frame = self.run_pre_process()
        y_pred = self.predict(scaled_features)
        segment_frame['raw_features'] = list(all_features)
        segment_frame['features'] = list(scaled_features)
        segment_frame['pred_event_type'] = y_pred
        self.segment_frame = segment_frame
        return y_pred, segment_frame


this_handler = ClassifierHandler(
    model_dir=model_save_dir,
    output_dir=artifact_dir,
    env_path=env_paths[0],
)
y_pred, segment_frame = this_handler.parse_and_predict()

# env_path = env_paths[0]
# basename = emg_path_df.basename.values[0]
# env = load_env_file(env_path)
# (
#     segment_dat_list,
#     feature_names,
#     inds,
#     )= run_AM_process(env)
#
# segment_frame = parse_segment_dat_list(segment_dat_list, inds)
# all_features = np.stack(segment_frame.features.values)
# scaled_segments = np.stack(segment_frame.segment_norm_interp.values)
#
# (
#     all_features,
#     feature_names,
#     scaled_features,
#     ) = generate_final_features(all_features, feature_names, scaled_segments,
#                                 artifact_dir=artifact_dir)
#
# event_code_dict_path = os.path.join(artifact_dir, 'event_code_dict.json')
# with open(event_code_dict_path, 'r') as f:
#     event_code_dict = json.load(f)
#
# inv_event_code_dict = {v: k for k, v in event_code_dict.items()}
#
# model_save_dir = os.path.join(artifact_dir, 'xgb_model')
# clf = xgb.XGBClassifier()
# clf.load_model(os.path.join(model_save_dir, 'xgb_model.json'))
#
# y_pred = clf.predict(scaled_features)
# y_pred_names = [inv_event_code_dict[x] for x in y_pred]

# Check that this matches the original prediction
(y_pred_names ==
 fin_segment_frame.loc[fin_segment_frame.basename == basename].pred_event_type.values).all()

############################################################

##############################
# 1 - Load env files
envs_list = [load_env_file(x) for x in tqdm(env_paths)]
##############################

# envs_list = []
# for ind, row in tqdm(emg_path_df.iterrows()):
#     env_path = glob(os.path.join(row.path, '*env.npy'))[0]
#     env = np.load(env_path)
#     # If nans are present, cut them out
#     non_nan_trials = ~np.isnan(env).any(axis = (0,2))
#     env = env[:,non_nan_trials,:]
#     envs_list.append(env)

# Also load info files to extract taste names
# info_files_paths = glob(os.path.join(data_dir, '*', '*.info'))
data_dir_list = [os.path.dirname(os.path.dirname(x)) for x in data_subdirs]
info_files_paths = [glob(os.path.join(x, '*.info'))[0] for x in data_dir_list]
info_basenames = [x.split('/')[-2].lower() for x in info_files_paths]
info_files_list = [json.load(open(x, 'r')) for x in info_files_paths]
taste_order_list = [x['taste_params']['tastes'] for x in info_files_list]
taste_order_dict = dict(zip(info_basenames, taste_order_list))

############################################################
# Process raw data
############################################################

##############################
# 2 - run_AM_process
process_outs = [run_AM_process(x) for x in tqdm(envs_list)]
##############################

original_feature_names = process_outs[0][1]
segment_dat_list = [x[0] for x in process_outs]
data_inds = [x[2] for x in process_outs]

segment_frame_list = [
    parse_segment_dat_list(dat, inds)
    for dat, inds in tqdm(zip(segment_dat_list, data_inds))
]

for i in range(len(segment_frame_list)):
    this_frame = segment_frame_list[i]
    basename = emg_path_df.basename.values[i]
    this_frame['basename'] = basename
    this_frame['animal_num'] = basename.split('_')[0]

    this_taste_order = taste_order_dict[basename]
    this_frame['taste_name'] = [this_taste_order[x]
                                for x in this_frame.taste.values]

    all_features = np.stack(this_frame.features.values)
    feature_names = original_feature_names.copy()
    scaled_segments = np.stack(this_frame.segment_norm_interp.values)
    all_features, feature_names, scaled_features = \
        generate_final_features(all_features, feature_names, scaled_segments,
                                artifact_dir=artifact_dir)

    this_frame['raw_features'] = list(all_features)
    this_frame['features'] = list(scaled_features)

fin_segment_frame = pd.concat(segment_frame_list)

############################################################
# Perform predictions
############################################################
event_code_dict_path = os.path.join(artifact_dir, 'event_code_dict.json')
with open(event_code_dict_path, 'r') as f:
    event_code_dict = json.load(f)

inv_event_code_dict = {v: k for k, v in event_code_dict.items()}

model_save_dir = os.path.join(artifact_dir, 'xgb_model')
clf = xgb.XGBClassifier()
clf.load_model(os.path.join(model_save_dir, 'xgb_model.json'))

X = np.stack(fin_segment_frame.features.values)
y_pred = clf.predict(X)
y_pred_names = [inv_event_code_dict[x] for x in y_pred]

fin_segment_frame['pred_event_type'] = y_pred_names

# Save as pkl
fin_segment_frame.to_pickle(os.path.join(
    artifact_dir, 'all_datasets_emg_pred.pkl'))
# Write out feature names
with open(os.path.join(artifact_dir, 'all_datasets_feature_names.txt'), 'w') as f:
    f.write('\n'.join(feature_names))
