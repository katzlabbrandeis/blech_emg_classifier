import xgboost as xgb
import os
import sys
from glob import glob
import numpy as np
import pandas as pd
import json
from tqdm import tqdm

from preprocessing import run_AM_process, parse_segment_dat_list, generate_final_features

def get_paths():
    script_path = os.path.abspath(__file__)
    src_dir = os.path.dirname(os.path.dirname(script_path))
    artifact_dir = os.path.join(src_dir, 'artifacts')
    model_save_dir = os.path.join(artifact_dir, 'model')
    return artifact_dir, model_save_dir

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
        non_nan_trials = ~np.isnan(env).any(axis = (0,2))
        env = env[:,non_nan_trials,:]
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

    def load_event_types(self):
        """
        Load event types
        """
        with open(os.path.join(self.output_dir, 'event_code_dict.json'), 'r') as f:
            event_code_dict = json.load(f)
        return event_code_dict

    def predict(self, X):
        """
        Predict on X
        """
        clf = self.load_model()
        y_pred = clf.predict(X)
        event_code_dict = self.load_event_types()
        inv_event_code_dict = {v: k for k, v in event_code_dict.items()}
        y_pred_names = [inv_event_code_dict[x] for x in y_pred]
        return y_pred, y_pred_names

    def parse_and_predict(self):
        """
        Run the entire process
        """
        all_features, feature_names, scaled_features, segment_frame = self.run_pre_process()
        y_pred, y_pred_names = self.predict(scaled_features)
        segment_frame['raw_features'] = list(all_features)
        segment_frame['features'] = list(scaled_features)
        segment_frame['pred'] = y_pred
        segment_frame['pred_names'] = y_pred_names
        self.segment_frame = segment_frame
        return y_pred, segment_frame
