"""
ClassifierHandler Module

This module provides functionality for handling EMG signal classification tasks.
It includes tools for loading EMG data, preprocessing signals, feature extraction,
and making predictions using an XGBoost classifier.

The main class ClassifierHandler orchestrates the entire pipeline from loading
raw EMG signals to generating movement classifications.

Key Features:
    - EMG signal preprocessing and feature extraction
    - Model loading and prediction
    - Automated movement detection and classification
    - Support for batch processing of trials

Dependencies:
    - xgboost
    - numpy
    - pandas
    - json
    - tqdm
    - preprocessing module (local)
"""

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
    """
    Get the paths to important directories in the project.

    Returns:
        tuple: Contains two strings:
            - artifact_dir (str): Path to the artifacts directory
            - model_save_dir (str): Path to the model directory within artifacts
    """
    script_path = os.path.abspath(__file__)
    src_dir = os.path.dirname(os.path.dirname(script_path))
    artifact_dir = os.path.join(src_dir, 'artifacts')
    model_save_dir = os.path.join(artifact_dir, 'model')
    return artifact_dir, model_save_dir


class ClassifierHandler():
    """
    A class to handle EMG signal classification operations.

    This class manages the complete pipeline for EMG signal processing and classification,
    including data loading, preprocessing, feature extraction, and prediction.

    Attributes:
        model_dir (str): Directory containing the trained model files
        output_dir (str): Directory for saving output files and predictions
        env_path (str): Path to the EMG envelope file
        feature_names (list): Names of extracted features (set after preprocessing)
        segment_frame (pd.DataFrame): DataFrame containing processed segments and predictions

    Methods:
        load_env_file: Loads and cleans EMG envelope data
        run_pre_process: Executes complete preprocessing pipeline
        load_model: Loads trained XGBoost classifier
        load_event_types: Loads event type mappings
        predict: Makes predictions on processed features
        parse_and_predict: Runs complete pipeline from raw data to predictions
    """

    def __init__(
            self,
            model_dir,
            output_dir,
            env_path,
    ):
        """
        Initialize classifier handler for EMG signal processing

        Inputs:
            model_dir: str, path to directory containing trained XGBoost model and artifacts
            output_dir: str, path to directory for saving predictions and processed data
            env_path: str, path to EMG envelope file (.npy) containing preprocessed signals
                     with shape (n_tastes, n_trials, n_timepoints)
        """
        self.model_dir = model_dir
        self.output_dir = output_dir
        self.env_path = env_path
        self.run_AM_process = run_AM_process
        self.parse_segment_dat_list = parse_segment_dat_list
        self.generate_final_features = generate_final_features

    def load_env_file(self):
        """
        Load EMG envelope file and remove any trials containing NaN values.

        The envelope file contains preprocessed EMG signals organized by tastes and trials.
        This method loads the file and cleans it by removing any trials that contain
        NaN values to ensure data quality.

        Returns:
            np.ndarray: Cleaned EMG envelope data with shape (n_tastes, n_trials, n_timepoints)
                where:
                - n_tastes: Number of taste stimuli
                - n_trials: Number of valid trials (after removing NaN trials)
                - n_timepoints: Number of time points in each trial
        """
        # Load the EMG envelope data array
        env = np.load(self.env_path)
        return env

    def run_pre_process(self):
        """
        Execute the complete preprocessing pipeline for EMG data.

        This method orchestrates the entire preprocessing workflow:
        1. Loads and cleans the EMG envelope data
        2. Performs automated movement detection
        3. Extracts segments and features
        4. Scales and normalizes the features

        Returns:
            tuple: Contains four elements:
                - all_features (np.ndarray): Raw extracted features
                - feature_names (list): Names of the extracted features
                - scaled_features (np.ndarray): Normalized and scaled features
                - segment_frame (pd.DataFrame): DataFrame containing processed segments
                  and their associated metadata
        """
        # Load and clean the EMG envelope data
        env = self.load_env_file()

        # Detect movements and extract initial features
        # segment_dat_list contains raw segments
        # inds contains timing information for each segment
        segment_dat_list, feature_names, inds = self.run_AM_process(env)

        # Convert raw segments into a DataFrame with metadata
        segment_frame = self.parse_segment_dat_list(segment_dat_list, inds)

        # Stack features and normalized segments into arrays
        # features are time-domain and frequency-domain characteristics
        all_features = np.stack(segment_frame.features.values)
        # scaled_segments are the normalized EMG traces themselves
        scaled_segments = np.stack(segment_frame.segment_norm_interp.values)
        # All features is modified to ensure it has the same shape as scaled_features
        all_features, feature_names, scaled_features = \
            self.generate_final_features(all_features, feature_names, scaled_segments,
                                         artifact_dir=self.output_dir)
        self.feature_names = feature_names
        return all_features, feature_names, scaled_features, segment_frame

    def load_model(self):
        """
        Load the trained XGBoost classifier model from disk.

        Loads a pre-trained XGBoost model from the specified model directory.
        The model should be saved as 'xgb_model.json' in the model directory.

        Returns:
            xgb.XGBClassifier: Loaded XGBoost classifier model ready for predictions
        """
        # Initialize an empty XGBoost classifier
        clf = xgb.XGBClassifier()
        # Load the pre-trained model parameters from disk
        # The model was trained on a standardized set of EMG movement features
        clf.load_model(os.path.join(self.model_dir, 'xgb_model.json'))
        return clf

    def load_event_types(self):
        """
        Load the event type mapping dictionary from disk.

        Reads the event code dictionary that maps between numerical class indices
        and human-readable movement type labels.

        Returns:
            dict: Mapping between movement type names (str) and their numerical
                 codes (int) used by the classifier
        """
        with open(os.path.join(self.output_dir, 'event_code_dict.json'), 'r') as f:
            event_code_dict = json.load(f)
        return event_code_dict

    def predict(self, X):
        """
        Make predictions on preprocessed EMG features.

        This method:
        1. Loads the trained classifier
        2. Makes predictions on the input features
        3. Converts numerical predictions to human-readable movement types
        4. Calculates prediction probabilities

        Args:
            X (np.ndarray): Preprocessed and scaled feature matrix

        Returns:
            tuple: Contains three elements:
                - y_pred (np.ndarray): Array of predicted class indices
                - y_pred_names (list): List of predicted movement type names
                - y_pred_proba (np.ndarray): Matrix of prediction probabilities
                  for each class, shape (n_samples, n_classes)
        """
        # Load the trained classifier model
        clf = self.load_model()

        # Get class predictions (as numerical indices)
        y_pred = clf.predict(X)
        # Get probability estimates for each class
        y_pred_proba = clf.predict_proba(X)

        # Load the mapping between numerical indices and movement type names
        event_code_dict = self.load_event_types()
        # Invert the dictionary to map from indices to names
        inv_event_code_dict = {v: k for k, v in event_code_dict.items()}
        # Convert numerical predictions to human-readable movement types
        y_pred_names = [inv_event_code_dict[x] for x in y_pred]
        return y_pred, y_pred_names, y_pred_proba

    def parse_and_predict(self):
        """
        Execute the complete pipeline from raw data to predictions.

        This method runs the entire workflow:
        1. Preprocesses the raw EMG data
        2. Extracts and scales features
        3. Makes predictions using the trained model
        4. Organizes results into a DataFrame

        Returns:
            tuple: Contains two elements:
                - y_pred (np.ndarray): Array of predicted class indices
                - segment_frame (pd.DataFrame): DataFrame containing all processed
                  data and predictions, including:
                    * raw_features: Original extracted features
                    * features: Scaled features
                    * pred: Numerical predictions
                    * pred_names: Human-readable prediction labels
                    * pred_proba: Prediction probabilities
        """
        all_features, feature_names, scaled_features, segment_frame = self.run_pre_process()
        y_pred, y_pred_names, y_pred_proba = self.predict(scaled_features)
        segment_frame['raw_features'] = list(all_features)
        segment_frame['features'] = list(scaled_features)
        segment_frame['pred'] = y_pred
        segment_frame['pred_names'] = y_pred_names
        segment_frame['pred_proba'] = list(y_pred_proba)
        self.segment_frame = segment_frame
        return y_pred, segment_frame, feature_names
