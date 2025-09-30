"""
Main script for running EMG signal classification pipeline.

This script loads EMG data from specified directories, processes it through
the classification pipeline, and generates predictions for movement segments.

The pipeline includes:
- Loading EMG envelope data (preprocessed EMG signal amplitude)
- Preprocessing and feature extraction (segmentation, normalization, feature calculation)
- Model prediction using pre-trained XGBoost classifier to identify movement types

Requirements:
    - Pre-trained XGBoost model in artifacts/model/
    - Preprocessed EMG envelope data in specified data directory
    - Required Python packages: numpy, pandas, xgboost, scikit-learn
"""

import os
import sys
from glob import glob
import numpy as np
import pandas as pd
import json
from tqdm import tqdm
import matplotlib.pyplot as plt

from ClassifierHandler import ClassifierHandler, get_paths
from visualize import (
        generate_raster_plot, generate_detailed_plot, plot_env_pred_overlay
        )

# Get paths to model artifacts (PCA, scaler objects) and output directories
# artifacts_dir: Contains PCA, scaler objects, and event code mappings
# model_save_dir: Contains the trained XGBoost model
artifact_dir, model_save_dir = get_paths()

print(f'Artifact directory: {artifact_dir}')

# Configure paths to EMG data
# Expected directory structure:
# /media/storage/ABU_GC-EMG_Data/
#   emg_process_only/          # Root directory for all subjects
#     <subject>/              # Individual subject folders
#       emg_output/           # Processed EMG data for each subject
#         emgad/             # Contains envelope files
#           *env.npy         # Numpy files containing EMG envelopes
#                           # Envelopes are time-series of muscle activation
data_dir = '/media/storage/ABU_GC-EMG_Data/emg_process_only'

# Get all EMG output directories - one per subject
emg_output_dirs = sorted(glob(os.path.join(data_dir, '*', 'emg_output')))

# Get path to first envelope file for initialization
# We use the first subject's data to initialize the classifier
# The same model will be used for all subjects
emg_env_path = glob(os.path.join(emg_output_dirs[0], 'emgad', '*env.npy'))[0]

# Initialize classifier handler with model and data paths
# ClassifierHandler manages the complete pipeline:
# 1. Loads and preprocesses EMG data
# 2. Extracts features from movement segments
# 3. Applies PCA and scaling transformations
# 4. Makes predictions using the pre-trained model
this_handler = ClassifierHandler(
    model_dir=model_save_dir,  # Directory containing trained XGBoost model
    output_dir=artifact_dir,   # Directory for PCA, scaler, and other artifacts
    env_path=emg_env_path,     # Path to EMG envelope data for initialization
)

# Run the complete prediction pipeline
# Returns:
# - y_pred: array of predicted movement types (e.g., 'reach', 'grasp', etc.)
# - segment_frame: DataFrame containing:
#   * Raw movement segments
#   * Extracted features
#   * Timing information
#   * Predicted movement types
y_pred, segment_frame, feature_names = this_handler.parse_and_predict()

# Plot the predictions - overview raster plot
fig, ax = generate_raster_plot(
    segments_frame=segment_frame,
)
plt.show()

# Generate a detailed plot for the first trial of the first taste
# This shows both the classification and timing details
env = this_handler.load_env_file()  # Get the raw EMG data
fig, ax = generate_detailed_plot(
    segments_frame=segment_frame,
    raw_emg=env,
    trial_idx=0,
    taste_idx=0
)
plt.show()

# Generate detailed plots for all trials
fig, ax = plot_env_pred_overlay(
    segments_frame=segment_frame,
    raw_emg=env,
)
plt.show()
