"""
Main script for running EMG signal classification pipeline.

This script loads EMG data from specified directories, processes it through
the classification pipeline, and generates predictions for movement segments.

The pipeline includes:
- Loading EMG envelope data
- Preprocessing and feature extraction
- Model prediction using pre-trained XGBoost classifier

Requirements:
    - Pre-trained model in artifacts/model/
    - Preprocessed EMG data in specified data directory
"""

import os
import sys
from glob import glob
import numpy as np
import pandas as pd
import json
from tqdm import tqdm

from ClassifierHandler import ClassifierHandler, get_paths

# Get paths to model artifacts and output directories
artifact_dir, model_save_dir = get_paths()

print(f'Artifact directory: {artifact_dir}')

# Configure paths to EMG data
# Expected directory structure:
# /media/storage/ABU_GC-EMG_Data/
#   emg_process_only/
#     <subject>/
#       emg_output/
#         emgad/
#           *env.npy (envelope files)
data_dir = '/media/storage/ABU_GC-EMG_Data/emg_process_only'

# Get all EMG output directories
emg_output_dirs = sorted(glob(os.path.join(data_dir, '*', 'emg_output')))

# Get path to first envelope file for initialization
emg_env_path = glob(os.path.join(emg_output_dirs[0], 'emgad', '*env.npy'))[0]

# Initialize classifier handler with model and data paths
this_handler = ClassifierHandler(
    model_dir=model_save_dir,  # Directory containing trained model
    output_dir=artifact_dir,   # Directory for output artifacts
    env_path=emg_env_path,     # Path to EMG envelope data
)

# Run prediction pipeline
# Returns:
# - y_pred: array of predicted class labels
# - segment_frame: DataFrame containing segments and their features
y_pred, segment_frame = this_handler.parse_and_predict()
