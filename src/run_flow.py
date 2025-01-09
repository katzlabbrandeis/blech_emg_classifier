import os
import sys
from glob import glob
import numpy as np
import pandas as pd
import json
from tqdm import tqdm

from ClassifierHandler import ClassifierHandler, get_paths

artifact_dir, model_save_dir = get_paths()

print(f'Artifact directory: {artifact_dir}')

############################################################
# Get paths to emg data
############################################################

data_dir = '/media/storage/ABU_GC-EMG_Data/emg_process_only'
emg_output_dirs = sorted(glob(os.path.join(data_dir,'*', 'emg_output')))
emg_env_path = glob(os.path.join(emg_output_dirs[0], 'emgad', '*env.npy'))[0] 

############################################################
############################################################

this_handler = ClassifierHandler(
        model_dir = model_save_dir,
        output_dir = artifact_dir,
        env_path = emg_env_path, 
        )
y_pred, segment_frame = this_handler.parse_and_predict()
