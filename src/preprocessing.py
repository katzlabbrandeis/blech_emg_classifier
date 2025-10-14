"""
Preprocessing Module

This module provides comprehensive functionality for processing EMG signals and
extracting relevant features for movement classification.

Key Components:
    - Movement detection and extraction
    - Signal normalization and standardization
    - Feature extraction including:
        * Duration measurements
        * Amplitude analysis
        * Interval calculations
        * Frequency analysis
        * PCA transformation

The module supports both single-trial and batch processing of EMG data,
with functions organized in a pipeline from raw signal processing to
feature generation.

Functions:
    extract_movements: Detects and extracts movement segments
    normalize_segments: Normalizes movement segments
    extract_features: Generates features from movement segments
    run_AM_process: Runs complete analysis pipeline
    generate_final_features: Prepares final feature set for classification

Dependencies:
    - numpy
    - scipy
    - sklearn
    - pandas
"""

import sys
import os
import numpy as np
from scipy.signal import welch
from scipy.ndimage import white_tophat
from sklearn.decomposition import PCA
import pandas as pd
from sklearn.preprocessing import StandardScaler
from time import time
from pickle import dump, load


def extract_movements(this_trial_dat, size=250):
    """
    Detect and extract movement segments from EMG signal using white tophat filtering.

    Args:
        this_trial_dat (np.array): Raw EMG signal data
        size (int, optional): Size of the structuring element for white tophat. Defaults to 250.

    Returns:
        tuple: Contains:
            - segment_starts (np.array): Indices where movements begin
            - segment_ends (np.array): Indices where movements end
            - segment_dat (list): Raw EMG data for each movement segment
            - filtered_segment_dat (list): Filtered EMG data for each movement segment
    """
    # Apply white tophat filter to enhance peaks and remove baseline
    filtered_dat = white_tophat(this_trial_dat, size=size)

    # Get indices where signal is non-zero after filtering
    segments_raw = np.where(filtered_dat)[0]

    # Create binary mask of movement segments
    segments = np.zeros_like(filtered_dat)
    segments[segments_raw] = 1

    # Find transitions from 0->1 (starts) and 1->0 (ends) using diff
    segment_starts = np.where(np.diff(segments) == 1)[0]
    segment_ends = np.where(np.diff(segments) == -1)[0]
    # If first start is after first end, drop first end
    # and last start
    if segment_starts[0] > segment_ends[0]:
        segment_starts = segment_starts[:-1]
        segment_ends = segment_ends[1:]
    segment_dat = [this_trial_dat[x:y]
                   for x, y in zip(segment_starts, segment_ends)]
    filtered_segment_dat = [filtered_dat[x:y]
                            for x, y in zip(segment_starts, segment_ends)]
    return segment_starts, segment_ends, segment_dat, filtered_segment_dat


def threshold_movement_lengths(
        segment_starts,
        segment_ends,
        segment_dat,
        min_len=50,
        max_len=500):
    """
    Filter movement segments based on their duration.

    Args:
        segment_starts (np.array): Start indices of segments
        segment_ends (np.array): End indices of segments
        segment_dat (list): List of segment data
        min_len (int, optional): Minimum allowed segment length. Defaults to 50.
        max_len (int, optional): Maximum allowed segment length. Defaults to 500.

    Returns:
        tuple: Contains filtered:
            - segment_starts (np.array): Start indices of valid segments
            - segment_ends (np.array): End indices of valid segments
            - segment_dat (list): Data of valid segments
    """
    keep_inds = [x for x, y in enumerate(segment_dat) if len(
        y) > min_len and len(y) < max_len]
    segment_starts = segment_starts[keep_inds]
    segment_ends = segment_ends[keep_inds]
    segment_dat = [segment_dat[x] for x in keep_inds]
    return segment_starts, segment_ends, segment_dat


def normalize_segments(segment_dat):
    """
    Normalize movement segments using min-max scaling and interpolate to fixed length.

    Args:
        segment_dat (list): List of movement segments with varying lengths

    Returns:
        np.array: Array of normalized segments, each with length 100 and
                 values scaled between 0 and 1
    """
    # Find longest segment length (not used but kept for reference)
    max_len = max([len(x) for x in segment_dat])

    # Interpolate each segment to fixed length of 100 points
    # This makes segments comparable regardless of original duration
    interp_segment_dat = [np.interp(
        np.linspace(0, 1, 100),  # Target x-coordinates (0 to 1, 100 points)
        # Source x-coordinates (0 to 1, original length)
        np.linspace(0, 1, len(x)),
        x)  # Source y-coordinates (original signal values)
        for x in segment_dat]

    # Stack segments into 2D array (n_segments x 100)
    interp_segment_dat = np.vstack(interp_segment_dat)

    # Min-max normalization along time axis
    # Subtract minimum of each segment
    interp_segment_dat = interp_segment_dat - \
        np.min(interp_segment_dat, axis=-1)[:, None]
    # Divide by maximum of each segment to get range [0,1]
    interp_segment_dat = interp_segment_dat / \
        np.max(interp_segment_dat, axis=-1)[:, None]
    return interp_segment_dat


def extract_features(
        segment_dat,
        segment_starts,
        segment_ends,
        mean_prestim=None
):
    """
    Function to extract features from a list of segments
    Applied to a single trial

    # Features to extract
    # 1. Duration of movement
    # 2. Amplitude
    # 3. Left and Right intervals (time between consecutive peaks)

    # No need to calculate PCA of time-adjusted waveform at this stage
    # as it's better to do it over the entire dataset
    # 4. PCA of time-adjusted waveform

    # 5. Normalized time-adjusted waveform

    If mean_prestim is provided, also return normalized amplitude
    """
    # Find peak location within each segment
    peak_inds = [np.argmax(x) for x in segment_dat]
    # Convert to absolute time by adding segment start times
    peak_times = [x+y for x, y in zip(segment_starts, peak_inds)]
    # Drop first and last segments because we can't get intervals for them
    segment_dat = segment_dat[1:-1]
    segment_starts = segment_starts[1:-1]
    segment_ends = segment_ends[1:-1]

    durations = [len(x) for x in segment_dat]
    # amplitudes_rel = [np.max(x) - np.min(x) for x in segment_dat]
    amplitude_abs = [np.max(x) for x in segment_dat]
    left_intervals = [peak_times[i] - peak_times[i-1]
                      for i in range(1, len(peak_times))][:-1]
    right_intervals = [peak_times[i+1] - peak_times[i]
                       for i in range(len(peak_times)-1)][1:]

    norm_interp_segment_dat = normalize_segments(segment_dat)

    welch_out = [welch(x, fs=1000, axis=-1) for x in segment_dat]
    max_freq = [x[0][np.argmax(x[1], axis=-1)] for x in welch_out]

    feature_list = [
        durations,
        amplitude_abs,
        left_intervals,
        right_intervals,
        max_freq,
    ]
    feature_names = [
        'duration',
        'amplitude_abs',
        'left_interval',
        'right_interval',
        'max_freq',
    ]

    if mean_prestim is not None:
        amplitude_norm = [x/mean_prestim for x in amplitude_abs]
        feature_list.append(amplitude_norm)
        feature_names.append('amplitude_norm')

    feature_array = np.stack(feature_list, axis=-1)

    return (
        feature_array,
        feature_names,
        segment_dat,
        segment_starts,
        segment_ends,
        norm_interp_segment_dat
    )


def run_AM_process(envs, pre_stim=2000):
    """
    Process EMG envelopes to extract movement segments and features.

    Args:
        envs (np.array): Array of EMG envelopes with shape (tastes, trials, time)
        pre_stim (int, optional): Number of pre-stimulus timepoints. Defaults to 2000.

    Returns:
        tuple: Contains:
            - segment_dat_list (list): List of processed segments containing:
                * feature arrays
                * raw segment data
                * normalized interpolated segments
                * segment bounds
            - feature_names (np.array): Names of extracted features
            - inds (list): List of (taste, trial) index tuples
    """

    this_day_prestim_dat = envs[..., :pre_stim]
    mean_prestim = np.nanmean(this_day_prestim_dat, axis=None)

    segment_dat_list = []
    inds = list(np.ndindex(envs.shape[:-1]))

    # Handling of nans
    inds = [x for x in inds if not any(np.isnan(envs[x]))]

    for this_ind in inds:
        this_trial_dat = envs[this_ind]

        (
            segment_starts,
            segment_ends,
            segment_dat,
            filtered_segment_dat
        ) = extract_movements(
            this_trial_dat, size=200)

        # Threshold movement lengths
        segment_starts, segment_ends, segment_dat = threshold_movement_lengths(
            segment_starts, segment_ends, filtered_segment_dat,
            min_len=50, max_len=500)

        assert len(segment_starts) == len(segment_ends) == len(segment_dat), \
            'Mismatch in segment lengths'
            
        # Features contain baseline-normalized amplitude
        (feature_array,
         feature_names,
         segment_dat,
         segment_starts,
         segment_ends,
         norm_interp_segment_dat,
         ) = extract_features(
            segment_dat, segment_starts, segment_ends, mean_prestim=mean_prestim)

        assert len(feature_array) == len(segment_dat) == len(segment_starts) == len(segment_ends), \
            'Mismatch in feature array lengths'

        segment_bounds = list(zip(segment_starts, segment_ends))
        merged_dat = [feature_array, segment_dat,
                      norm_interp_segment_dat, segment_bounds]
        segment_dat_list.append(merged_dat)

    return segment_dat_list, feature_names, inds


def parse_segment_dat_list(this_segment_dat_list, inds):
    """
    Convert segment data list into a structured pandas DataFrame.

    Args:
        this_segment_dat_list (list): List of lists where each entry contains:
            - features: Extracted feature array
            - segment_raw: Raw segment data
            - segment_norm_interp: Normalized interpolated segments
            - segment_bounds: Start and end indices
        inds (list): List of (taste, trial) tuples identifying each trial

    Returns:
        pd.DataFrame: DataFrame with columns:
            - features: Extracted feature array
            - segment_raw: Raw segment data
            - segment_norm_interp: Normalized interpolated segments
            - segment_bounds: Start and end indices
            - taste: Taste condition
            - trial: Trial number
    """

    # Standardize features
    wanted_data = dict(
        features=[x[0] for x in this_segment_dat_list],
        segment_raw=[x[1] for x in this_segment_dat_list],
        segment_norm_interp=[x[2] for x in this_segment_dat_list],
        segment_bounds=[x[3] for x in this_segment_dat_list],
    )
    gape_frame = pd.DataFrame(wanted_data)
    gape_frame['taste'] = [x[0] for x in inds]
    gape_frame['trial'] = [x[1] for x in inds]
    gape_frame = gape_frame.explode(list(wanted_data.keys()))
    return gape_frame


def generate_final_features(
        all_features,
        feature_names,
        scaled_segments,
        artifact_dir,
        create_new_objs=False
):
    """
    Generate final features for classification

    Inputs:
        all_features (np.array): Array of shape (n_segments, n_features)
        feature_names (np.array): Array of shape (n_features,)
            - Expected features:
                - duration
                - right_interval
                - left_interval
                - max_freq
                - amplitude_abs
                - amplitude_norm
        scaled_segments (np.array): Array of shape (n_segments, n_time)

    Outputs:
        all_features (np.array): Array of shape (n_segments, n_features)
        feature_names (np.array): Array of shape (n_features,)
        scaled_features (np.array): Array of shape (n_segments, n_features)
    """

    pca_save_path = os.path.join(artifact_dir, 'pca_obj.pkl')
    scale_save_path = os.path.join(artifact_dir, 'scale_obj.pkl')

    if create_new_objs:
        pca_obj = PCA(n_components=3)
        scale_obj = StandardScaler()
    else:
        if os.path.exists(pca_save_path) and os.path.exists(scale_save_path):
            print('PCA and scale objects found, loading')
            pca_obj = load(open(pca_save_path, 'rb'))
            scale_obj = load(open(scale_save_path, 'rb'))
        else:
            raise ValueError(
                'PCA and scale object not found and create_new_objs is False')

    # Drop 'amplitude_abs' from features
    drop_inds = [i for i, x in enumerate(
        feature_names) if 'amplitude_abs' in x]
    all_features = np.delete(all_features, drop_inds, axis=1)
    feature_names = np.delete(feature_names, drop_inds)

    # Get PCA features
    if create_new_objs:
        pca_obj.fit(scaled_segments)
    pca_features = pca_obj.transform(scaled_segments)[:, :3]

    # Add PCA features to all features
    all_features = np.concatenate([all_features, pca_features], axis=-1)

    # Scale features
    if create_new_objs:
        scale_obj.fit(all_features)
    scaled_features = scale_obj.transform(all_features)

    # Correct feature_names
    pca_feature_names = ['pca_{}'.format(i) for i in range(3)]
    feature_names = np.concatenate([feature_names, pca_feature_names])

    if artifact_dir is not None and create_new_objs:
        dump(pca_obj, open(pca_save_path, 'wb'))
        dump(scale_obj, open(scale_save_path, 'wb'))

    return all_features, feature_names, scaled_features
