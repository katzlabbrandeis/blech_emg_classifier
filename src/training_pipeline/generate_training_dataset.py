"""
Streamlined script to generate training dataset for the model.
"""
############################################################
############################################################
# get_data.py
############################################################
############################################################

import os
import sys
from glob import glob

import numpy as np
import pandas as pd
from tqdm import tqdm
import json
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NeighborhoodComponentsAnalysis
from scipy.spatial.distance import mahalanobis
from pickle import dump, load

from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from umap import UMAP
import matplotlib.pyplot as plt

this_dir = os.path.dirname(os.path.abspath(__file__))  # src/training_pipeline
repo_root = os.path.abspath(os.path.join(this_dir, '..', '..'))
sys.path.append(this_dir)  # noqa
from utils.extract_scored_data import process_scored_data  # noqa
from utils.gape_clust_funcs import (extract_movements,
                                            normalize_segments,
                                            extract_features,
                                            JL_process,
                                            # gen_segment_frame,
                                            parse_segment_dat_list,
                                            parse_gapes_Li,
                                            threshold_movement_lengths,
                                            )  # noqa

artifact_dir = os.path.join(this_dir, 'artifacts')
if not os.path.isdir(artifact_dir):
    os.makedirs(artifact_dir)

plot_dir = os.path.join(artifact_dir, 'pipeline_test_plots')
if not os.path.isdir(plot_dir):
    os.makedirs(plot_dir)

############################################################
# Helper functions
############################################################


def get_scored_data_paths(data_dir):
    """
    Get scored data from data_dir

    Inputs:
        data_dir (str): Directory containing data

    Outputs:
        scores_path_df (pd.DataFrame): Dataframe containing paths to scored data
    """
    scored_data_paths = glob(os.path.join(
        data_dir, '**', '*scores.csv'), recursive=True)
    scored_data_basenames = [os.path.basename(x).lower().split('.')[
        0] for x in scored_data_paths]
    scored_data_basenames = ["_".join(x.split('_')[:-1])
                             for x in scored_data_basenames]

    scores_path_df = pd.DataFrame({'path': scored_data_paths,
                                   'basename': scored_data_basenames,
                                   'scores': True})
    return scores_path_df


def get_emg_paths(data_dir):
    """
    Get emg paths from data_dir

    Expects the layout used in this repo's bundled data:
        data_dir/raw/<ANIMAL>/<basename>/emg_env.npy
        data_dir/raw/<ANIMAL>/<basename>/<basename>.info
        data_dir/raw/<ANIMAL>/<basename>/taste_order.csv

    `taste_order.csv` is a precomputed replacement for the taste-delivery
    order that would otherwise be derived from the session's raw .h5 file
    (see utils/extract_scored_data.py:return_taste_orders) — the multi-GB
    .h5 files themselves are not part of this repo.

    Inputs:
        data_dir (str): Directory containing data

    Outputs:
        emg_path_df (pd.DataFrame): Dataframe containing paths to emg data,
            info files, and precomputed taste orders
    """

    session_dirs = sorted(glob(os.path.join(data_dir, 'raw', '*', '*')))
    session_dirs = [x for x in session_dirs if os.path.isdir(x)]

    basenames = [os.path.basename(x).lower() for x in session_dirs]
    info_file_paths = [glob(os.path.join(x, '*.info'))[0]
                       for x in session_dirs]
    taste_order_paths = [os.path.join(x, 'taste_order.csv')
                         for x in session_dirs]

    emg_path_df = pd.DataFrame({'path': session_dirs,
                                'basename': basenames,
                                'info_file_path': info_file_paths,
                                'taste_order_path': taste_order_paths,
                                'emg': True})

    return emg_path_df


def load_raw_data(session_data_df):
    """
    Given session_data_df, use paths to load:
        1- scored data
        2- raw emg data
        3- taste orders (precomputed; see data/README.md)
        4- info file

    Inputs:
        session_data_df (pd.DataFrame): Dataframe containing paths to data

    Outputs:
        session_data_df (pd.DataFrame): Dataframe with loaded data
    """

    # Unnecessary columns in scored data to delete
    scored_cols_to_delete = [
        'Observation id', 'Observation date', 'Description',
        'Observation duration', 'Observation type',
        'Source', 'Media duration (s)', 'FPS', 'Subject',
        'Behavioral category', 'Media file name', 'Image index',
        'Image file path', 'Comment',
    ]

    scored_data_list = []
    envs_list = []
    taste_map_list = []
    taste_order_list = []
    for ind, row in tqdm(session_data_df.iterrows()):

        ###############
        # scored_data_path = scored_data_paths[ind]
        scored_data_path = row.path_scores
        scored_data = pd.read_csv(scored_data_path)
        scored_data.drop(columns=scored_cols_to_delete, inplace=True)
        # Indicate these scores were from video to differentiate from emg only
        scored_data['scoring_type'] = 'video'
        scored_data_list.append(scored_data)

        ###############
        # tastes x trials x time
        env_path = glob(os.path.join(row.path_emg, '*env.npy'))[0]
        env = np.load(env_path)
        envs_list.append(env)

        ###############
        info_file_path = row.info_file_path
        info_file = json.load(open(info_file_path, 'r'))
        taste_names = info_file['taste_params']['tastes']
        pal_rankings = info_file['taste_params']['pal_rankings']
        taste_map = dict(zip(taste_names, pal_rankings))
        taste_map_list.append(taste_map)

        ###############
        # Taste are delivered in a pseudorandom order; the order for each
        # session was precomputed from the session's raw .h5 file (see
        # data/README.md) and is stored as a small CSV alongside the
        # envelope/info files.
        taste_order_df = pd.read_csv(row.taste_order_path)
        taste_orders = np.array([taste_order_df['taste'].values])
        taste_order_list.append(taste_orders)

    ############################################################
    # Append to session_data_df and drop any rows with None
    session_data_df['scored_data'] = scored_data_list
    session_data_df['env'] = envs_list
    session_data_df['taste_orders'] = taste_order_list
    session_data_df['taste_map'] = taste_map_list

    # session_data_df.basename[merge_df.taste_orders.isna()]
    session_data_df.dropna(inplace=True)
    merge_keep_inds = session_data_df.index
    session_data_df.reset_index(inplace=True, drop=True)

    return session_data_df


def run_JL_process(envs, pre_stim=2000, post_stim=5000):
    """
    Run Jenn Li process on envs

    Inputs:
        envs (np.array): Array of shape (tastes, trials, time)
         - Contains enveloped emg data
        pre_stim (int): Time before stimulus onset
         - Used to calculate baseline
        post_stim (int): Time after stimulus onset

    Outputs:
        gapes_Li (np.array): Array of shape (tastes, trials, time)
         - Binary array indicating predicted gapes
    """
    gapes_Li = np.zeros(envs.shape)
    inds = list(np.ndindex(envs.shape[:-1]))
    this_day_prestim_dat = envs[..., :pre_stim]

    for this_ind in inds:
        this_trial_dat = envs[this_ind]

        ### Jenn Li Process ###
        # Get peak indices
        gape_peak_inds, time_list = JL_process(
            this_trial_dat,
            this_day_prestim_dat,
            pre_stim,
            post_stim,
        )

        if gape_peak_inds is not None:
            gapes_Li[this_ind][gape_peak_inds] = 1

    return gapes_Li


def run_AM_process(envs, pre_stim=2000):
    """
    Run AM process on envs from a single session

    Inputs:
        envs (np.array): Array of shape (tastes, trials, time)

    Outputs:
        segment_dat_list (list): List of segment data
        feature_names (np.array): Array of feature names
        inds (list): List of indices
    """
    this_day_prestim_dat = envs[..., :pre_stim]
    mean_prestim = np.mean(this_day_prestim_dat, axis=None)

    segment_dat_list = []
    inds = list(np.ndindex(envs.shape[:-1]))

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
    pca_obj.fit(scaled_segments)
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

    if artifact_dir is not None:
        dump(pca_obj, open(pca_save_path, 'wb'))
        dump(scale_obj, open(scale_save_path, 'wb'))

    return all_features, feature_names, scaled_features


if __name__ == '__main__':
    ############################################################
    # Make sure we have 1) scored data and 2) emg env/info/taste-order files
    # for each session (see data/README.md for the expected layout)
    ############################################################

    data_dir = os.path.join(repo_root, 'data')
    scores_path_df = get_scored_data_paths(data_dir)
    emg_path_df = get_emg_paths(data_dir)

    ##############################
    # Merge dataframes
    # Keep all data in a "session_level" dataframe for organization
    session_data_df = pd.merge(scores_path_df, emg_path_df, on='basename',
                               suffixes=('_scores', '_emg'),
                               how='outer')

    # Add more identifying information to session_data_df
    session_data_df['animal_num'] = [
        x.split('_')[0] for x in session_data_df.basename.values]
    session_data_df['session_ind'] = session_data_df.index

    # Drop any rows with NaN
    session_data_df.dropna(inplace=True)
    session_data_df.reset_index(inplace=True, drop=True)

    ############################################################
    # Load raw data
    ############################################################
    session_data_df = load_raw_data(session_data_df)

    # Confirm that there are as many 'trial start's in scored data
    # as trials in the ephys data
    # This is needed to ensure scoring and emg can be aligned later
    for ind, row in session_data_df.iterrows():
        scored_data = row.scored_data
        taste_orders = row.taste_orders[0]
        n_trial_starts = len(
            scored_data.loc[scored_data.Behavior == 'trial start'])
        n_trials = len(taste_orders)
        if n_trials != n_trial_starts:
            print('Mismatch in trials for {}'.format(row.basename))

    ############################################################
    # Add scored data to session_data_df
    ############################################################
    fin_score_table_list = []
    for ind, row in tqdm(session_data_df.iterrows()):
        scored_data = session_data_df.loc[ind, 'scored_data'].copy()
        taste_orders = session_data_df.loc[ind, 'taste_orders'][0].copy()
        # Convert scored data into more amenable format
        # See artifacts/explanatory_plots/scored_data_to_processed_score_table.png
        fin_scored_table = process_scored_data(scored_data, taste_orders)
        fin_score_table_list.append(fin_scored_table)

    ##############################
    # Also add nothing scores from emg only
    ##############################

    nothing_labels_path = os.path.join(artifact_dir, 'nothing_labels.csv')
    nothing_labels = pd.read_csv(nothing_labels_path)
    nothing_labels_ind_path = os.path.join(
        artifact_dir, 'nothing_label_inds.npy')
    nothing_labels_inds = np.load(nothing_labels_ind_path)

    # Break down by session
    nothing_selected_samples = nothing_labels.Abu.values
    fin_nothing_inds = nothing_labels_inds[nothing_selected_samples]
    fin_nothing_inds_df = pd.DataFrame(fin_nothing_inds, columns=[
                                       'session', 'taste_num', 'taste_trial'])
    nothing_groups_inds, nothing_groups_dfs = zip(
        *list(fin_nothing_inds_df.groupby('session')))

    ##############################

    fin_scored_data_list = []
    for this_session in range(len(fin_score_table_list)):
        this_scored_data = fin_score_table_list[this_session]
        if this_session in nothing_groups_inds:
            which_nothing_group = np.where(
                np.array(nothing_groups_inds) == this_session)[0][0]
            this_nothing_df = nothing_groups_dfs[which_nothing_group]

            nothing_frame_list = []
            for ind, row in this_nothing_df.iterrows():
                this_taste = row.taste_num
                this_trial = row.taste_trial
                this_nothing_frame = pd.DataFrame({
                    'Behavior': ['nothing'],
                    'scoring_type': ['emg'],
                    'taste_num': [this_taste],
                    'taste_trial': [this_trial],
                    'rel_time': [(-2, -1)],
                })
                nothing_frame_list.append(this_nothing_frame)
            this_nothing_frame = pd.concat(nothing_frame_list)
            this_scored_data = pd.concat(
                [this_scored_data, this_nothing_frame])
            this_scored_data.reset_index(inplace=True, drop=True)
            fin_scored_data_list.append(this_scored_data)
        else:
            fin_scored_data_list.append(this_scored_data)

    ##############################

    session_data_df['scored_data'] = fin_scored_data_list

    ############################################################
    # At this stage, all raw data has been loaded
    # Following steps are for feature extraction + engineering
    ############################################################

    ############################################################
    # Extract mouth movements
    ############################################################
    pre_stim = 2000
    post_stim = 5000

    ###############
    # Run everything through JL Process
    gapes_Li_list = [run_JL_process(row.env, pre_stim, post_stim)
                     for ind, row in tqdm(session_data_df.iterrows())]
    session_data_df['gapes_Li'] = gapes_Li_list

    ###############
    # Run everything through AM Process

    # run_AM_process returns:
    # segment_dat_list, feature_names, norm_interp_segment_dat
    segment_dat_list_raw = [run_AM_process(
        row.env) for ind, row in tqdm(session_data_df.iterrows())]
    feature_names = segment_dat_list_raw[0][1]
    # Segment_dat_list:
    # Sessions
    #  - Trials
    #    - feature_array, segment_dat, norm_interp_segment_dat, segment_bounds
    segment_dat_list = [x[0] for x in segment_dat_list_raw]
    session_data_df['segment_dat_list'] = segment_dat_list

    # Note, session_data_df is a "session level" frame
    # Columns:
    # ['path_scores', 'basename', 'scores', 'path_emg', 'info_file_path',
    #        'taste_order_path', 'emg', 'animal_num', 'session_ind',
    #        'scored_data', 'env', 'taste_orders', 'taste_map', 'gapes_Li',
    #        'segment_dat_list']
    # Shape:
    # (n_sessions, 18)

    # We now have to extract data for individual segments and generate a gape frame

    ##############################
    # Generate gape frame
    # Extract features returns:
    # feature_array, feature_names, segment_dat, segment_starts, segment_ends
    # Convert these to a dataframe using gen_segment_frame

    # segment_frame_list = [gen_segment_frame(row.segment_dat_list, row.gapes_Li)
    #                    for ind, row in tqdm(session_data_df.iterrows())]

    # First pass to parse segment_dat_list
    segment_frame_list = []
    for _, this_row in tqdm(session_data_df.iterrows()):
        this_dat_list = this_row.segment_dat_list
        this_env = this_row.env
        inds = list(np.ndindex(this_env.shape[:-1]))
        segment_frame_list.append(
            parse_segment_dat_list(this_dat_list, inds))

    # segment_frame_list = [parse_segment_dat_list(row.segment_dat_list) \
    #                    for ind, row in tqdm(session_data_df.iterrows())]
    # Second pass to add gapes_Li
    segment_frame_list = [parse_gapes_Li(this_Li, segment_frame)
                          for this_Li, segment_frame in tqdm(zip(session_data_df.gapes_Li, segment_frame_list))]

    # Bounds for segment_frame are in 0-7000 time
    # Adjust to make -2000 -> 5000
    # Adjust segment_bounds by removing pre_stim
    for segment_frame in segment_frame_list:
        all_segment_bounds = segment_frame.segment_bounds.values
        adjusted_segment_bounds = [
            np.array(x)-pre_stim for x in all_segment_bounds]
        segment_frame['segment_bounds'] = adjusted_segment_bounds

        # Calculate segment centers
        segment_frame['segment_center'] = [
            np.mean(x)/1000 for x in segment_frame.segment_bounds]

    # Save segment_frame with both raw and adjusted bounds
    session_data_df['segment_frame'] = segment_frame_list
    segment_frame_raw_list = segment_frame_list.copy()
    session_data_df['segment_frame_raw'] = segment_frame_raw_list

    ##############################

    # Make sure that each segment in segment_frame is in fin_score_table
    # That is, all extracted segments are scored
    # If not, remove the segment from segment_frame
    segment_frame_list = []
    for ind, row in tqdm(session_data_df.iterrows()):
        segment_frame = row.segment_frame
        segment_frame['scored'] = False
        scored_data = row.scored_data
        scored_data.rename(
            columns={'rel_time': 'segment_bounds'}, inplace=True)

        score_bounds_list = []
        for event_ind, event_row in tqdm(segment_frame.iterrows()):
            taste = event_row.taste
            taste_trial = event_row.trial
            segment_center = event_row.segment_center
            wanted_score_table = scored_data.loc[
                (scored_data.taste_num == taste) &
                (scored_data.taste_trial == taste_trial)]
            if len(wanted_score_table):
                # Check if segment center is in any of the scored segments
                for _, score_row in wanted_score_table.iterrows():
                    min_bool = score_row.segment_bounds[0] <= segment_center
                    max_bool = segment_center <= score_row.segment_bounds[1]
                    if min_bool & max_bool:
                        segment_frame.loc[event_ind, 'scored'] = True
                        segment_frame.loc[event_ind,
                                          'event_type'] = score_row.Behavior
                        score_bounds_list.append(
                            np.array(score_row.segment_bounds)*1000)
                        break

        segment_frame = segment_frame.loc[segment_frame.scored]
        segment_frame['score_bounds'] = score_bounds_list
        segment_frame_list.append(segment_frame)

    session_data_df['segment_frame'] = segment_frame_list

    # scored_segment_frame.to_pickle(os.path.join(code_dir, 'data', 'scored_segment_frame.pkl'))
    session_data_df.to_pickle(os.path.join(
        artifact_dir, 'session_data_df.pkl'))

    ############################################################
    # Preprocessing
    ############################################################
    # Get all gape frames

    fin_segment_frames = []
    for i, (_, this_row) in enumerate(session_data_df.iterrows()):
        this_frame = this_row.segment_frame_raw
        this_frame['session_ind'] = i
        this_frame['animal_num'] = this_row.animal_num
        this_frame['basename'] = this_row.basename
        fin_segment_frames.append(this_frame)
    cat_segment_frame = pd.concat(
        fin_segment_frames, axis=0).reset_index(drop=True)

    # Match gape frames with palatability
    pal_dicts = [{i: x for i, x in enumerate(this_dict.items())}
                 for this_dict in session_data_df.taste_map.values]
    pal_frames = [pd.DataFrame.from_dict(x, orient='index', columns=['taste_name', 'pal'])
                  for x in pal_dicts]
    pal_frames = [x.reset_index(drop=False) for x in pal_frames]
    pal_frames = [x.rename(columns={'index': 'taste'}) for x in pal_frames]

    fin_pal_frames = []
    for i, this_frame in enumerate(pal_frames):
        this_frame['session_ind'] = i
        this_frame['animal_num'] = session_data_df.animal_num.values[i]
        this_frame['basename'] = session_data_df.basename.values[i]
        fin_pal_frames.append(this_frame)
    cat_pal_frame = pd.concat(fin_pal_frames, axis=0).reset_index(drop=True)

    cat_segment_frame = pd.merge(
        cat_segment_frame,
        cat_pal_frame,
        on=['session_ind', 'taste', 'animal_num', 'basename'],
        how='inner')

    ############################################################
    # Generate final features
    ############################################################
    all_features = np.stack(cat_segment_frame.features.values)
    scaled_segments = np.stack(cat_segment_frame.segment_norm_interp.values)
    all_features, feature_names, scaled_features = \
        generate_final_features(all_features, feature_names, scaled_segments,
                                artifact_dir=artifact_dir,
                                create_new_objs=True)

    ##############################
    cat_segment_frame['raw_features'] = list(all_features)
    cat_segment_frame['features'] = list(scaled_features)

    ############################################################
    # Continue only with labelled data
    ############################################################
    scored_df = cat_segment_frame[cat_segment_frame.scored == True]

    # Correct event_types
    types_to_drop = ['to discuss', 'other',
                     'unknown mouth movement', 'out of view']
    scored_df = scored_df[~scored_df.event_type.isin(types_to_drop)]

    # Remap event_types
    event_type_map = {
        'mouth movements': 'mouth or tongue movement',
        'tongue protrusion': 'mouth or tongue movement',
        'mouth or tongue movement': 'mouth or tongue movement',
        'lateral tongue movement': 'lateral tongue protrusion',
        'lateral tongue protrusion': 'lateral tongue protrusion',
        'gape': 'gape',
        'no movement': 'no movement',
        'nothing': 'no movement',
    }

    scored_df['event_type'] = scored_df['event_type'].map(event_type_map)
    scored_df['event_codes'] = scored_df['event_type'].astype(
        'category').cat.codes
    scored_df['is_gape'] = (scored_df['event_type'] == 'gape')*1

    scored_df.dropna(subset=['event_type'], inplace=True)

    ############################################################
    # Expand no_movement category
    ############################################################
    features = np.stack(scored_df.features.values)
    event_codes = scored_df.event_codes.values
    event_code_map = scored_df[['event_codes', 'event_type']].drop_duplicates()

    ##############################
    # UMAP / NCA plot to visualize distinction of clusters by event type
    nca_obj = NeighborhoodComponentsAnalysis(n_components=2)
    nca_features = nca_obj.fit_transform(features, event_codes)

    ##############################
    # Fit a GMM to 3D NCA of 'no movement' data
    # and assign all points within 3 mahalanobis distances
    # as no movement

    # Get no movement data
    no_movement_code = event_code_map[event_code_map.event_type ==
                                      'no movement'].event_codes.values[0]
    no_movement_inds = np.where(event_codes == no_movement_code)
    no_movement_3d_nca = nca_features[no_movement_inds]

    # Fit gaussian
    mean_vec = np.mean(no_movement_3d_nca, axis=0)
    cov_mat = np.cov(no_movement_3d_nca, rowvar=False)

    # Calculate mahalanobis distance
    mahal_dist = np.array(
        [mahalanobis(x, mean_vec, np.linalg.inv(cov_mat)) for x in nca_features])
    mahal_thresh = 2

    updated_codes = event_codes.copy()
    updated_codes[mahal_dist < mahal_thresh] = no_movement_code

    # Update codes in scored_df
    scored_df['updated_codes'] = updated_codes
    scored_df['updated_event_type'] = scored_df.updated_codes.map(
        event_code_map.set_index('event_codes').event_type)

    # Plot
    fig, ax = plt.subplots(1, 2, sharex=True, sharey=True,)
    for event_type in np.unique(event_codes):
        inds = np.where(event_codes == event_type)
        ax[0].scatter(nca_features[inds, 0], nca_features[inds, 1], label=event_type,
                      alpha=0.5, s=5)
    for event_type in np.unique(updated_codes):
        inds = np.where(updated_codes == event_type)
        ax[1].scatter(nca_features[inds, 0], nca_features[inds, 1], label=event_type,
                      alpha=0.5, s=5)
    ax[0].legend(title='Event Type')
    ax[0].set_xlabel('NCA 1')
    ax[0].set_ylabel('NCA 2')
    ax[1].set_xlabel('NCA 1')
    ax[0].set_title('Original Event Type')
    ax[1].set_title('Updated Event Type')
    plt.suptitle('NCA of Features Colored by Event Type')
    plt.tight_layout()
    fig.savefig(os.path.join(plot_dir, 'scored_df_nca.png'),
                bbox_inches='tight', dpi=300)
    plt.close(fig)

    # Over-write with updated event coes and types
    scored_df['event_type'] = scored_df['updated_event_type']

    scored_df.dropna(subset=['event_type'], inplace=True)

    # Remove lateral tongue protrusion
    scored_df = scored_df[scored_df.event_type != 'lateral tongue protrusion']

    # Abbreviate mouth or tongue movement
    scored_df['event_type'] = scored_df['event_type'].replace(
        ['mouth or tongue movement'],
        'MTMs'
    )

    # Update event codes to match with BSA
    event_code_dict = {
        'gape': 1,
        'MTMs': 2,
        'no movement': 0,
    }
    # Save event_code_dict
    event_code_dict_path = os.path.join(artifact_dir, 'event_code_dict.json')
    with open(event_code_dict_path, 'w') as f:
        json.dump(event_code_dict, f)

    scored_df['event_codes'] = scored_df['event_type'].map(event_code_dict)

    # Drop unnecessary columns
    scored_df.drop(columns=['scored', 'updated_codes',
                            'updated_event_type'], inplace=True)

    # Save scored_df
    training_dataset_dir = os.path.join(repo_root, 'data', 'training')
    if not os.path.isdir(training_dataset_dir):
        os.makedirs(training_dataset_dir)
    scored_df.to_pickle(os.path.join(
        training_dataset_dir, 'fin_training_dataset.pkl'))

    ############################################################
    # Plotting to check process quality
    ############################################################

    bsa_event_map = {
        0: 'no movement',
        1: 'gape',
        2: 'MTMs',
    }
    event_color_map = {
        0: '#D1D1D1',
        1: '#EF8636',
        2: '#3B75AF',
    }
    inv_bsa_event_map = {v: k for k, v in bsa_event_map.items()}

    # scored_df.reset_index(inplace=True)
    scored_df['animal_code'] = scored_df.animal_num.astype(
        'category').cat.codes
    scored_df['session_code'] = scored_df.basename.astype('category').cat.codes

    cmap = ListedColormap(list(event_color_map.values()), name='NBT_cmap')

    ############################################################
    # 1- Features UMAP colored by predicted class
    # Create UMAP
    scored_df = scored_df.sort_values(
        ['event_codes', 'animal_code', 'session_code'])
    feature_array = np.stack(scored_df.features.values)
    event_codes = scored_df.event_codes.values
    # Clip at +/- 3
    # feature_array = np.clip(feature_array, -3, 3)

    umap = UMAP(n_components=2)
    # umap = PCA(n_components=2)
    umap.fit(feature_array)
    X_umap = umap.transform(feature_array)

    # plt.imshow(umap.components_, interpolation='none', cmap='viridis')
    # plt.show()

    # Plot
    fig, ax = plt.subplots()
    for event_type in np.unique(event_codes):
        inds = np.where(event_codes == event_type)
        ax.scatter(X_umap[inds, 0], X_umap[inds, 1], label=bsa_event_map[event_type],
                   alpha=0.5, s=5)
    ax.legend(title='Predicted Class')
    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')
    plt.title('UMAP of Features Colored by Predicted Class')
    fig.savefig(os.path.join(plot_dir, 'scored_df_umap.png'),
                bbox_inches='tight', dpi=300)
    plt.close(fig)

    ############################################################
    # 3- Heatmap of features per class and session
    fig, ax = plt.subplots(1, 5, sharey=True, sharex='col',
                           figsize=(20, 5))
    # Sort by prediction
    animal_codes = np.stack(scored_df.animal_code.values)
    session_codes = np.stack(scored_df.session_code.values)
    ax[0].imshow(X_umap, aspect='auto', cmap='viridis',
                 interpolation='none')
    ax[1].imshow(feature_array, aspect='auto', cmap='viridis', vmin=-3, vmax=3,
                 interpolation='none')
    ax[1].set_xlabel('Feature #')
    ax[1].set_xticks(np.arange(len(feature_names)))
    ax[1].set_xticklabels(feature_names, rotation=90)
    ax[2].imshow(event_codes[:, None], aspect='auto', cmap=cmap)
    # Generate legend
    legend_elements = [Patch(facecolor=event_color_map[x],
                             label=bsa_event_map[x]) for x in np.unique(event_codes)]
    ax[2].legend(handles=legend_elements, title='Scored Class')
    ax[3].imshow(animal_codes[:, None], aspect='auto', cmap='tab20')
    ax[4].imshow(session_codes[:, None], aspect='auto', cmap='tab20')
    ax[0].set_title('PCA Features')
    ax[1].set_title('Feature Heatmap')
    ax[2].set_title('Predicted Class')
    ax[3].set_title('Animal Code')
    ax[4].set_title('Session Code')
    fig.savefig(os.path.join(plot_dir, 'scored_heatmap_ind.png'),
                bbox_inches='tight', dpi=300)
    plt.close(fig)
