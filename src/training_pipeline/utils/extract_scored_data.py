import os
import sys
from glob import glob

import numpy as np
import tables
import pylab as plt
import pandas as pd

# NOTE: return_taste_orders() below requires the original per-session .h5
# file (read via pytables) and is not used by generate_training_dataset.py
# in this repo — the taste order for the bundled sessions is precomputed
# and stored in data/raw/<animal>/<session>/taste_order.csv instead. This
# function is kept for reference / regenerating that file from raw data.

############################################################
############################################################

# Extract dig-ins


def return_taste_orders(h5_files):
    """
    Returns the order of tastes for each session

    Inputs:
        h5_files: list of paths to hdf5 files

    Outputs:
        all_taste_orders: array of shape (days, tastes)
    """

    dig_in_list = []
    for i, h5_file in enumerate(h5_files):
        session_dig_in_list = []
        h5 = tables.open_file(h5_file, 'r')
        for this_dig in h5.root.digital_in._f_iter_nodes():
            session_dig_in_list.append(this_dig[:])
        h5.close()
        dig_in_list.append(session_dig_in_list)

    all_starts = []
    for this_session in dig_in_list:
        session_starts = []
        for this_dig in this_session:
            starts = np.where(np.diff(this_dig) == 1)[0]
            session_starts.append(starts)
        all_starts.append(session_starts)

    all_starts = np.stack(all_starts)
    all_starts = all_starts / 30000
    all_starts = np.round(all_starts).astype(int)

    # Find order of deliveries for each session
    all_taste_orders = []
    for this_session in all_starts:
        bin_array = np.zeros((this_session.max(), len(this_session)))
        for i, this_dig in enumerate(this_session):
            bin_array[this_dig-1, i] = 1
        taste_order = np.where(bin_array)[1]
        all_taste_orders.append(taste_order)
    all_taste_orders = np.array(all_taste_orders)

    return all_taste_orders


############################################################
# Process scoring
############################################################


def process_scored_data(scored_data, taste_orders):
    """
    Processes scored data for a single session

    Inputs:
        scored_data: pandas dataframe of scored data
        taste_orders: array of shape (trials,) with taste order

    Outputs:
        fin_table : pandas dataframe with processed data
    """
    n_trials = scored_data.loc[scored_data.Behavior == 'trial start'].shape[0]

    # Mark absolute trial number
    scored_data.loc[scored_data.Behavior ==
                    'trial start', 'abs_trial'] = np.arange(n_trials)
    # Forward fill 'abs_trial_num' column
    scored_data['abs_trial'] = scored_data['abs_trial'].ffill()
    scored_data.abs_trial = scored_data.abs_trial.astype(int)

    # Add taste_num column
    scored_data['taste_num'] = taste_orders[scored_data.abs_trial.values]

    # Get taste_trial
    taste_order_df = pd.DataFrame(taste_orders, columns=['taste_num'])
    taste_order_df['taste_trial'] = taste_order_df.groupby(
        'taste_num').cumcount()
    taste_order_df['abs_trial'] = np.arange(len(taste_order_df))

    # Merge taste_trial and abs_trial
    scored_data = scored_data.merge(taste_order_df,
                                    on=['taste_num', 'abs_trial'],
                                    how='left')

    # Calculate time from 'trial start'
    scored_data['trial_start'] = scored_data.loc[scored_data.Behavior ==
                                                 'trial start', 'Time']
    scored_data.trial_start = scored_data.trial_start.ffill()
    scored_data['rel_time'] = scored_data.Time - scored_data.trial_start
    scored_data.drop(columns=['trial_start'], inplace=True)

    # Round rel_time to 3 decimal places
    scored_data['rel_time'] = scored_data['rel_time'].round(3)

    # Convert events to start and stop
    scored_data = scored_data.loc[scored_data.Behavior != 'trial start']

    start_table = scored_data.copy()
    stop_table = scored_data.copy()
    start_table = start_table.loc[start_table['Behavior type'] == 'START']
    stop_table = stop_table.loc[stop_table['Behavior type'] == 'STOP']
    start_table.reset_index(inplace=True, drop=True)
    stop_table.reset_index(inplace=True, drop=True)
    start_table['event_num'] = np.arange(len(start_table))
    stop_table['event_num'] = np.arange(len(stop_table))

    merge_cols = start_table.columns.to_numpy()
    diff_cols = ['Time', 'rel_time', 'Behavior type']
    merge_cols = np.setdiff1d(merge_cols, diff_cols)
    # make sure that the merge columns are the same for start and stop
    assert all(start_table[merge_cols] ==
               stop_table[merge_cols]), 'Mismatch in merge columns'
    fin_table = start_table.copy().drop(columns=diff_cols)

    for this_col in diff_cols:
        fin_table[this_col] = list(
            zip(start_table[this_col], stop_table[this_col]))

    return fin_table


############################################################
# Test plots
############################################################
if __name__ == '__main__':
    data_dir = '/home/abuzarmahmood/Desktop/blech_clust/emg/gape_QDA_classifier/_experimental/mouth_movement_clustering/data/NB27'

    # For each day of experiment, load env and table files
    data_subdirs = sorted(glob.glob(os.path.join(data_dir, '*')))
    # Make sure that path is a directory
    data_subdirs = [subdir for subdir in data_subdirs if os.path.isdir(subdir)]
    # Make sure that subdirs are in order
    subdir_basenames = [os.path.basename(
        subdir).lower() for subdir in data_subdirs]

    env_files = [glob.glob(os.path.join(subdir, '*env.npy'))[0]
                 for subdir in data_subdirs]
    # Load env and table files
    # days x tastes x trials x time
    envs = np.stack([np.load(env_file) for env_file in env_files])

    ############################################################
    # Get dig-in info
    ############################################################
    # Extract dig-in from datasets
    raw_data_dir = '/media/fastdata/NB_data/NB27'
    # Find HDF5 files
    h5_files = glob.glob(os.path.join(raw_data_dir, '**', '*', '*.h5'))
    h5_files = sorted(h5_files)
    h5_basenames = [os.path.basename(x) for x in h5_files]
    # Make sure order of h5 files is same as order of envs
    order_bool = [x in y for x, y in zip(subdir_basenames, h5_basenames)]
    if not all(order_bool):
        raise Exception('Bubble bubble, toil and trouble')

    all_taste_orders = return_taste_orders(h5_files)
    fin_table = process_scored_data(data_subdirs, all_taste_orders)

    plot_group = list(fin_table.groupby(['day_ind', 'taste', 'taste_trial']))
    plot_inds = [x[0] for x in plot_group]
    plot_dat = [x[1] for x in plot_group]

    t = np.arange(-2000, 5000)

    event_types = fin_table.event.unique()
    cmap = plt.get_cmap('tab10')
    event_colors = {event_types[i]: cmap(i) for i in range(len(event_types))}

    # Generate custom legend
    from matplotlib.patches import Patch

    legend_elements = [Patch(facecolor=event_colors[event], edgecolor='k',
                             label=event) for event in event_types]

    plot_n = 15
    fig, ax = plt.subplots(plot_n, 1, sharex=True,
                           figsize=(10, plot_n*2))
    for i in range(plot_n):
        this_scores = plot_dat[i]
        this_inds = plot_inds[i]
        this_env = envs[this_inds]
        ax[i].plot(t, this_env)
        for _, this_event in this_scores.iterrows():
            event_type = this_event.event
            start_time = this_event.rel_time_start
            stop_time = this_event.rel_time_stop
            this_event_c = event_colors[event_type]
            ax[i].axvspan(start_time, stop_time,
                          color=this_event_c, alpha=0.5, label=event_type)
    ax[0].legend(handles=legend_elements, loc='upper right',
                 bbox_to_anchor=(1.5, 1.1))
    ax[0].set_xlim([0, 5000])
    fig.subplots_adjust(right=0.75)
    plt.show()
