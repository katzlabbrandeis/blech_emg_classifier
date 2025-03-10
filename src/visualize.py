"""
Code for visualization of classifier data
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd
import os
from matplotlib.colors import ListedColormap
import matplotlib as mpl
# import seaborn as sns
from sklearn.preprocessing import StandardScaler
# from umap import UMAP
from sklearn.decomposition import PCA
from tqdm import tqdm

def return_pred_array(taste_frame):
    """
    Given a taste_frame, return a 2D array of predictions
    
    Inputs:
        taste_frame : pd.DataFrame

    Outputs:
        pred_array : np.array
            2D array with shape (n_trials, max_time)
    """

    assert len(taste_frame.taste.unique()) == 1
    if 'basename' in taste_frame.columns:
        assert len(taste_frame.basename.unique()) == 1
    n_trials = taste_frame.trial.max() + 1
    max_time = np.max([x for y in taste_frame.segment_bounds for x in y] )
    # Round up to nearest 1000
    max_time = int(np.ceil(max_time/100) * 100)
    pred_array = np.zeros((n_trials, max_time))
    pred_array[:] = np.nan 
    for _, this_row in taste_frame.iterrows():
        this_trial = this_row.trial
        this_bounds = this_row.segment_bounds
        this_pred = this_row.pred
        pred_array[this_trial, this_bounds[0]:this_bounds[1]] = this_pred
    return pred_array

taste_frame_list = [x[1] for x in segment_frame.groupby('taste')]
taste_pred_array_list = [return_pred_array(this_taste_frame) for this_taste_frame in tqdm(taste_frame_list)]

event_color_map = {
        0 : '#D1D1D1',
        1 : '#EF8636',
        2 : '#3B75AF',
        }
cmap = ListedColormap(list(event_color_map.values()), name = 'NBT_cmap')

def generate_raster_plot(
        segments_frame: pd.DataFrame,

    cols = segments_frame

##############################

def plot_raster(
        pred_array,
        ax = None,
        cmap = None,
    ):
    """
    Plot a raster of predictions

    Inputs:
        pred_array : np.array, shape (n_trials, max_time)
        ax : matplotlib axis
        cmap : matplotlib colormap

    Outputs:
        ax : matplotlib axis
    """
    if ax is None:
        fig, ax = plt.subplots()
    if cmap is None:
        event_color_map = {
                0 : '#D1D1D1',
                1 : '#EF8636',
                2 : '#3B75AF',
                }
        cmap = ListedColormap(list(event_color_map.values()), name = 'NBT_cmap')

    max_trials = pred_array.shape[0]
    x_vec = np.arange(pred_array.shape[1])
    im = ax.pcolormesh(
            x_vec, np.arange(max_trials), 
            pred_array, 
              cmap=cmap,vmin=0,vmax=2,)
    ax.set_ylabel(f'{taste}' + '\nTrial #')
    ax.axvline(2000, color = 'r', linestyle = '--')
    return ax, im

fig, ax = plt.subplots(
    len(taste_pred_array_list), 1,
    sharex=True, sharey=True,
    figsize=(5,10))
for taste in range(4):
    ax[taste], im = plot_raster(taste_pred_array_list[taste], ax = ax[taste], cmap = cmap)
ax[0].set_title('XGB')
ax[-1].set_xlabel('Time (ms)')
# cbar_ax = fig.add_axes([0.98, 0.15, 0.02, 0.7])
cbar = fig.colorbar(im, )#cax=cbar_ax)
cbar.set_ticks([0.5,1,1.5])
cbar.set_ticklabels(['nothing','gape','MTMs'])
plt.tight_layout()
# plt.subplots_adjust(top=0.9)
# fig.suptitle(session_name)
plt.show()

# fig.savefig(os.path.join(xgb_pred_plot_dir, session_name + '_xgb_bsa_pred_test.png'),
#             bbox_inches='tight', dpi = 300)
# plt.close(fig)
