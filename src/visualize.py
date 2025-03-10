"""
Code for visualization of classifier data

This module provides functions for visualizing EMG signal classification results.
It includes tools for:
- Converting prediction data to visualization-ready formats
- Generating raster plots of movement predictions
- Creating detailed visualizations of EMG signals with overlaid classifications
- Color-coded movement type displays
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from matplotlib.colors import ListedColormap
import matplotlib as mpl
from sklearn.preprocessing import StandardScaler
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
    return ax, im


def generate_raster_plot(
        segments_frame: pd.DataFrame,
        session_name: str = None,
        ):
    """
    Generate a raster plot of the segments frame

    This function creates a visualization of movement classifications across trials.
    Each row represents a trial, and colors indicate the type of movement detected
    at each time point.

    Inputs:
        segments_frame : pd.DataFrame
            DataFrame containing the segments with classification results
        session_name : str
            Name of the session to display in the plot title

    Outputs:
        fig, ax : matplotlib figure and axis objects
    """
    # Check segments_frame has a single basename
    if 'basename' in segments_frame.columns:
        assert len(segments_frame.basename.unique()) == 1

    # Convert segments frame to np arrays by taste
    taste_frame_list = [x[1] for x in segments_frame.groupby('taste')]
    taste_pred_array_list = [return_pred_array(this_taste_frame) for this_taste_frame in taste_frame_list]

    # Generate the plot
    fig, ax = plt.subplots(
            len(taste_pred_array_list), 1,
            sharex=True, sharey=True,
            figsize=(5,10))
    
    # Handle case with only one taste (single subplot)
    if len(taste_pred_array_list) == 1:
        ax = [ax]
        
    for taste in range(len(taste_pred_array_list)):
        ax[taste], im = plot_raster(taste_pred_array_list[taste], ax = ax[taste]) 
        ax[taste].set_ylabel(f'Taste {taste}\nTrial #')
        
    ax[0].set_title('Movement Classification')
    ax[-1].set_xlabel('Time (ms)')
    cbar_ax = fig.add_axes([0.98, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax, shrink=0.5)
    cbar.set_ticks([0.5,1,1.5])
    cbar.set_ticklabels(['nothing','gape','MTMs'])
    plt.tight_layout()
    
    if session_name is not None:
        fig.suptitle(session_name)
    return fig, ax

def generate_detailed_plot(segments_frame, raw_emg=None, trial_idx=0, taste_idx=0):
    """
    Generate a detailed plot showing raw EMG signal and movement classifications.
    
    This function creates a visualization that shows both the raw EMG signal
    and the classified movements for a specific trial, allowing for detailed
    inspection of classification results.
    
    Args:
        segments_frame (pd.DataFrame): DataFrame containing segment information and predictions
        raw_emg (np.ndarray, optional): Raw EMG signal data. If None, only shows classifications.
        trial_idx (int, optional): Trial index to visualize. Defaults to 0.
        taste_idx (int, optional): Taste index to visualize. Defaults to 0.
        
    Returns:
        tuple: (fig, ax) matplotlib figure and axes objects
    """
    # Filter segments for the specified trial and taste
    trial_segments = segments_frame[(segments_frame.trial == trial_idx) & 
                                   (segments_frame.taste == taste_idx)]
    
    # Create figure with two subplots (if raw_emg provided) or one subplot
    if raw_emg is not None:
        fig, ax = plt.subplots(2, 1, figsize=(12, 8), sharex=True, 
                              gridspec_kw={'height_ratios': [1, 3]})
        
        # Get the raw EMG for this trial
        if len(raw_emg.shape) == 3:  # If shape is (tastes, trials, time)
            trial_emg = raw_emg[taste_idx, trial_idx, :]
        else:
            # Assume it's already the right trial or a flattened array
            trial_emg = raw_emg
            
        # Plot raw EMG in bottom subplot
        time_axis = np.arange(len(trial_emg))
        ax[1].plot(time_axis, trial_emg, 'k-', linewidth=0.8)
        ax[1].set_ylabel('EMG Amplitude')
        ax[1].set_xlabel('Time (ms)')
        
        # Use the top subplot for movement classifications
        class_ax = ax[0]
    else:
        fig, class_ax = plt.subplots(1, 1, figsize=(12, 4))
    
    # Set up colors for different movement types
    event_color_map = {
        0: '#D1D1D1',  # Nothing/No movement
        1: '#EF8636',  # Gape
        2: '#3B75AF',  # MTMs
    }
    
    # Plot each segment with its classification
    for _, segment in trial_segments.iterrows():
        start, end = segment.segment_bounds
        pred = segment.pred
        color = event_color_map[pred]
        
        # Plot the segment classification
        class_ax.axvspan(start, end, alpha=0.5, color=color)
        
        # Add text label in the middle of the segment
        mid_point = (start + end) // 2
        class_ax.text(mid_point, 0.5, segment.pred_names, 
                     horizontalalignment='center',
                     verticalalignment='center',
                     transform=class_ax.get_xaxis_transform())
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=event_color_map[0], alpha=0.5, label='Nothing'),
        Patch(facecolor=event_color_map[1], alpha=0.5, label='Gape'),
        Patch(facecolor=event_color_map[2], alpha=0.5, label='MTMs')
    ]
    class_ax.legend(handles=legend_elements, loc='upper right')
    
    # Set labels and title
    class_ax.set_ylabel('Movement Type')
    class_ax.set_title(f'Taste {taste_idx}, Trial {trial_idx} - Movement Classifications')
    
    if raw_emg is None:
        class_ax.set_xlabel('Time (ms)')
    
    plt.tight_layout()
    return fig, ax

if __name__ == "__main__":
    # Example usage - this would normally be loaded from a file
    # Create a sample segments_frame for demonstration
    import pandas as pd
    
    # Sample data - in real usage, load your actual data
    sample_segments = {
        'taste': [0, 0, 0, 1, 1],
        'trial': [0, 0, 1, 0, 0],
        'segment_bounds': [(100, 200), (300, 400), (150, 250), (100, 200), (300, 400)],
        'pred': [1, 2, 1, 2, 1],
        'pred_names': ['gape', 'MTMs', 'gape', 'MTMs', 'gape']
    }
    
    sample_frame = pd.DataFrame(sample_segments)
    
    # Generate visualization from sample data
    taste_frame_list = [x[1] for x in sample_frame.groupby('taste')]
    taste_pred_array_list = [return_pred_array(this_taste_frame) for this_taste_frame in tqdm(taste_frame_list)]

    event_color_map = {
            0 : '#D1D1D1',
            1 : '#EF8636',
            2 : '#3B75AF',
            }
    cmap = ListedColormap(list(event_color_map.values()), name = 'NBT_cmap')

    # Generate the plot
    fig, ax = generate_raster_plot(
            segments_frame = sample_frame,
            session_name = "Example Visualization"
            )
    plt.show()
    
    # Example of detailed plot
    fig, ax = generate_detailed_plot(
            segments_frame = sample_frame,
            trial_idx = 0,
            taste_idx = 0
            )
    plt.show()
