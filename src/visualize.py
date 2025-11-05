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
from matplotlib.patches import Patch
from sklearn.preprocessing import StandardScaler
from scipy.stats import median_abs_deviation
from tqdm import tqdm, trange

def return_pred_array(taste_frame):
    """
    Convert a DataFrame of segment predictions into a 2D time-series array.

    This function takes a DataFrame containing movement segment predictions and
    converts it into a 2D array where each row represents a trial and each column
    represents a time point. Predictions are filled in for the duration of each
    segment, with NaN values elsewhere.

    Args:
        taste_frame (pd.DataFrame): DataFrame containing segment data for a single taste.
                                   Must have columns:
                                   - 'taste': Taste identifier (must be unique)
                                   - 'trial': Trial number
                                   - 'segment_bounds': Tuple of (start, end) indices
                                   - 'pred': Predicted movement class
                                   Optional:
                                   - 'basename': Session identifier (must be unique if present)

    Returns:
        np.array: 2D array with shape (n_trials, max_time) where:
                 - n_trials: Maximum trial number + 1
                 - max_time: Rounded up to nearest 100 samples
                 - Values are prediction classes during segments, NaN elsewhere

    Raises:
        AssertionError: If taste_frame contains multiple tastes or basenames
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
    Create a color-coded raster plot of movement predictions over time.

    This function generates a heatmap-style visualization where each row represents
    a trial and colors indicate the type of movement at each time point.

    Args:
        pred_array (np.array): Array of shape (n_trials, max_time) containing
                              prediction values:
                              - 0: No movement (gray)
                              - 1: Gape (orange)
                              - 2: MTMs/Mouth/Tongue Movements (blue)
                              - NaN: No data
        ax (matplotlib.axes.Axes, optional): Matplotlib axis to plot on.
                                            If None, creates new figure and axis.
                                            Defaults to None.
        cmap (matplotlib.colors.Colormap, optional): Custom colormap for predictions.
                                                     If None, uses default NBT colormap.
                                                     Defaults to None.

    Returns:
        tuple: Contains two elements:
            - ax (matplotlib.axes.Axes): The axis with the raster plot
            - im (matplotlib.collections.QuadMesh): The pcolormesh image object
                                                   for colorbar creation
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

def plot_env_pred_overlay(
        segments_frame,
        raw_emg,
        cmap = None,
        mad_scale = 5,
        ):
    """
    Create a comprehensive grid plot showing raw EMG signals with overlaid predictions.

    This function generates a multi-panel visualization where each subplot shows
    the raw EMG signal for a specific trial and taste, with color-coded overlays
    indicating the predicted movement types during each segment.

    Args:
        segments_frame (pd.DataFrame): DataFrame containing segment information with columns:
                                      - 'taste': Taste condition index
                                      - 'trial': Trial number
                                      - 'segment_bounds': Tuple of (start, end) indices
                                      - 'pred': Predicted movement class (0=nothing, 1=gape, 2=MTMs)
                                      Optional:
                                      - 'basename': Session identifier (must be unique if present)
        raw_emg (np.array): Array of shape (n_tastes, n_trials, n_timepoints) containing
                           raw EMG signal amplitudes
        cmap (matplotlib.colors.Colormap, optional): Custom colormap for movement types.
                                                     If None, uses default color scheme:
                                                     - Gray (#D1D1D1): No movement
                                                     - Orange (#EF8636): Gape
                                                     - Blue (#3B75AF): MTMs
                                                     Defaults to None.
        mad_scale (float, optional): Scaling factor for median absolute deviation

    Returns:
        tuple: Contains two elements:
            - fig (matplotlib.figure.Figure): The complete figure object
            - ax (np.array): 2D array of matplotlib axes with shape (n_trials, n_tastes)

    Raises:
        AssertionError: If segments_frame contains multiple basenames

    Note:
        Y-axis limits are automatically set based on median absolute deviation (MAD)
        to handle outliers and provide consistent scaling across subplots.
    """

    if cmap is None:
        event_color_map = {
            0: '#D1D1D1',  # Nothing/No movement
            1: '#EF8636',  # Gape
            2: '#3B75AF',  # MTMs
        }

    # Check segments_frame has a single basename
    if 'basename' in segments_frame.columns:
        assert len(segments_frame.basename.unique()) == 1

    fig, ax = plt.subplots(
            nrows=raw_emg.shape[1],
            ncols=raw_emg.shape[0],
            figsize=(10, 10),
            sharex=True, sharey=True,
            )
    median_value = np.median(raw_emg[~np.isnan(raw_emg)])
    mad_value = median_abs_deviation(raw_emg[~np.isnan(raw_emg)])
    y_lims = (-median_value, median_value + mad_scale * mad_value)
    for taste in range(raw_emg.shape[0]):
        for trial in trange(raw_emg.shape[1]):
            this_ax = ax[trial, taste]
            this_emg = raw_emg[taste, trial, :]
            if this_emg is None or np.all(np.isnan(this_emg)):
                continue
            this_ax.plot(this_emg, 'k-', linewidth=0.8)
            # Set y-limits based on median absolute deviation
            this_ax.set_ylim(y_lims)
            if taste == 0:
                this_ax.set_ylabel(f'Trial\n{trial}')
            if trial == raw_emg.shape[1] - 1:
                this_ax.set_xlabel('Time (ms)')
            if trial == 0:
                this_ax.set_title(f'Taste {taste}')
            # Overlay predictions
            for _, segment in segments_frame.iterrows():
                if segment.taste == taste and segment.trial == trial:
                    start, end = segment.segment_bounds
                    pred = segment.pred
                    color = event_color_map[pred]
                    this_ax.axvspan(start, end, alpha=0.5, color=color)
    # Add legend
    if cmap: 
        legend_elements = [
            Patch(facecolor=cmap(0), alpha=0.5, label='Nothing'),
            Patch(facecolor=cmap(1), alpha=0.5, label='Gape'),
            Patch(facecolor=cmap(2), alpha=0.5, label='MTMs')
            ]
    else:
        legend_elements = [
            Patch(facecolor=event_color_map[0], alpha=0.5, label='Nothing'),
            Patch(facecolor=event_color_map[1], alpha=0.5, label='Gape'),
            Patch(facecolor=event_color_map[2], alpha=0.5, label='MTMs')
            ]
    # Put legend at bottom of figure
    fig.legend( 
            handles=legend_elements,
            loc='lower center',
            ncol=3,
            )

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
