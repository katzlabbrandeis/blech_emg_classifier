# EMG Movement Classifier

This repository contains code for classifying EMG (electromyography) movement patterns using XGBoost.

## Project Structure

```
.
├── artifacts/               # Trained models and data
│   ├── model/              # XGBoost model files
│   ├── pca_obj.pkl         # PCA transformation object
│   ├── scale_obj.pkl       # Data scaling object
│   └── event_code_dict.json # Movement type mappings
├── src/                    # Source code
│   ├── ClassifierHandler.py # Main classifier interface
│   ├── preprocessing.py     # Data preprocessing functions
│   ├── visualize.py        # Visualization functions
│   └── run_flow.py         # Example usage script
└── requirements.txt        # Python dependencies
```

## Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd <repository-name>
```

2. Create and activate a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
.\venv\Scripts\activate  # Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

The main interface is the `ClassifierHandler` class which handles:

- Loading EMG envelope data
- Preprocessing signals
- Feature extraction
- Movement classification using XGBoost

Basic usage:

```python
from ClassifierHandler import ClassifierHandler, get_paths

# Get artifact directories
artifact_dir, model_save_dir = get_paths()

# Initialize classifier
handler = ClassifierHandler(
    model_dir=model_save_dir,
    output_dir=artifact_dir,
    env_path='path/to/emg/envelope.npy'
)

# Run classification
predictions, segments = handler.parse_and_predict()
```

### Columns of `segments`:

| Column | Description |
|--------|-------------|
| `features` | Features of the segment (with normalized amplitude) |
| `segment_raw` | Raw segment |
| `segment_norm_interp` | Amplitude normalized and constant-length interpolated segment |
| `segment_bounds` | Start and end of segment |
| `taste` | Taste of the trial |
| `trial` | Trial number given taste |
| `raw_features` | Features of the segment (without amplitude normalization) |
| `pred` | Predicted movement type |
| `pred_names` | Predicted movement type name |
| `pred_proba` | Predicted movement type probability |

The classifier identifies 3 movement types:
- No movement (0)
- Gape (1)
- MTMs (2)

See `run_flow.py` for a complete example of both classification and visualization.

## Creating Visualizations

The `visualize.py` module provides several functions for visualizing EMG classification results. These visualizations help you understand movement patterns across trials and inspect classification quality.

### Available Visualization Functions

#### 1. Raster Plot (`generate_raster_plot`)

Creates an overview visualization showing movement classifications across all trials and tastes. Each row represents a trial, and colors indicate the type of movement detected at each time point.

```python
from visualize import generate_raster_plot

# Generate raster plot from segments DataFrame
fig, ax = generate_raster_plot(
    segments_frame=segment_frame,
    session_name="My Session"  # Optional: adds title to plot
)
plt.show()
```

**Color coding:**
- Gray (#D1D1D1): No movement
- Orange (#EF8636): Gape
- Blue (#3B75AF): MTMs (Mouth and Tongue Movements)

#### 2. Detailed Plot (`generate_detailed_plot`)

Creates a detailed visualization of a single trial showing both the raw EMG signal and the classified movements. Useful for inspecting classification results in detail.

```python
from visualize import generate_detailed_plot

# Load raw EMG data
env = handler.load_env_file()

# Generate detailed plot for a specific trial
fig, ax = generate_detailed_plot(
    segments_frame=segment_frame,
    raw_emg=env,           # Optional: if provided, shows EMG signal
    trial_idx=0,           # Trial to visualize
    taste_idx=0            # Taste to visualize
)
plt.show()
```

#### 3. EMG Overlay Grid (`plot_env_pred_overlay`)

Creates a grid visualization showing all trials with raw EMG signals and overlaid movement classifications. This provides a comprehensive view of all data at once.

```python
from visualize import plot_env_pred_overlay

# Generate grid of all trials with overlaid predictions
fig, ax = plot_env_pred_overlay(
    segments_frame=segment_frame,
    raw_emg=env  # Shape: (n_tastes, n_trials, time)
)
plt.show()
```

### Complete Visualization Example

Here's a complete example that runs classification and generates all visualizations:

```python
from ClassifierHandler import ClassifierHandler, get_paths
from visualize import generate_raster_plot, generate_detailed_plot, plot_env_pred_overlay
import matplotlib.pyplot as plt

# Initialize classifier
artifact_dir, model_save_dir = get_paths()
handler = ClassifierHandler(
    model_dir=model_save_dir,
    output_dir=artifact_dir,
    env_path='path/to/emg/envelope.npy'
)

# Run classification
predictions, segments, feature_names = handler.parse_and_predict()

# Load raw EMG data for visualization
env = handler.load_env_file()

# 1. Generate raster plot overview
fig1, ax1 = generate_raster_plot(
    segments_frame=segments,
    session_name="Session 1"
)
plt.savefig('raster_plot.png')
plt.show()

# 2. Generate detailed plot for first trial
fig2, ax2 = generate_detailed_plot(
    segments_frame=segments,
    raw_emg=env,
    trial_idx=0,
    taste_idx=0
)
plt.savefig('detailed_trial_0.png')
plt.show()

# 3. Generate grid of all trials
fig3, ax3 = plot_env_pred_overlay(
    segments_frame=segments,
    raw_emg=env
)
plt.savefig('all_trials_overlay.png')
plt.show()
```

### Customizing Visualizations

You can customize the color scheme by providing a custom colormap:

```python
from matplotlib.colors import ListedColormap

# Define custom colors
custom_colors = {
    0: '#CCCCCC',  # No movement
    1: '#FF6B6B',  # Gape
    2: '#4ECDC4',  # MTMs
}
cmap = ListedColormap(list(custom_colors.values()))

# Use custom colormap
fig, ax = plot_env_pred_overlay(
    segments_frame=segments,
    raw_emg=env,
    cmap=cmap
)
```

See `run_flow.py` for a complete working example.

## Components

### ClassifierHandler.py

Main interface class that coordinates:
- Loading EMG data and trained models
- Running preprocessing pipeline
- Generating features
- Making predictions

### preprocessing.py

Contains functions for:
- Movement extraction from EMG signals
- Feature calculation (duration, amplitude, intervals etc.)
- PCA transformation
- Data scaling

### visualize.py

Contains functions for visualizing classification results:
- `generate_raster_plot`: Creates raster plots showing movement types across trials
- `generate_detailed_plot`: Detailed visualization of a single trial with raw EMG and classifications
- `plot_env_pred_overlay`: Grid visualization of all trials with EMG signals and overlaid classifications
- Helper functions for data formatting and plotting

### Artifacts

- `xgb_model.json`: Trained XGBoost classifier
- `pca_obj.pkl`: Fitted PCA transformation
- `scale_obj.pkl`: Fitted data scaler
- `event_code_dict.json`: Movement type label mappings

## Requirements

- Python 3.6+
- NumPy
- Pandas
- Scikit-learn
- XGBoost
- SciPy
- Matplotlib
- tqdm

See requirements.txt for specific versions.
