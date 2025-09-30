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
