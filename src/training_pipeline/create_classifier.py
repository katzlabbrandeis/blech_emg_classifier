"""
Generate training dataset and train a classifier.
"""

from joblib import Parallel, delayed
from skopt.space import Real, Integer
from skopt.plots import plot_convergence, plot_objective
from skopt import Optimizer, dump, load
import json
from scipy.stats import spearmanr
from sklearn.model_selection import GroupKFold
from tqdm import tqdm, trange
import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from time import time
import xgboost as xgb
import sys
from ast import literal_eval
from sklearn.metrics import accuracy_score
from scipy.stats import ttest_rel
from sklearn.model_selection import train_test_split

this_dir = os.path.dirname(os.path.abspath(__file__))  # src/training_pipeline
repo_root = os.path.abspath(os.path.join(this_dir, '..', '..'))
artifact_dir = os.path.join(this_dir, 'artifacts')
training_dataset_path = os.path.join(
    repo_root, 'data', 'training', 'fin_training_dataset.pkl')

############################################################
# Load data
############################################################
scored_df = pd.read_pickle(training_dataset_path)

############################################################
# Perform hyperparameter optimization
############################################################


xgb_optim_artifacts_dir = os.path.join(artifact_dir, 'xgb_optim_artifacts')
if not os.path.exists(xgb_optim_artifacts_dir):
    os.makedirs(xgb_optim_artifacts_dir)

X_array = np.stack(scored_df.features.values)
y_array = scored_df.event_codes.values
animals_array = scored_df.animal_num.values
unique_animals = np.unique(animals_array)

############################################################
search_spaces = {
    'learning_rate': (0.01, 1.0, 'log-uniform'),
    'min_child_weight': (0, 10),
    'max_depth': (0, 50),
    'max_delta_step': (0, 20),
    'subsample': (0.01, 1.0, 'uniform'),
    'colsample_bytree': (0.01, 1.0, 'uniform'),
    'colsample_bylevel': (0.01, 1.0, 'uniform'),
    'reg_lambda': (1e-9, 1000, 'log-uniform'),
    'reg_alpha': (1e-9, 1.0, 'log-uniform'),
    'gamma': (1e-9, 0.5, 'log-uniform'),
    'min_child_weight': (0, 5),
    'n_estimators': (50, 100),
    'scale_pos_weight': (1e-6, 500, 'log-uniform')
}

dimensions = [
    Real(0.001, 1.0, name='learning_rate', prior='log-uniform'),
    Integer(1, 10, name='min_child_weight'),
    Integer(1, 100, name='max_depth'),
    Integer(0, 20, name='max_delta_step'),
    Real(0.01, 1.0, name='subsample', prior='uniform'),
    Real(0.01, 1.0, name='colsample_bytree', prior='uniform'),
    Real(0.01, 1.0, name='colsample_bylevel', prior='uniform'),
    Real(1e-9, 1000, name='reg_lambda', prior='log-uniform'),
    Real(1e-9, 1.0, name='reg_alpha', prior='log-uniform'),
    Real(1e-9, 0.5, name='gamma', prior='log-uniform'),
    Integer(50, 100, name='n_estimators'),
    # Real(1e-6, 500, name='scale_pos_weight', prior='log-uniform')
]

# Convert search dimensions to dataframe
dim_names = [dim.name for dim in dimensions]
dim_lows = [dim.low for dim in dimensions]
dim_ups = [dim.high for dim in dimensions]
dim_priors = [dim.prior for dim in dimensions]

dim_df = pd.DataFrame(
    columns=['name', 'low', 'high', 'prior'],
    data=list(zip(dim_names, dim_lows, dim_ups, dim_priors))
)
# make low and high floats
dim_df['low'] = dim_df['low'].astype(float)
dim_df['high'] = dim_df['high'].astype(float)

# Print to artifact_dir
dim_df.to_csv(os.path.join(xgb_optim_artifacts_dir,
              'xgb_optim_search_dimensions.csv'))


def run_xgb_cv_loao(params_kwargs, X_array, y_array, animals_array):
    """
    Run cross-validation with leave-one-animal-out

    Parameters
    ----------
    params_kwargs : dict
    X_array : np.array
    y_array : np.array
    animals_array : np.array

    Returns
    -------
    mean_acc : float
    """

    cv_accs = []
    for i, this_animal in enumerate(unique_animals):

        test_idx = animals_array == this_animal
        train_idx = ~test_idx

        X_train = X_array[train_idx]
        y_train = y_array[train_idx]
        X_test = X_array[test_idx]
        y_test = y_array[test_idx]

        clf = xgb.XGBClassifier(
            **params_kwargs,
            use_label_encoder=False,
            n_jobs=1,
            eval_metric='mlogloss'
        )
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        cv_accs.append(acc)

    mean_acc = np.mean(cv_accs)
    return mean_acc


def run_xgb_cv(
        params,
        X_array,
        y_array,
        test_frac=0.2,
):
    """
    Run cross-validation with random splits

    Parameters
    ----------
    params : dict
    X_array : np.array
    y_array : np.array
    test_frac : float

    Returns
    -------
    mean_acc : float
    """

    train_X, test_X, train_y, test_y = train_test_split(
        X_array, y_array, test_size=test_frac, random_state=42
    )

    clf = xgb.XGBClassifier(
        **params,
        use_label_encoder=False,
        n_jobs=1,
        eval_metric='mlogloss'
    )
    clf.fit(train_X, train_y)
    y_pred = clf.predict(test_X)
    acc = accuracy_score(test_y, y_pred)
    return acc


def run_xgb_cv_wrapper(params):
    # return run_xgb_cv(params, X_array, y_array, animals_array)
    return run_xgb_cv(params, X_array, y_array)


###############
# Run optimizer
n_optim_repeats = 20

results_list = []
optimizer_list = []

for repeat_ind in range(len(results_list), n_optim_repeats):

    optimizer = Optimizer(
        dimensions=dimensions,
        base_estimator='GBRT',
        n_initial_points=10,
        acq_func='EI',
        acq_optimizer='auto',
    )

    n_iter = 100
    n_parallel = 4

    for i in trange(n_iter):
        params = optimizer.ask(n_points=n_parallel)
        params_kwargs = [{key: val for key, val in zip([dim.name for dim in dimensions], this_params)}
                         for this_params in params]

        # mean_acc = run_xgb_cv_wrapper(params_kwargs)
        mean_acc = Parallel(n_jobs=n_parallel)(delayed(run_xgb_cv_wrapper)(
            this_params) for this_params in params_kwargs)
        mean_acc = [-this_acc for this_acc in mean_acc]

        result = optimizer.tell(params, mean_acc)

    results_list.append(result)
    optimizer_list.append(optimizer)

    # Save optimizer results
    dump(
        optimizer,
        os.path.join(xgb_optim_artifacts_dir,
                     f'xgb_optimization_results_{repeat_ind}.pkl')
    )

# Save best hyperparameters
best_result_ind = np.argmax([np.max(-result.func_vals)
                            for result in results_list])
best_result = results_list[best_result_ind]
best_params = best_result.x
param_names = [dim.name for dim in dimensions]
best_params_kwargs = {key: val for key, val in zip(param_names, best_params)}
params_df = pd.DataFrame(
    columns=['name', 'value'],
    data=list(best_params_kwargs.items())
)


def return_type(val):
    if type(val) == float:
        return 'float'
    elif type(val) in [int, np.int64]:
        return 'int'
    else:
        return 'other'


params_df['dtype'] = [return_type(val) for val in best_params]
params_df.to_csv(os.path.join(xgb_optim_artifacts_dir, 'best_xgb_hyperparams.csv'),
                 index=False)

############################################################
# Train model
############################################################
# Load hyperparameters
hyperparam_path = os.path.join(
    artifact_dir, 'xgb_optim_artifacts', 'best_xgb_hyperparams.csv')
best_hyperparams = pd.read_csv(hyperparam_path)
hparam_names = []
hparam_vals = []
for i, row in best_hyperparams.iterrows():
    hparam_names.append(row['name'])
    raw_value = row['value']
    dtype = row['dtype']
    if dtype == 'int':
        hparam_vals.append(int(raw_value))
    elif dtype == 'float':
        hparam_vals.append(float(raw_value))
hparam_dict = dict(zip(hparam_names, hparam_vals))

# Train on all data and save model
X_train = np.stack(scored_df.features.values)
y_train = scored_df.event_codes.values

# Calculate sample weights and normalize weight for each class
class_weights = scored_df.event_codes.value_counts(normalize=True)
inv_class_weights = 1 / class_weights
sample_weights = inv_class_weights.loc[y_train].values

# Train model
clf = xgb.XGBClassifier(**hparam_dict)
clf.fit(X_train, y_train, sample_weight=sample_weights)
y_pred = clf.predict(X_train)
acc = accuracy_score(y_train, y_pred)

# Save model
save_dir = os.path.join(artifact_dir, 'xgb_model')
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
clf.save_model(os.path.join(save_dir, 'xgb_model.json'))

# ############################################################
# # Data testing
# # Making sure the training data is loaded correctly
# ############################################################
# # Leave one-animal-out cross validation
# unique_animals = scored_df.animal_num.unique()
#
# # Convert xgb_pred to int
# scored_df['xgb_pred'] = y_pred
# scored_df['xgb_pred'] = scored_df['xgb_pred'].astype('int')
#
# # Add event name to xgb_pred
# bsa_event_map = {
#         0 : 'nothing',
#         1 : 'gapes',
#         2 : 'MTMs',
#         }
#
# scored_df['xgb_pred_event'] = scored_df['xgb_pred'].map(bsa_event_map)
