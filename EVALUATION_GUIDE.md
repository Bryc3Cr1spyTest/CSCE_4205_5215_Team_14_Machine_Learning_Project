# Usage Guide
This guide explains how to load the data and use the shared evaluator. 
- `data_preparation/data_preparation.py` loads the UCI HAR data and cv splits.
- `evaluation/evaluator.py` trains and scores a model on the splits and saves the results. 

## Usage by Model - Evaluation

| Model | Features | Evaluation Usage |
| --- | --- | --- |
| PCA + KNN | `'eng'` | `Evaluator`, `fit_predict_timing`, `evaluate_cv`, `evaluate_test` | 
| Random Forests | `'eng'` | `Evaluator`, `fit_predict_timing`, `evaluate_cv`, `evaluate_test` |
| 1D CNN | `'raw'` | `Evaluator`, `make_model_fn`, `evaluate_cv`, `evaluate_test`, `evaluate_removed_channel`|

## Loading Data
Below is an example of loading training data with the `'eng'` engineered features. Use this for loading training data, the `Evaluator` handles testing data loading. 
```python
from data_preparation.data_preparation import load_split

data = load_split(splits=('train',), features=('eng',))
X_train = data['X_train_eng']
y_train = data['y_train']
subject_train = data['subject_train']
```

`load_split` returns a dict with the following keys:

| Key | Values |
| --- | --- |
| `X_train_eng`, `X_test_eng` | Engineered features |
| `X_train_raw`, `X_test_raw` | Raw sensors with 128 timesteps and 9 sensor channels |
| `y_train`, `y_test` | training and testing targets (activity numbering 0-5) |
| `subject_train`, `subject_test` | Participant ID for each row in training and testing, respectively |

`load_activity_labels()` gives the associated activity name dict for each target key 0-5. 

`load_feature_names()` loads the 561 engineered feature names, where duplicated row names are differentiated by a `_#` appended at the end.

Individual loaders also exist, if needed (e.g. `load_X_train_eng()`). To find these, inspect `data_preparation.py` to locate the needed loader.

## Loading CV Splits
The 5-fold cross validation exists within `splits/cv_splits.npz` with the seed 14. These are loaded in `Evaluator` by default. 

This contains row indices for training and validation per fold as `(train_idx, val_idx)` pairs per fold.

## Evaluator
The `Evaluator` loads the training data, participant IDs, and CV folds for the chosen feature type (`'raw'`, or `'eng'`). 

```python
from evaluation.evaluator import (
    # Model-specific functions (in Usage by Model - Evaluation table), 
    Evaluator, save_results, save_participant_results,
    save_confusion_matrices, save_hardware_info
)

evaluator = Evaluator('eng') # or 'raw'
```

## Model Function
Each model uses a `model_fn` with these parameters and outputs.
```python
model_fn(X_train, y_train, X_eval) -> preds, train_time, inference_time
```

If your `model_fn` accepts `subjects` as an argument, that can be used to pass training participant IDs to it.

The `model_fn` can be created in two ways.

### First Method
Write a no argument `create_model` function that returns an object with `fit(X, y)` and `predict(X)` capabilities. Then, use `fit_predict_timing` to do the training and inference timing. Note that `fit_predict_timing` can use a `sync` function to do GPU synchronization. 

For PCA + KNN, this can be done by wrapping the PCA and KNN in a class with fit and predict. 

### Second Method
Write a `train_fn` and a `predict_fn` and use `make_model_fn`.
```python
def train_fn(X_train, y_train, subjects=None):
    # returns new model 

def predict_fn(model, X_eval):
    # returns 1D array of predictions
model_fn = make_model_fn(train_fn, predict_fn)
```

The `subjects` argument for `train_fn` can be added, When using GPU, call `torch.cuda.synchronize()` at the end, so the timings are accurate.

## Running an evaluation
```python
result = evaluator.evaluate_cv(model_fn, 'name')
result = evaluator.evaluate_test(model_fn, 'name')
```

`evaluate_cv` trains a model on each fold's training rows and scores the validation participants. `evaluate_test` trains on the entire training set and scores the 9 test participants. These return a dict containing three components. Use `evaluate_cv` to compare settings and models. `evaluate_test` should retrain with your chosen settings on all training rows and score the test participants once.

| Key | Values |
| --- | --- |
| `rows` | One row per fold (a single row for test), with: `model_name`, `feat_type`, `split`, `fold` (fold k), `n_train` (train index length), `n_val` (validation index length), `accuracy`, `macro_f1`, `train_time`, and `inference_time`, and per-class f1 scores. |
| `participant_rows` | One row per fold and participant, with: `model_name`, `split`, `fold`, `participant` (ID), `n` (amount participant data instances in fold), `accuracy`, and `macro_f1`. |
| `confusion_matrices` | A list of 6x6 confusion matrix arrays per fold. |

## Saving results
Save each run to a file so that tables and figures are created from those files.
```python
split = 'cv' # or 'test'
name = 'model_name'
save_results(result, 'results', f'{name}_results.csv')
save_participant_results(result, 'results', f'{name}_participants.csv')
save_confusion_matrices(result, 'results', f'{name}_cm_{split}.npz')
save_hardware_info('results', f'hardware_your_name.json')
```

`save_results` and `save_participant_results` append to the CSV. `save_results` adds a timestamp and the seed (use seed=14). `save_confusion_matrices` and `save_hardware_info` overwrite existing files in the same directory with that name. 

## Channel-Removal Experiment
`evaluate_removed_channel` uses `'raw'` features and a `train_fn` and `predict_fn`. This corresponds to the second `model_fn` method. 
```python
evaluator = Evaluator('raw')
result = evaluator.evaluate_removed_channel(train_fn, predict_fn, 'name', split='cv')

# Then run save_results, save_participant_results, and save_confusion_matrices
```

The model is trained once per cv fold or once for test, then scored 10 times, with the first using all channels, then removing one of the channels for the next 9 scoring runs by setting that channel's values to 0. 
`f1_gap` is the gap between the full-channel macro-F1 to the current macro-F1 for that iteration. 
Confusion matrices are saved for each fold, with the matrices following the same 10 iteration format before moving on to the next fold. 

## Note About Test Split
Do not use the test split until the the pipeline and model design choices are finalized. 