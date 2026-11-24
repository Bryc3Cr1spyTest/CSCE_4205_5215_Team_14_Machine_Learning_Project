
import os
import platform
import json
import time
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
import warnings
import inspect

from data_preparation.data_preparation import (
    load_activity_labels,
    load_cv_splits,
    load_split,
    remove_sensor_channel,
    signals,
    axes,
    PROJECT_ROOT
)

# Default list of results columns
COLUMNS = ['timestamp', 'model_name', 'feat_type', 'split', 'fold', 'removed_channel', 'n_train',
           'n_val', 'accuracy', 'macro_f1', 'f1_gap', 'train_time', 'inference_time', 'seed']

N_CLASSES = 6
N_FOLDS = 5
N_TIMESTEPS = 128
N_CHANNELS = 9
N_ENG_FEATURES = 561

# Error checking for feat_type name
def _check_feat_type(feat_type):
    if feat_type not in ('raw', 'eng'):
        raise ValueError(f"feat_type given: {feat_type} - accepted 'raw' or 'eng'")

# Error checking for string content
def _check_valid_str(value, name):
    if not isinstance(value, str) or not value.strip():
        raise TypeError(f'{name} needs to be a nonempty string. given: {value}')

# Error checking for file extension format
def _check_extension(filename, extension):
    _check_valid_str(filename, 'filename')
    if not filename.endswith(extension):
        raise ValueError(f'filename given: {filename}. accepted extension: {extension}')

# Error checking for result type and keys
def _check_result(result, key):
    if not isinstance(result, dict):
        raise TypeError(f'Given result type: {type(result).__name__}. accepted: dict.')
    if key not in result:
        raise ValueError(f'Result missing key {key}.')
    if len(result[key]) == 0:
        raise ValueError(f"result['{key}'] is empty.")    

# Error checking for verifying feature shapes
def _check_feature_shape(X, feat_type, name):
    X = np.asarray(X)
    if feat_type == 'raw':
        if X.ndim != 3 or X.shape[1:] != (N_TIMESTEPS, N_CHANNELS):
            raise ValueError(f'{name} raw feature shape does not match (n, {N_TIMESTEPS}, {N_CHANNELS})')
    else:
        if X.ndim != 2 or X.shape[1] != N_ENG_FEATURES:
            raise ValueError(f'{name} eng feature shape does not match (n, {N_ENG_FEATURES})')

# Error checking for split name
def _check_split_name(split):
    if split not in ('cv', 'test'):
        raise ValueError(f"split given: {split}. Allows 'cv' or 'test'.")

# Error checking for predictions
def _check_preds(preds, n_allowed):
    preds = np.asarray(preds)
    if preds.ndim != 1:
        raise ValueError(f'Given preds shape: {preds.shape}. Allows 1D.')
    if len(preds) != n_allowed:
        raise ValueError(f'Given preds length: {len(preds)}. Allows {n_allowed}.')
    if preds.size == 0:
        raise ValueError('preds is empty.')
    if not np.issubdtype(preds.dtype, np.integer):
        raise TypeError(f'Given dtype: {preds.dtype}. preds allows integer class labels.')
    if preds.min() < 0 or preds.max() >= N_CLASSES:
        raise ValueError(f'Given preds class range: {preds.min()}-{preds.max()}. Allows 0-{N_CLASSES-1}.')

# Error checking for train/inference times
def _check_times(train_time, inference_time):
    for name, value in (('train_time', train_time), ('inference_time', inference_time)):
        try:
            t = float(value)
        except (TypeError, ValueError):
            raise TypeError(f'{name} given: {value}. {name} allows float.')
        if not np.isfinite(t) or t < 0:
            raise ValueError(f'{name} given: {value}. {name} only allows a finite non-negative value.')

# Error checking for label range (0-5)
def _check_label_range(y, name):
    y = np.asarray(y)
    if y.min() < 0 or y.max() >= N_CLASSES:
        raise ValueError(f'{name} given range: {y.min()}-{y.max()}. Allows 0-{N_CLASSES - 1}.')

# Warns if classes missing from y
def _warn_missing_classes(y, name):
    missing = sorted(set(range(N_CLASSES)) - set(np.unique(y)))
    if missing:
        warnings.warn(f'{name} is missing classes {missing}.')


# Calls the fn with subjects parameter if subjects included
def _fn_subjects(model_fn, *args, subjects=None):
    if subjects is not None and 'subjects' in inspect.signature(model_fn).parameters:
        return model_fn(*args, subjects=subjects)
    return model_fn(*args)



        
# Defines an output path based on a given subfolder of the project and filename.
def _output_path(folder, filename):
    _check_valid_str(folder, 'folder')
    _check_valid_str(filename, 'filename')
    path = PROJECT_ROOT / folder / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    return path  

# Saves the data from the result['rows'] to a CSV given a folder and filename, 
# and adds a timestamp, the seed, removed_channel set to none if 
# removed_channel is not a column, and reindexes the columns to
# the COLUMNS list first, then other columns in addt_columns.
def save_results(result, folder, filename, seed=14):
    _check_result(result, 'rows')
    _check_extension(filename, '.csv')
    if not isinstance(seed, (int, np.integer)):
        raise TypeError(f'seed given: {seed}. Allowed format: int value.')
    df = pd.DataFrame(result['rows'])
    df['timestamp'] = datetime.now().isoformat(timespec='seconds')
    df['seed'] = seed
    if 'removed_channel' not in df:
        df['removed_channel'] = 'none'
    addt_columns = [col for col in df.columns if col not in COLUMNS]
    df = df.reindex(columns=COLUMNS + addt_columns)

    path = _output_path(folder, filename)
    df.to_csv(path, mode='a', header=not path.exists(), index=False)

# Saves the data from the result['participant_rows'] to a CSV
# given a folder and filename.
def save_participant_results(result, folder, filename):
    _check_result(result, 'participant_rows')
    _check_extension(filename, '.csv')
    df = pd.DataFrame(result['participant_rows'])
    path = _output_path(folder, filename)
    df.to_csv(path, mode='a', header=not path.exists(), index=False)

# Saves the confusion matrix data from result['confusion_matrices'] to an .npz
# given a folder and filename. This stores each matrix under cm_i where i
# is the matrix position in the list of confusion matrices. 
def save_confusion_matrices(result, folder, filename):
    _check_result(result, 'confusion_matrices')
    _check_extension(filename, '.npz')
    for i, cm in enumerate(result['confusion_matrices']):
        cm = np.asarray(cm)
        if cm.ndim != 2 or cm.shape[0] != cm.shape[1]:
            raise ValueError(f'confusion matrix {i} shape given: {cm.shape}. Allows only 2D square dimensions.')
        if cm.shape[0] != N_CLASSES:
            raise ValueError(f'confusion matrix {i} shape given: {cm.shape}. Allows {N_CLASSES}x{N_CLASSES}.')
    path = _output_path(folder, filename)
    np.savez(path, **{f'cm_{i}': cm for i, cm in enumerate(result['confusion_matrices'])})

# Saves the device hardware info to a json file under the specified
# folder and filename. This includes the os (operating system),
# machine (CPU instruction set), processor, cpu_count, python_version,
# torch_version, cuda_available (whether cuda is available), 
# gpu (gpu name), and mps_available (whether mps is available).
def save_hardware_info(folder, filename):
    _check_valid_str(folder, 'folder')
    _check_extension(filename, '.json')
    info = {
        'os': platform.platform(),
        'machine': platform.machine(),
        'processor': platform.processor(),
        'cpu_count': os.cpu_count(),
        'python_version': platform.python_version()
    }

    try:
        import torch
        info['torch_version'] = torch.__version__
        info['cuda_available'] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info['gpu'] = torch.cuda.get_device_name(0)
        info['mps_available'] = torch.backends.mps.is_available()
    except ImportError:
        info['cuda_available'] = False

    path = _output_path(folder, filename)
    with open(path, 'w') as f:
        json.dump(info, f, indent=2)

# Helper function for model_fn to automatically time fit
# and prediction given a create_model function (w/o args)
# using the created model object's fit and predict functions.
# Returns preds (model predictions), train_time, and inference_time
def fit_predict_timing(create_model, X_train, y_train, X_eval, sync=None):
    if not callable(create_model):
        raise TypeError('create_model needs to be callable with no args.')
    if sync is not None and not callable(sync):
        raise TypeError('sync needs to be callable or None.')
    if len(X_train) != len(y_train):
        raise ValueError(f'X_train (rows: {len(X_train)}) does not equal y_train (rows: {len(y_train)}) in num rows.')
    if np.shape(X_train)[1:] != np.shape(X_eval)[1:]:
        raise ValueError(f"X_train features ({np.shape(X_train)[1:]}) does not match X_eval features {np.shape(X_eval)[1:]}.")

    sync = sync or (lambda: None)
    model = create_model()
    if not (hasattr(model, 'fit') and hasattr(model, 'predict')):
        raise TypeError('create_model() needs to return an object with fit() and predict().')
    sync()
    start_time = time.perf_counter()
    model.fit(X_train, y_train)
    sync()
    train_time = time.perf_counter() - start_time

    start_time = time.perf_counter()
    preds = model.predict(X_eval)
    sync()
    inference_time = time.perf_counter() - start_time

    if len(preds) != len(X_eval):
        raise ValueError(f'predict() returned {len(preds)} predictions for {len(X_eval)} rows.')

    return np.asarray(preds), train_time, inference_time

# Function to create a model_fn from the train and predict functions to be used in
# evaluate_cv or evaluate_test. Returns preds, train_time, inference_time
def make_model_fn(train_fn, predict_fn):
    if not callable(train_fn) or not callable(predict_fn):
        raise TypeError('train_fn and predict_fn need to be callable.')

    def model_fn(X_train, y_train, X_eval, subjects=None):
        start_time = time.perf_counter()
        model = _fn_subjects(train_fn, X_train, y_train, subjects=subjects)
        train_time = time.perf_counter() - start_time
        if model is None:
            raise ValueError('train_fn returned None.')

        start_time = time.perf_counter()
        preds = predict_fn(model, X_eval)
        inference_time = time.perf_counter() - start_time
        return preds, train_time, inference_time
    return model_fn


# Evaluator class for conducting model
# evaluation given a specific feat_type ('raw' or 'eng') 
class Evaluator:
    # Evaluator constructor, taking in a feat_type ('raw' or 'eng'),
    # and creates member variables for feat_type, X_train (training inputs),
    # y_train (training targets), subject_train (training participant IDs),
    # cv_splits, and activity_labels, and a placeholder _test.
    def __init__(self, feat_type):
        _check_feat_type(feat_type)
        self.feat_type = feat_type

        train = load_split(splits=('train',), features=(feat_type,))
        self.X_train = train[f'X_train_{feat_type}']
        self.y_train = train['y_train']
        self.subject_train = train['subject_train']

        self.cv_splits = load_cv_splits()
        self.activity_labels = load_activity_labels()

        _check_feature_shape(self.X_train, feat_type, 'X_train')
        if not (len(self.X_train) == len(self.y_train) == len(self.subject_train)):
            raise ValueError('X_train, y_train, and subject_train rows have different row amounts.')
        if len(self.cv_splits) != N_FOLDS:
            raise ValueError(f'Given fold amt: {len(self.cv_splits)}. Allows {N_FOLDS} folds.')
        if sorted(self.activity_labels) != list(range(N_CLASSES)):
            raise ValueError(f'Given activity_labels key range: {sorted(self.activity_labels)}. Allows 0-{N_CLASSES - 1}.')
        self._test = None

    # This loads the test data based on the feat_type ('raw' or 'eng'), and
    # returns X (inputs), y (target values), and participant IDs for that split.
    def _load_test(self):
        if self._test is None:
            test = load_split(splits=('test',), features=(self.feat_type,))

            X_test = test[f'X_test_{self.feat_type}']
            y_test = test['y_test']
            subject_test = test['subject_test']
            _check_feature_shape(X_test, self.feat_type, 'X_test')
            if np.shape(X_test)[1:] != np.shape(self.X_train)[1:]:
                raise ValueError(f'X_test features ({np.shape(X_test)[1:]}) does not match X_train features ({np.shape(self.X_train)[1:]}).')
            if not (len(X_test) == len(y_test) == len(subject_test)):
                raise ValueError('X_test, y_test, and subject_test rows have different row amounts.')
            overlap = set(self.subject_train) & set(subject_test)
            if overlap:
                raise ValueError(f'Participant IDs spotted in both training and test splits: {sorted(overlap)}')

            self._test = (
                test[f'X_test_{self.feat_type}'],
                test['y_test'],
                test['subject_test'],
            )
        return self._test

    # Run model_fn on the given training and evaluation inputs (validation or test),
    # before returning preds (predictions), the train_time, and the inference_time.
    def _run(self, model_fn, X_train, y_train, X_eval, subject_train=None):
        if not callable(model_fn):
            raise TypeError('model_fn needs to be callable.')
        if np.shape(X_train)[1:] != np.shape(X_eval)[1:]:
            raise ValueError(f'X_train features ({np.shape(X_train)[1:]}) does not match X_eval features ({np.shape(X_eval)[1:]}).')
        
        out = _fn_subjects(model_fn, X_train, y_train, X_eval, subjects=subject_train)

        if not isinstance(out, (tuple, list)) or len(out) != 3:
            raise ValueError('model_fn did not return as (preds, train_time, inference_time).')

        preds, train_time, inference_time = out

        _check_preds(preds, len(X_eval))
        _check_times(train_time, inference_time)

        preds = np.asarray(preds)

        return preds, float(train_time), float(inference_time)

    # Evaluation function for the Evaluator feat_type ('raw' or 'eng'). (cv)
    # The model_fn trains and predicts, and returns preds, train_time, and inference_time.
    # Then, each CV fold is scored.
    # This returns a dict of rows, participant_rows, and confusion_matrices.
    # Each row contains a summary per cv 
    # fold containing the model_name, feat_type, split,
    # fold (fold k), n_train (train index length), n_val (validation index length), 
    # accuracy, macro_f1, train_time, and inference_time, and per-class f1 scores. 
    # Each participant row contains the model_name, split, fold, participant (ID),
    # n (amount participant data instances in fold), accuracy, and macro_f1
    # Each confusion matrix takes the preds and y_val for the fold and adds that
    # confusion matrix to the confusion_matrices list.
    def evaluate_cv(self, model_fn, model_name):
        _check_valid_str(model_name, 'model_name')
        labels = sorted(self.activity_labels)
        rows = []
        participant_rows = []
        confusion_matrices = []

        for k, (train_idx, val_idx) in enumerate(self.cv_splits):
            if len(train_idx) == 0 or len(val_idx) == 0:
                raise ValueError(f'Fold {k}: empty train or validation index.')
            if min(train_idx.min(), val_idx.min()) < 0 or max(train_idx.max(), val_idx.max()) >= len(self.y_train):
                raise ValueError(f'Fold {k}: index out of training range: {len(self.y_train)}.')
            if np.intersect1d(train_idx, val_idx).size > 0:
                raise ValueError(f'Fold {k}: contains rows in both train and validation.')

            X_train, y_train = self.X_train[train_idx], self.y_train[train_idx]
            X_val, y_val = self.X_train[val_idx], self.y_train[val_idx]
            subject_val = self.subject_train[val_idx]
            _warn_missing_classes(y_val, f'Fold {k} y_val')

            preds, train_time, inference_time = self._run(model_fn, X_train, y_train, X_val, self.subject_train[train_idx])

            row = {
                'model_name': model_name,
                'feat_type': self.feat_type,
                'split': 'cv',
                'fold': k,
                'n_train': len(train_idx),
                'n_val': len(val_idx),
                'accuracy': accuracy_score(y_val, preds),
                'macro_f1': f1_score(y_val, preds, average='macro'),
                'train_time': train_time,
                'inference_time': inference_time
            }

            class_scores = f1_score(y_val, preds, labels=labels, average=None, zero_division=0)
            for label, score in zip(labels, class_scores):
                row[f'f1_{self.activity_labels[label]}'] = score
            rows.append(row)

            confusion_matrices.append(confusion_matrix(y_val, preds, labels=labels))

            for subject in np.unique(subject_val):
                mask = subject_val == subject
                participant_rows.append({
                    'model_name': model_name,
                    'split': 'cv',
                    'fold': k,
                    'participant': int(subject),
                    'n': int(mask.sum()),
                    'accuracy': accuracy_score(y_val[mask], preds[mask]),
                    'macro_f1': f1_score(y_val[mask], preds[mask], average='macro')
                })
        return {'rows': rows, 'participant_rows': participant_rows, 'confusion_matrices': confusion_matrices}

    # Evaluation function for the Evaluator feat_type ('raw' or 'eng'). (test)
    # The model_fn trains and predicts, and returns preds, train_time, and inference_time.
    # Then, the test set is scored.
    # This returns a dict of rows, participant_rows, and confusion_matrices.
    # The row contains a summary with the model_name, feat_type, split,
    # fold (blank), n_train (train index length), n_val (test index length), 
    # accuracy, macro_f1, train_time, and inference_time, and per-class f1 scores. 
    # The participant row contains the model_name, split, fold (blank), participant (ID),
    # n (amount participant data instances in test), accuracy, and macro_f1
    # The confusion matrix takes the preds and y_test and uses that
    # confusion matrix as the confusion_matrices list.
    def evaluate_test(self, model_fn, model_name):
        _check_valid_str(model_name, 'model_name')
        labels = sorted(self.activity_labels)
        X_test, y_test, subject_test = self._load_test()
        _check_label_range(y_test, 'y_test')
        _warn_missing_classes(y_test, 'y_test')

        preds, train_time, inference_time = self._run(model_fn, self.X_train, self.y_train, X_test, self.subject_train)

        row = {
                'model_name': model_name,
                'feat_type': self.feat_type,
                'split': 'test',
                'fold': '',
                'n_train': len(self.y_train),
                'n_val': len(y_test),
                'accuracy': accuracy_score(y_test, preds),
                'macro_f1': f1_score(y_test, preds, average='macro'),
                'train_time': train_time,
                'inference_time': inference_time
        }

        class_scores = f1_score(y_test, preds, labels=labels, average=None, zero_division=0)
        for label, score in zip(labels, class_scores):
            row[f'f1_{self.activity_labels[label]}'] = score

        participant_rows = []
        for subject in np.unique(subject_test):
            mask = subject_test == subject
            participant_rows.append({
                'model_name': model_name,
                'split': 'test',
                'fold': '',
                'participant': int(subject),
                'n': int(mask.sum()),
                'accuracy': accuracy_score(y_test[mask], preds[mask]),
                'macro_f1': f1_score(y_test[mask], preds[mask], average='macro')
            })

        return {
            'rows': [row],
            'participant_rows': participant_rows,
            'confusion_matrices': [confusion_matrix(y_test, preds, labels=labels)]
        }

    # Scoring function for CNN channel variations. Takes in a model, predict_fn, 
    # X, y, subjects, train_time, base_row, and k. Scores the 
    # trained model on a given evaluation X (X_val or X_test), evaluation y 
    # (y_val or y_test), and evaluation subjects (participant IDs) for 
    # a specified k (numbered fold or blank for the test scoring). 
    # This returns the rows, participant_rows, and confusion_matrices.
    # Each row contains a summary per
    # fold and channel variation containing the base_row entries,
    # fold (fold k), removed_channel, 
    # accuracy, macro_f1, f1_gap (the fold and variation difference to full_channel_f1),
    # train_time, inference_time, and per-class f1 scores. 
    # Each participant row contains the model_name, split, fold, removed_channel,
    # participant (ID), n (amount participant data instances in fold), accuracy, macro_f1,
    # and f1_gap (the fold and variation difference to participant_f1),
    # Each confusion matrix takes the preds and y for the fold and variation
    # and adds that confusion matrix to the confusion_matrices list.
    def _score_channels(self, model, predict_fn, X, y, subjects, train_time, base_row, k=''):
        if not callable(predict_fn):
            raise TypeError('predict_fn needs to be callable.')
        _check_feature_shape(X, 'raw', 'X')
        if not (len(X) == len(y) == len(subjects)):
            raise ValueError('X, y, and subjects rows have different row amounts.')
        missing_keys = {'model_name', 'feat_type', 'split'} - set(base_row)
        if missing_keys:
            raise ValueError(f'base_row missing keys: {sorted(missing_keys)}.')
        
        labels = sorted(self.activity_labels)
        channels = [f'{signal}_{axis}' for signal in signals for axis in axes]
        rows = []
        full_channel_f1 = None
        participant_rows = []
        cms = []
        participant_f1 = {}

        for removed in ['none'] + channels:
            if removed != 'none' and full_channel_f1 is None:
                raise RuntimeError("'none' needs to be evaluated first for f1_gap.")
            X_eval = X if removed == 'none' else remove_sensor_channel(X, removed)

            start_time = time.perf_counter()
            preds = np.asarray(predict_fn(model, X_eval))
            inference_time = time.perf_counter() - start_time
            _check_preds(preds, len(X_eval))

            macro_f1 = f1_score(y, preds, average='macro')
            if removed == 'none':
                full_channel_f1 = macro_f1

            row = {
                **base_row,
                'fold': k,
                'removed_channel': removed,
                'accuracy': accuracy_score(y, preds),
                'macro_f1': macro_f1,
                'f1_gap': full_channel_f1 - macro_f1,
                'train_time': train_time,
                'inference_time': inference_time
            }
            class_scores = f1_score(y, preds, labels=labels, average=None, zero_division=0)
            for label, score in zip(labels, class_scores):
                row[f'f1_{self.activity_labels[label]}'] = score
            rows.append(row)

            cms.append(confusion_matrix(y, preds, labels=labels))

            for subject in np.unique(subjects):
                mask = subjects == subject
                participant_macro_f1 = f1_score(y[mask], preds[mask], average='macro')
                if removed == 'none':
                    participant_f1[int(subject)] = participant_macro_f1
                participant_rows.append({
                    'model_name': base_row['model_name'],
                    'split': base_row['split'],
                    'fold': k,
                    'removed_channel': removed,
                    'participant': int(subject),
                    'n': int(mask.sum()),
                    'accuracy': accuracy_score(y[mask], preds[mask]),
                    'macro_f1': participant_macro_f1,
                    'f1_gap': participant_f1[int(subject)] - participant_macro_f1
                })
        return rows, participant_rows, cms

    # Evaluation for CNN channel variations. First trains the model using the train_fn parameter and
    # scores the channel variations for either the cv or test split.
    # This returns a dict of rows, participant_rows, and confusion_matrices.
    # Takes in a train_fn, predict_fn, model_name, and split (cv or test). 
    # base is the base_row parameter used in _score_channels. This contains the model_name,
    # feat_type, and split. The train time will be computed around the train_fn and 
    # _score_channels will either be used per fold with cv or once with test. 
    # _score_channels builds the rows, participant_rows, and confusion_matrices.
    def evaluate_removed_channel(self, train_fn, predict_fn, model_name, split='cv'):
        _check_valid_str(model_name, 'model_name')
        _check_split_name(split)
        if self.feat_type != 'raw':
            raise ValueError(f"Given feat_type: {self.feat_type}. 'raw' feat_type needed for channel variation evaluation.")
        if not callable(train_fn) or not callable(predict_fn):
            raise TypeError('train_fn and predict_fn need to be callable.')

        
        base = {
            'model_name': model_name,
            'feat_type': self.feat_type,
            'split': split
        }
        rows = []
        participant_rows = []
        cms = []

        if split == 'cv':
            for k, (train_idx, val_idx) in enumerate(self.cv_splits):
                start_time = time.perf_counter()
                model = _fn_subjects(train_fn, self.X_train[train_idx], self.y_train[train_idx], subjects=self.subject_train[train_idx])
                train_time = time.perf_counter() - start_time
                if model is None:
                    raise ValueError(f'train_fn returned None on fold {k}.')
                fold_rows, fold_participant_rows, fold_cms = self._score_channels(model, predict_fn, self.X_train[val_idx], self.y_train[val_idx], self.subject_train[val_idx], train_time, base, k)
                rows += fold_rows
                participant_rows += fold_participant_rows
                cms += fold_cms
        elif split == 'test':
            X_test, y_test, subject_test = self._load_test()
            start_time = time.perf_counter()
            model = _fn_subjects(train_fn, self.X_train, self.y_train, subjects=self.subject_train)
            train_time = time.perf_counter() - start_time
            if model is None:
                raise ValueError('train_fn returned None.')
            rows, participant_rows, cms = self._score_channels(model, predict_fn, X_test, y_test, subject_test, train_time, base)

        return {'rows': rows, 'participant_rows': participant_rows, 'confusion_matrices': cms}