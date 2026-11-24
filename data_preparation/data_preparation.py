import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_ROOT = PROJECT_ROOT / 'UCI HAR Dataset'
CV_SPLIT_PATH = PROJECT_ROOT / 'splits' / 'cv_splits.npz'

signals = ['body_acc', 'body_gyro', 'total_acc']
axes = ['x', 'y', 'z']

# Error check for path existance
def _check_path(path):
    if not path.exists():
        raise FileNotFoundError(
            f'File not found at path: {path}.'
        )
    return path

# Error check for y label values (0-5)
def _check_labels(y, name):
    if y.min() < 0 or y.max() > 5:
        raise ValueError(f'{name} out of range 0-5, observed range: {y.min()}-{y.max()}')
    return y

# Function to convert dataset txt to numpy
def read_txt(path):
    df = pd.read_csv(_check_path(path), sep=r'\s+', header=None)
    return df.to_numpy()

# Function to make feature names unique for engineered features
def rm_dup_feature_names(feature_names):
    encountered_names = {}
    new_names = []
    for name in feature_names:
        if name in encountered_names:
            encountered_names[name] += 1
            new_names.append(f'{name}_{encountered_names[name]}')
        else:
            encountered_names[name] = 0
            new_names.append(name)
    return new_names

# Function to remove sensor channel
def remove_sensor_channel(X, rm_channel):
    channel_names = [f'{signal}_{axis}' for signal in signals for axis in axes]
    if rm_channel not in channel_names:
        raise ValueError(f'Incorrect channel: {rm_channel}.')
    if X.ndim != 3 or X.shape[2] != len(channel_names):
        raise ValueError('Incorrect input shape. Format: (n, 128, n_channels).')
    X_new = X.copy()
    X_new[:,:, channel_names.index(rm_channel)] = 0.0
    return X_new

# Accessor for activity labels
def load_activity_labels():
    # Activity Labels (0-5 indexed)
    activity_labels_path = DATASET_ROOT / 'activity_labels.txt'
    activity_labels_df = pd.read_csv(_check_path(activity_labels_path), sep=r'\s+', header=None)
    activity_labels_dict = dict(zip(activity_labels_df[0] - 1, activity_labels_df[1]))
    return activity_labels_dict

# Accessor for feature names
def load_feature_names():
    # Engineered Feature Names
    feature_names_path = DATASET_ROOT / 'features.txt'
    feature_names_series = pd.read_csv(_check_path(feature_names_path), sep=r'\s+', header=None, index_col=0)[1]
    dup_feature_names = feature_names_series.tolist()
    feature_names = rm_dup_feature_names(dup_feature_names)
    return feature_names

# Accessor for y train
def load_y_train():
    # Train Targets (0-5)
    y_train_path = DATASET_ROOT / 'train' / 'y_train.txt'
    y_train_df = pd.read_csv(_check_path(y_train_path), header=None)
    y_train = y_train_df.to_numpy().ravel() - 1
    return _check_labels(y_train, 'y_train')

# Accessor for y train (named)
def load_y_train_named():
    activity_labels_dict = load_activity_labels()
    # Train Targets (named)
    y_train_named = np.array([activity_labels_dict[i] for i in load_y_train()])
    return y_train_named

# Accessor for training participants
def load_subject_train():
    # Train participants (subject_train)
    subject_train_path = DATASET_ROOT / 'train' / 'subject_train.txt'
    subject_train_df = pd.read_csv(_check_path(subject_train_path), header=None)
    subject_train = subject_train_df.to_numpy().ravel()
    return subject_train

# Accessor for X train raw
def load_X_train_raw():
    # Train Raw Features
    train_channels = []
    for signal in signals:
        for axis in axes:
            path = DATASET_ROOT / 'train' / 'Inertial Signals' / f'{signal}_{axis}_train.txt'
            train_channels.append(read_txt(_check_path(path)))

    X_train_raw = np.stack(train_channels, axis=-1)
    return X_train_raw

# Accessor for X train engineered
def load_X_train_eng():
    # Train Engineered Features 
    X_train_eng_path = DATASET_ROOT / 'train' / 'X_train.txt'
    X_train_eng = read_txt(_check_path(X_train_eng_path))
    return X_train_eng


# Accessor for y test
def load_y_test():
    # Test Targets (0-5)
    y_test_path = DATASET_ROOT / 'test' / 'y_test.txt'
    y_test_df = pd.read_csv(_check_path(y_test_path), header=None)
    y_test = y_test_df.to_numpy().ravel() - 1
    return _check_labels(y_test, 'y_test')

# Accessor for y test (named)
def load_y_test_named():
    # Test Targets (named)
    activity_labels_dict = load_activity_labels()
    y_test_named = np.array([activity_labels_dict[i] for i in load_y_test()])
    return y_test_named

# Accessor for testing participants
def load_subject_test():
    # Test participants (subject_test)
    subject_test_path = DATASET_ROOT / 'test' / 'subject_test.txt'
    subject_test_df = pd.read_csv(_check_path(subject_test_path), header=None)
    subject_test = subject_test_df.to_numpy().ravel()
    return subject_test

# Accessor for X test raw
def load_X_test_raw():
    # Test Raw Features
    test_channels = []
    for signal in signals:
        for axis in axes:
            path = DATASET_ROOT / 'test' / 'Inertial Signals' / f'{signal}_{axis}_test.txt'
            test_channels.append(read_txt(_check_path(path)))

    X_test_raw = np.stack(test_channels, axis=-1)
    return X_test_raw

# Accessor for X test engineered
def load_X_test_eng():
    # Test Engineered Features
    X_test_eng_path = DATASET_ROOT / 'test' / 'X_test.txt'
    X_test_eng = read_txt(_check_path(X_test_eng_path))
    return X_test_eng

# Loads the specified data split and features type
def load_split(splits=('train', 'test'), features=('raw', 'eng')):
    allowed_splits, allowed_features = {'train', 'test'}, {'raw', 'eng'}
    if isinstance(splits, str) or isinstance(features, str):
        raise TypeError("splits or features given as string, expected tuples. e.g. splits=('train',)")
    
    invalid_splits = set(splits) - allowed_splits
    invalid_features = set(features) - allowed_features

    if invalid_splits: 
        raise ValueError(f'Invalid splits provided: {invalid_splits}')
    if invalid_features:
        raise ValueError(f'Invalid features provided: {invalid_features}')
    loaders = {
        'train': dict(raw=load_X_train_raw, eng=load_X_train_eng, y=load_y_train, subject=load_subject_train),
        'test': dict(raw=load_X_test_raw, eng=load_X_test_eng, y=load_y_test, subject=load_subject_test)
    }
    data = {}
    for split in splits:
        for feat_type in features:
            data[f'X_{split}_{feat_type}'] = loaders[split][feat_type]()
        data[f'y_{split}'] = loaders[split]['y']()
        data[f'subject_{split}'] = loaders[split]['subject']()

        y_split_len = len(data[f'y_{split}'])
        subject_len = len(data[f'subject_{split}'])
        if subject_len != y_split_len:
            raise ValueError(f'{split} y (rows: {y_split_len}) does not equal {split} subject (rows: {subject_len}) in num rows.')
        for feat_type in features:
            X_split_len = len(data[f'X_{split}_{feat_type}'])
            if X_split_len != y_split_len:
                raise ValueError(f'{split} {feat_type}: X (rows: {X_split_len}) does not equal y (rows: {y_split_len}) in num rows.')
    if 'train' in splits and 'test' in splits:
        overlap = set(data['subject_train']) & set(data['subject_test'])
        if overlap:
            raise ValueError(f'Participant IDs spotted in both training and test splits: {sorted(overlap)}')
    return data

    
# Loads the stored cv_splits
def load_cv_splits():
    cv_splits = np.load(_check_path(CV_SPLIT_PATH))
    n_splits = int(cv_splits['n_splits'])
    folds = [(cv_splits[f'train_idx_{k}'], cv_splits[f'val_idx_{k}']) for k in range(n_splits)]
    
    train_len = len(load_y_train())
    subjects = load_subject_train()
    total_val = np.concatenate([val for _, val in folds])
    if len(total_val) != train_len or len(np.unique(total_val)) != train_len:
        raise ValueError("Validation indices doesn't cover every training row once.")
    for k, (train, val) in enumerate(folds):
        if train.max() >= train_len or val.max() >= train_len:
            raise ValueError(f'Fold {k} has index out of training range: {train_len}.')
        if not set(subjects[train]).isdisjoint(subjects[val]):
            raise ValueError(f'Fold {k} has a participant overlap in training and validation.')
    return folds