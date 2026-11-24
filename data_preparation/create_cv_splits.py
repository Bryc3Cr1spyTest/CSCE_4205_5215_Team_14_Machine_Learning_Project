import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

from data_preparation import CV_SPLIT_PATH, load_split

N_SPLITS = 5
SEED = 14

# Loads training data and participants
data = load_split(splits=('train',))
y = data['y_train']
y_len = len(y)
training_participants = data['subject_train']

# Generates 5 participant-grouped stratified cv folds. 
# Each participant appears in either training or validation for a given fold. 
# Then, folds creates a list of (train_idx, val_idx) index pairs for each fold,
# which correspond to indices in the original training data.
strat_k_fold = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
folds = list(strat_k_fold.split(np.zeros(y_len), y, groups=training_participants))

# Saves the cv splits to the CV_SPLIT_PATH as a .npz.
# Stores the seed, n_splits, and the per fold train_idx, val_idx, and val_subjects
CV_SPLIT_PATH.parent.mkdir(exist_ok=True)
arrays = {'seed': np.array(SEED), 'n_splits': np.array(N_SPLITS)}
for k, (train, val) in enumerate(folds):
    arrays[f'train_idx_{k}'] = train
    arrays[f'val_idx_{k}'] = val
    arrays[f'val_subjects_{k}'] = np.unique(training_participants[val])
np.savez(CV_SPLIT_PATH, **arrays)