import numpy as np
from typing import List, Dict, Tuple
from sklearn.model_selection import GroupKFold

from .get_json import get_data_from_json


def group_child_id_split(
    all_data: List[Dict], 
    n_splits: int, 
    fold: int = 0
) -> Tuple[List[Dict], List[Dict]]:
    
    # 1. Convert to NumPy array for speed (CRITICAL for large datasets)
    groups = np.array([item["child_id"] for item in all_data])
    
    # Create a dummy X of the same length
    dummy_X = np.zeros(len(all_data))
    
    # import hashlib and compute hash to check all_data is the same across runs
    import hashlib
    data_hash = hashlib.md5(str(all_data).encode()).hexdigest()
    print(f"Data hash: {data_hash}")

    # 2. Perform the split
    gkf = GroupKFold(n_splits=n_splits)
    
    # 3. Instead of converting the whole generator to a list (which hangs),
    # iterate only until we reach the fold we need.
    train_idx, val_idx = None, None
    for i, (t_idx, v_idx) in enumerate(gkf.split(dummy_X, groups=groups)):
        if i == fold:
            # CONVERT TO NATIVE PYTHON LISTS HERE
            train_idx = t_idx.tolist()
            val_idx = v_idx.tolist()
            break

    print(val_idx[:20])

    if train_idx is None:
        raise ValueError(f"Fold {fold} is out of bounds for n_splits={n_splits}")
        
    # 'i' is a native Python integer
    train_data = [all_data[i] for i in train_idx]
    val_data = [all_data[i] for i in val_idx]
    
    print(f"--- Fold {fold+1} Split Info ---")
    print(f"Train utterances: {len(train_data)}")
    print(f"Val utterances:   {len(val_data)}")
    print()
    
    # Quick sanity check to ensure no leakage
    train_children = set(item["child_id"] for item in train_data)
    val_children = set(item["child_id"] for item in val_data)
    # print(val_children)
    leakage = train_children.intersection(val_children)
    assert len(leakage) == 0, f"Data leakage detected! Overlapping children: {leakage}"
    
    return train_data, val_data

def simple_random_split(
    all_data: List[Dict],
    test_size: float = 0.1,
    random_seed: int = 42,
    fold: int = 0,
) -> Tuple[List[Dict], List[Dict]]:
    """
    Simple random 90/10 train/val split (no cross-validation).

    If entries have a 'child_id' field, the split is grouped so that
    no child appears in both train and val (prevents leakage).
    The `fold` argument is accepted for interface compatibility but ignored.
    """
    from sklearn.model_selection import GroupShuffleSplit, ShuffleSplit

    has_child_id = all("child_id" in item for item in all_data)
    n = len(all_data)
    dummy_X = np.zeros(n)

    if has_child_id:
        groups = np.array([item["child_id"] for item in all_data])
        splitter = GroupShuffleSplit(
            n_splits=1, test_size=test_size, random_state=random_seed
        )
        train_idx, val_idx = next(splitter.split(dummy_X, groups=groups))
    else:
        splitter = ShuffleSplit(
            n_splits=1, test_size=test_size, random_state=random_seed
        )
        train_idx, val_idx = next(splitter.split(dummy_X))

    train_data = [all_data[i] for i in train_idx.tolist()]
    val_data = [all_data[i] for i in val_idx.tolist()]

    print(f"--- Simple Split (test_size={test_size}) ---")
    print(f"Train utterances: {len(train_data)}")
    print(f"Val utterances:   {len(val_data)}")
    print()

    if has_child_id:
        train_children = set(item["child_id"] for item in train_data)
        val_children = set(item["child_id"] for item in val_data)
        assert len(train_children & val_children) == 0, "Data leakage detected!"

    return train_data, val_data


if __name__ == "__main__":
    # Quick debug test
    from omegaconf import OmegaConf
    cfg = OmegaConf.load("configs/default.yaml")
    all_data = get_data_from_json(cfg)

                
    print(f"Loaded {len(all_data)} total utterances from disk.")

    train_data, val_data = group_child_id_split(all_data=all_data, n_splits=cfg.cv.n_folds, fold=0)

    # check if splits are deterministic
    train_data_2, val_data_2 = group_child_id_split(all_data=all_data, n_splits=cfg.cv.n_folds, fold=0)
    print("Train splits are deterministic:", train_data == train_data_2)
    print("Val splits are deterministic:", val_data == val_data_2)

    # get the set of utterance IDs and compare between train_data1 and train_data_2
    train_utt_ids_1 = set(item["utterance_id"] for item in train_data)
    train_utt_ids_2 = set(item["utterance_id"] for item in train_data_2)
    assert train_utt_ids_1 == train_utt_ids_2, "Train utterance IDs differ between splits!"
    print("Train utterance IDs are consistent between splits.")

    # print the first few utterance IDs from train_data_1 and train_data_2 to visually confirm they match
    print("First 5 utterance IDs from train_data_1:")
    for item in train_data[:5]:
        print(item["utterance_id"])
    print("First 5 utterance IDs from train_data_2:")
    for item in train_data_2[:5]:
        print(item["utterance_id"])