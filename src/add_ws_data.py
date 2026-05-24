"""
add_ws_data.py

Assembles the watershed-segmented dataset from pre-saved .npy mask files
and creates stratified train/val/test splits. Run this after generating
watershed masks with data_prep.py --data.

Expected input structure:
    data_watershed/
        BENT/*.npy
        COMP/*.npy
        FRI/*.npy
        FRII/*.npy

Output is written to data_ws/.
"""

import os
import glob
import numpy as np
from sklearn.model_selection import train_test_split


LABEL_DICT = {
    'BENT': [1, 0, 0, 0],
    'COMP': [0, 1, 0, 0],
    'FRI':  [0, 0, 1, 0],
    'FRII': [0, 0, 0, 1],
}


def make_watershed_dataset(subdirs=('BENT', 'COMP', 'FRI', 'FRII')):
    """
    Load watershed masks from data_watershed/ and save a train/val/test
    split to data_ws/.

    Splits are stratified with random_state=42 (80/10/10 approximately).
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_dir  = os.path.join(script_dir, '..', 'data_watershed')
    output_dir = os.path.join(script_dir, '..', 'data_ws')
    os.makedirs(output_dir, exist_ok=True)

    X, y = [], []
    for sub in subdirs:
        folder = os.path.join(input_dir, sub)
        files  = glob.glob(os.path.join(folder, '*.npy'))
        print(f"{sub}: {len(files)} files")
        if not files:
            print(f"  Warning: no .npy files found in {folder}")
            continue
        for f in files:
            X.append(np.load(f))
            y.append(LABEL_DICT[sub])

    if not X:
        print("No files found. Check that data_watershed/ exists and is populated.")
        return

    X = np.array(X)
    y = np.array(y)

    np.save(os.path.join(output_dir, 'galaxy_X_ws.npy'), X)
    np.save(os.path.join(output_dir, 'galaxy_y_ws.npy'), y)
    print(f"\nSaved full dataset → {output_dir}")

    if len(X) < 10:
        print(f"Only {len(X)} samples — skipping train/val/test split.")
        return

    X_tmp, X_test, y_tmp, y_test = train_test_split(
        X, y, test_size=0.1, random_state=42, shuffle=True, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_tmp, y_tmp, test_size=0.11, random_state=42, shuffle=True, stratify=y_tmp
    )

    np.save(os.path.join(output_dir, 'galaxy_X_train1_ws.npy'), X_train)
    np.save(os.path.join(output_dir, 'galaxy_y_train_ws.npy'),  y_train)
    np.save(os.path.join(output_dir, 'galaxy_X_val1_ws.npy'),   X_val)
    np.save(os.path.join(output_dir, 'galaxy_y_val_ws.npy'),    y_val)
    np.save(os.path.join(output_dir, 'galaxy_X_test1_ws.npy'),  X_test)
    np.save(os.path.join(output_dir, 'galaxy_y_test_ws.npy'),   y_test)

    print(f"Splits saved  —  train: {len(X_train)}, val: {len(X_val)}, test: {len(X_test)}")


if __name__ == '__main__':
    make_watershed_dataset()
