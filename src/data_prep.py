"""
data_prep.py

Loads FITS images from the FRGMRC dataset, applies four thresholding methods
(binary, watershed, adaptive mean, adaptive Gaussian), rotationally standardizes
each mask via PCA, and saves train/val/test splits for CNN training.

Also includes utilities for comparing masks (IoU, Dice) and a multi-level Otsu
fallback used in earlier experiments.

Usage:
    python data_prep.py --data      # run preprocessing pipeline
    python data_prep.py --compare   # compute IoU/Dice agreement between methods

Dataset: https://doi.org/10.5281/zenodo.7645530
"""

import warnings
import os
import glob
import argparse

import numpy as np
import pandas as pd
import cv2 as cv
from astropy.io import fits
from scipy import ndimage as ndi
from scipy.ndimage import binary_fill_holes
from skimage import filters, segmentation, feature, color
from skimage.filters import gaussian, threshold_multiotsu
from skimage.measure import label
from skimage.morphology import dilation, square, remove_small_objects
from skimage.transform import rotate
from sklearn.model_selection import train_test_split


# -----------------------------------------------------------------------
# Saving utilities
# -----------------------------------------------------------------------

def save_original_copy(img, sub, file, folder='data_raw'):
    """Save a cropped, normalized image (before rotation)."""
    out_dir = f'{folder}/'
    os.makedirs(os.path.join(out_dir, sub), exist_ok=True)
    fname = os.path.basename(file).replace('.fits', '.npy')
    np.save(os.path.join(out_dir, sub, fname), img)

def save_segmented_copy(mask, sub, file, method='threshold', folder_prefix='data'):
    """Save a binary segmentation mask."""
    out_dir = f'{folder_prefix}_{method}/'
    os.makedirs(os.path.join(out_dir, sub), exist_ok=True)
    fname = os.path.basename(file).replace('.fits', '.npy')
    np.save(os.path.join(out_dir, sub, fname), mask)


# -----------------------------------------------------------------------
# Image normalization
# -----------------------------------------------------------------------

def norm_image(img):
    """Normalize a FITS image to [0, 1]."""
    bot, top = np.nanmin(img), np.nanmax(img)
    return (img - bot) / (top - bot) if top > bot else img


# -----------------------------------------------------------------------
# Thresholding methods
# -----------------------------------------------------------------------

def basic_thresh(image, s):
    """
    Binary thresholding from Brand et al. (2023).

    Uses percentile-based cutoffs; falls back to a simple 0.1 threshold
    for very low-noise images (s <= 14).
    """
    if s <= 14:
        return np.where(image > 0.1, 1, 0).astype(np.uint8)
    q985 = np.quantile(image, 0.985)
    q98 = np.quantile(image, 0.98)
    first = np.where(image > q985, 1, 0)
    first = remove_small_components(np.copy(first), 10)
    exten = np.where(image >= q98, 1, 0)
    se = square(3)
    prev, cur = np.copy(first), np.copy(first)
    cur = np.multiply(dilation(cur, se), exten)
    while (cur - prev).any():
        prev, cur = np.copy(cur), np.multiply(dilation(cur, se), exten)
    return cur.astype(np.uint8)


def watershed_segment(img, min_size=50, sigma_blur=1.5, min_distance=5):
    """
    Watershed segmentation tuned for faint, diffuse radio galaxies.

    Places markers at local maxima of the distance transform (rather than
    local minima) to avoid oversegmentation, then floods outward.
    """
    img = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)
    img_blur = gaussian(img, sigma=sigma_blur)
    grad = filters.sobel(img_blur)

    try:
        thresh_val = filters.threshold_otsu(img_blur)
    except ValueError:
        thresh_val = np.mean(img_blur) + 0.5 * np.std(img_blur)
    mask = img_blur > thresh_val
    if np.sum(mask) < 10:
        mask = img_blur > 0.8 * thresh_val

    distance = ndi.distance_transform_edt(mask)
    local_max = feature.peak_local_max(distance, min_distance=min_distance, labels=mask)
    markers = np.zeros_like(img_blur, dtype=int)
    for i, (r, c) in enumerate(local_max, 1):
        markers[r, c] = i
    markers = ndi.label(markers)[0]
    labels = segmentation.watershed(grad, markers, mask=mask)
    return labels


def mean_thresholding(image, block_size=31, C=3, min_size=50,
                      fill_holes=True, smooth_sigma=1.0, global_thresh_frac=0.5):
    """
    Adaptive mean thresholding for isolated central radio sources.

    Denoises first, then applies OpenCV's adaptive mean threshold with a
    global Otsu pre-filter to suppress faint background.

    Parameters
    ----------
    image : ndarray
        Normalized grayscale image (values in [0, 1]).
    block_size : int
        Neighborhood size for local mean (must be odd).
    C : int
        Constant subtracted from the local mean.
    min_size : int
        Minimum connected component size to retain.
    fill_holes : bool
        Whether to fill holes in the binary mask.
    smooth_sigma : float
        Gaussian smoothing sigma applied before thresholding.
    global_thresh_frac : float
        Fraction of Otsu threshold used as a background floor.
    """
    if image.ndim == 3:
        image = color.rgb2gray(image)

    img_smooth = filters.gaussian(image, sigma=smooth_sigma)
    img_u8 = (img_smooth * 255).astype(np.uint8)

    binary_image = cv.adaptiveThreshold(
        img_u8, 255,
        cv.ADAPTIVE_THRESH_MEAN_C,
        cv.THRESH_BINARY,
        block_size, C
    )
    binary_mask = binary_image > 0

    global_thresh = filters.threshold_otsu(img_smooth)
    binary_mask &= (img_smooth > global_thresh * global_thresh_frac)

    if fill_holes:
        binary_mask = binary_fill_holes(binary_mask)
    if min_size > 0:
        binary_mask = remove_small_objects(binary_mask, min_size=min_size)

    return binary_mask.astype(np.uint8)


def gaussian_thresholding(image, block_size=31, C=-1, min_size=50,
                           fill_holes=True, smooth_sigma=2.0,
                           global_thresh_frac=0.6, fallback=True):
    """
    Adaptive Gaussian thresholding for radio galaxy images.

    Same structure as mean_thresholding but uses a Gaussian-weighted
    neighborhood. The stricter default C=-1 partially compensates for the
    tendency of Gaussian weighting to underestimate thresholds on compact
    sources (see paper §3 for failure mode discussion).

    Parameters
    ----------
    image : ndarray
        Grayscale image, shape (H, W) or (H, W, 1).
    block_size : int
        Neighborhood size (must be odd).
    C : float
        Constant subtracted from Gaussian-weighted mean (negative = stricter).
    min_size : int
        Minimum connected component size to retain.
    fill_holes : bool
        Fill holes in the binary mask.
    smooth_sigma : float
        Gaussian smoothing sigma.
    global_thresh_frac : float
        Background floor as a fraction of Otsu threshold.
    fallback : bool
        If the mask is empty, fall back to a mean + 0.2*std threshold.
    """
    if image.ndim == 3 and image.shape[-1] == 1:
        image = image[..., 0]
    elif image.ndim == 3:
        image = color.rgb2gray(image)

    img_smooth = filters.gaussian(image, sigma=smooth_sigma)
    img_u8 = np.clip(img_smooth * 255, 0, 255).astype(np.uint8)

    binary_image = cv.adaptiveThreshold(
        img_u8, 255,
        cv.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv.THRESH_BINARY,
        block_size, C
    )
    binary_mask = binary_image > 0

    try:
        global_thresh = filters.threshold_otsu(img_smooth)
        binary_mask &= (img_smooth > global_thresh * global_thresh_frac)
    except Exception:
        pass

    if fill_holes:
        binary_mask = binary_fill_holes(binary_mask)
    if min_size > 0:
        binary_mask = remove_small_objects(binary_mask, min_size=min_size)

    if fallback and np.sum(binary_mask) == 0:
        t = img_smooth.mean() + 0.2 * img_smooth.std()
        binary_mask = img_smooth > t
        if fill_holes:
            binary_mask = binary_fill_holes(binary_mask)
        if min_size > 0:
            binary_mask = remove_small_objects(binary_mask, min_size=min_size)

    return binary_mask.astype(np.uint8)


def multilevel_thresholding(image, levels=3, min_size=50, fill_holes=True):
    """
    Multi-level Otsu thresholding for separating background, lobes, and cores.

    Returns a labeled mask with values in [0, levels-1]. Not used in the
    final paper experiments but included for completeness.

    Parameters
    ----------
    image : ndarray
        Normalized grayscale image (values in [0, 1]).
    levels : int
        Number of threshold levels.
    """
    img = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)
    img = np.clip(img, 0, 1)

    unique_vals = np.unique(img)
    if len(unique_vals) < levels:
        print("Low-variance image — falling back to binary Otsu.")
        try:
            thresh = filters.threshold_otsu(img)
            return (img > thresh).astype(np.uint8)
        except Exception:
            return (img > np.mean(img)).astype(np.uint8)

    try:
        thresholds = threshold_multiotsu(img, classes=levels, nbins=256)
        labeled = np.digitize(img, bins=thresholds)
    except ValueError:
        print("Multi-Otsu failed — reverting to binary mask.")
        try:
            thresh = filters.threshold_otsu(img)
            return (img > thresh).astype(np.uint8)
        except Exception:
            return (img > np.mean(img)).astype(np.uint8)

    for label_val in range(1, np.max(labeled) + 1):
        region = labeled == label_val
        if fill_holes:
            region = binary_fill_holes(region)
        region = remove_small_objects(region, min_size=min_size)
        labeled[region] = label_val

    return labeled.astype(np.uint8)


# -----------------------------------------------------------------------
# Connected component helpers (used by basic_thresh)
# -----------------------------------------------------------------------

def remove_small_components(img, thresh=110):
    """Remove connected components smaller than thresh pixels."""
    total = np.sum(img)
    if total < thresh:
        return img
    new_img = np.zeros_like(img)
    for r in range(img.shape[0]):
        for c in range(img.shape[1]):
            if img[r, c] == 1:
                comp, size = find_connected_component(img, r, c)
                img -= comp
                if size > thresh:
                    new_img += comp
    return new_img

def find_connected_component(img, r, c):
    """Flood-fill a single connected component starting at (r, c)."""
    comp = np.zeros_like(img)
    comp[r, c] = 1.0
    se = square(3)
    prev_comp = np.copy(comp)
    comp = np.multiply(dilation(comp, se), img)
    while (comp - prev_comp).any():
        prev_comp = np.copy(comp)
        comp = np.multiply(dilation(comp, se), img)
    return comp, np.sum(comp)


# -----------------------------------------------------------------------
# PCA-based rotational standardization
# -----------------------------------------------------------------------

def rotate_axes(img):
    """
    Compute the PCA rotation angle to align the largest mask component
    with the +x axis, avoiding 180° flips.

    Returns the rotation angle in degrees, or 0.0 if the mask is empty.
    """
    labeled = label(img)
    if labeled.max() == 0:
        return 0.0

    counts = np.bincount(labeled.flat)[1:]
    largest_label = np.argmax(counts) + 1
    mask = np.zeros_like(img)
    mask[labeled == largest_label] = 1

    coords = np.column_stack(np.nonzero(mask))
    coords_centered = coords - coords.mean(axis=0)
    u, s, vh = np.linalg.svd(coords_centered, full_matrices=False)
    vec = u[:, 0]

    if vec[0] < 0:
        vec = -vec

    angle = np.degrees(np.arctan2(vec[1], vec[0]))
    return angle


# -----------------------------------------------------------------------
# Segmentation agreement metrics
# -----------------------------------------------------------------------

def compute_iou(mask1, mask2):
    """Intersection-over-Union between two binary masks."""
    mask1, mask2 = mask1.astype(bool), mask2.astype(bool)
    inter = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    return inter / union if union > 0 else np.nan

def compute_dice(mask1, mask2):
    """Dice coefficient between two binary masks."""
    mask1, mask2 = mask1.astype(bool), mask2.astype(bool)
    inter = np.logical_and(mask1, mask2).sum()
    total = mask1.sum() + mask2.sum()
    return 2 * inter / total if total > 0 else np.nan

def evaluate_segmentation_agreement(subdirs=('BENT', 'COMP', 'FRI', 'FRII')):
    """
    Compute pairwise IoU and Dice scores across all thresholding methods
    for every galaxy in the dataset. Saves results to results/segmentation_comparison.csv.
    """
    results = []
    for sub in subdirs:
        files_thresh = glob.glob(f'data_threshold/{sub}/*.npy')
        for f_thresh in files_thresh:
            fname = os.path.basename(f_thresh)
            f_ws = f'data_watershed/{sub}/{fname}'
            f_mt = f'data_mean_thresh/{sub}/{fname}'
            f_gt = f'data_gaussian_thresh/{sub}/{fname}'
            if not all(os.path.exists(p) for p in [f_ws, f_mt, f_gt]):
                continue
            mask_t = np.load(f_thresh)
            mask_w = np.load(f_ws)
            mask_m = np.load(f_mt)
            mask_g = np.load(f_gt)
            results.append({
                'subclass': sub,
                'filename': fname,
                'IoU_thresh_watershed':  compute_iou(mask_t, mask_w),
                'Dice_thresh_watershed': compute_dice(mask_t, mask_w),
                'IoU_thresh_mean':       compute_iou(mask_t, mask_m),
                'Dice_thresh_mean':      compute_dice(mask_t, mask_m),
                'IoU_thresh_gaussian':   compute_iou(mask_t, mask_g),
                'Dice_thresh_gaussian':  compute_dice(mask_t, mask_g),
                'IoU_mean_gaussian':     compute_iou(mask_m, mask_g),
                'Dice_mean_gaussian':    compute_dice(mask_m, mask_g),
            })
    df = pd.DataFrame(results)
    os.makedirs('results', exist_ok=True)
    df.to_csv('results/segmentation_comparison.csv', index=False)
    print("\nSegmentation agreement summary:")
    print(df.describe())
    return df


# -----------------------------------------------------------------------
# Main preprocessing pipeline
# -----------------------------------------------------------------------

def load_galaxy_data_and_split(cnn_input_method='mean_thresh'):
    """
    Load FITS images, run all four segmentations, and save train/val/test splits.

    Splits are stratified 80/10/10 (approximately) using random_state=42
    for reproducibility. Preprocessed arrays are saved under data/.

    Parameters
    ----------
    cnn_input_method : str
        Which mask to use as CNN input. One of:
        'threshold', 'watershed', 'mean_thresh', 'gaussian_thresh', 'multilevel'.
    """
    valid_methods = ['threshold', 'watershed', 'mean_thresh', 'gaussian_thresh', 'multilevel']
    if cnn_input_method not in valid_methods:
        raise ValueError(f"cnn_input_method must be one of {valid_methods}")

    if not os.path.exists('FITS/'):
        warnings.warn('FITS data not found. Download from https://doi.org/10.5281/zenodo.7645530', RuntimeWarning)
        return

    rootdir = 'FITS/'
    subdirs = ['BENT', 'COMP', 'FRI', 'FRII']
    label_dict = {'BENT': [1,0,0,0], 'COMP': [0,1,0,0], 'FRI': [0,0,1,0], 'FRII': [0,0,0,1]}

    X_raw, y = [], []

    for sub in subdirs:
        files = glob.glob(os.path.join(rootdir, sub, '*.fits'))
        for file in files:
            img = fits.open(file)[0].data
            focus = img[75:225, 75:225].astype(np.float32)
            focus = norm_image(focus)
            focus = np.nan_to_num(focus, nan=0.0, posinf=1.0, neginf=0.0)

            # Compute all masks
            mask_t  = basic_thresh(focus, s=15).astype(np.uint8)
            mask_w  = (watershed_segment(focus) > 0).astype(np.uint8)
            mask_m  = (mean_thresholding(focus, block_size=31, C=1, min_size=50) > 0).astype(np.uint8)
            mask_g  = gaussian_thresholding(focus)
            mask_g  = (mask_g > np.percentile(mask_g, 90)).astype(np.uint8)
            mask_ml = multilevel_thresholding(focus, levels=3)
            mask_ml = (mask_ml == mask_ml.max()).astype(np.uint8)

            # Save originals and all masks
            save_original_copy(focus, sub, file, folder='data_raw')
            save_segmented_copy(mask_t,  sub, file, 'threshold',      'data')
            save_segmented_copy(mask_w,  sub, file, 'watershed',      'data')
            save_segmented_copy(mask_m,  sub, file, 'mean_thresh',    'data')
            save_segmented_copy(mask_g,  sub, file, 'gaussian_thresh','data')
            save_segmented_copy(mask_ml, sub, file, 'multilevel',     'data')

            # Select CNN input
            method_map = {
                'threshold':      mask_t,
                'watershed':      mask_w,
                'mean_thresh':    mask_m,
                'gaussian_thresh':mask_g,
                'multilevel':     mask_ml,
            }
            cnn_img = method_map[cnn_input_method].astype(np.float32)[..., np.newaxis]

            X_raw.append(cnn_img)
            y.append(label_dict[sub])
            print(f"Processed {file}")

    X_raw = np.array(X_raw, dtype=np.float32)
    y = np.array(y, dtype=np.uint8)

    X_train1, X_test1, y_train, y_test = train_test_split(
        X_raw, y, test_size=0.1, random_state=42, stratify=y
    )
    X_train1, X_val1, y_train, y_val = train_test_split(
        X_train1, y_train, test_size=0.11, random_state=42, stratify=y_train
    )

    os.makedirs('data', exist_ok=True)
    np.save('data/galaxy_X_train1.npy', X_train1)
    np.save('data/galaxy_y_train.npy',  y_train)
    np.save('data/galaxy_X_val1.npy',   X_val1)
    np.save('data/galaxy_y_val.npy',    y_val)
    np.save('data/galaxy_X_test1.npy',  X_test1)
    np.save('data/galaxy_y_test.npy',   y_test)

    print(f"\nSplits saved to data/  —  train: {len(X_train1)}, val: {len(X_val1)}, test: {len(X_test1)}")
    print("Done.")


# -----------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Preprocess FRGMRC FITS images and compare segmentation methods."
    )
    parser.add_argument('--data',    action='store_true', help="Run preprocessing pipeline")
    parser.add_argument('--compare', action='store_true', help="Compute IoU/Dice between methods")
    args = parser.parse_args()

    if args.data:
        load_galaxy_data_and_split()
    elif args.compare:
        evaluate_segmentation_agreement()
    else:
        parser.print_help()
