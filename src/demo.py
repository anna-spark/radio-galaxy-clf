"""
demo.py

Visualizes the preprocessing pipeline for a single galaxy: original image,
thresholded mask, and PCA-rotated coordinate overlay.

Requires galaxy_X.npy to be present in ../data/. Edit the index on line 44
to inspect different galaxies.
"""

import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import dilation, square

from data_prep import (
    norm_image,
    basic_thresh,
    remove_small_components,
    rotate_axes,
)


def rotated_for_display(rotated, img_shape, padding=5):
    """
    Scale PCA-rotated coordinates to fit within the original image for display.

    Parameters
    ----------
    rotated : ndarray, shape (2, N)
        PCA-aligned coordinates from rotate_axes.
    img_shape : tuple
        Shape of the original image (height, width).
    padding : int
        Minimum pixel margin from image edges.

    Returns
    -------
    ndarray, shape (2, N)
        Coordinates scaled and y-flipped for matplotlib imshow.
    """
    if rotated.size == 0:
        return rotated

    min_vals = rotated.min(axis=1)
    coords = rotated - min_vals[:, np.newaxis]

    max_vals = coords.max(axis=1)
    scale = min(
        (img_shape[1] - 1 - 2 * padding) / max_vals[0],
        (img_shape[0] - 1 - 2 * padding) / max_vals[1]
    )
    coords = coords * scale + padding
    coords[1, :] = img_shape[0] - 1 - coords[1, :]
    return coords


# Load dataset and pick a galaxy to inspect
X = np.load('../data/galaxy_X.npy')
img = X[24]

# Preprocess
normed  = norm_image(img)
threshed, basic = basic_thresh(normed), True   # basic_thresh returns a mask directly

# Note: if using the full pipeline from data_prep, call:
#   threshed = basic_thresh(normed, s=15)
threshed = basic_thresh(normed, s=15)
threshed = remove_small_components(threshed, thresh=10)

# PCA rotation
angle = rotate_axes(threshed)

# For display, recover pixel coordinates of the mask aligned to the rotation angle
# (rotate_axes returns an angle; here we overlay the mask itself for clarity)
rot_for_fig = None
if hasattr(angle, '__len__'):  # older version returned coords directly
    rot_for_fig = rotated_for_display(angle, img.shape)

# Plot
fig, axes = plt.subplots(1, 3, figsize=(12, 4))

axes[0].imshow(img, cmap='gray')
axes[0].set_title('Original')

axes[1].imshow(threshed, cmap='gray')
axes[1].set_title('Thresholded mask')

axes[2].imshow(img, cmap='gray')
if rot_for_fig is not None:
    axes[2].scatter(
        rot_for_fig[0, :], rot_for_fig[1, :],
        s=10, c='blue', edgecolors='white', linewidths=0.5
    )
axes[2].set_title(f'PCA rotation angle: {angle:.1f}°')
axes[2].set_aspect('equal')

plt.tight_layout()
plt.show()
