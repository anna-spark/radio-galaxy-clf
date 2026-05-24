# reference/

These are the original scripts from Brand et al. (2023), included for
reproducibility. Our reimplementation in `src/` is based on these files
with modifications to the thresholding and rotational standardization pipeline.

> Brand, T., Leahy, J. P., & Sherwood-Taylor, H. (2023). Morphological
> classification of radio galaxies with wGAN-supported augmentation.
> *Monthly Notices of the Royal Astronomical Society*, 526(1), 282–300.

Note: `train_models.py` here depends on `tensorflow_addons`, which is no
longer maintained as of TF 2.16+. The version in `src/` removes this
dependency while keeping the SCNN architecture identical.
