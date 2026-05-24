# Radio galaxy morphological classification

Code for our paper submitted to *Research Notes of the AAS*:

> **Thresholding algorithms for PCA-based rotational standardization of radio galaxy images**

We compare four image thresholding methods — binary (Brand et al. 2023), watershed
segmentation, adaptive mean, and adaptive Gaussian — as preprocessing steps for CNN
morphological classification of radio galaxies. Adaptive mean thresholding achieves
the highest macro F1 among standardization methods (0.917), narrowing the gap with
rotational augmentation (0.938) while training ~5× faster.

## Results summary

| Preprocessing        | Bent F1 | Compact F1 | FRI F1 | FRII F1 | Macro F1 | Time (s) |
|----------------------|---------|------------|--------|---------|----------|----------|
| None                 | 0.837   | 0.976      | 0.865  | 0.930   | 0.902    | 932      |
| Binary (Brand 2023)  | 0.850   | 0.837      | 0.750  | 0.909   | 0.837    | 538      |
| Watershed            | 0.923   | 0.933      | 0.875  | 0.921   | 0.913    | 898      |
| **Mean thresholding**| **0.923**| **0.930** | **0.875**| **0.923**| **0.917**| **817** |
| Gaussian thresholding| 0.850   | 0.844      | 0.750  | 0.907   | 0.838    | 734      |
| Augmentation         | 0.950   | 0.933      | 0.909  | 0.946   | 0.938    | 4007     |

## Dataset

The FRGMRC dataset (960 radio galaxies, four classes: bent, compact, FRI, FRII)
is publicly available on Zenodo:

```
https://doi.org/10.5281/zenodo.7645530
```

Download and place the FITS files under `FITS/` before running preprocessing.

## Setup

```bash
pip install numpy scipy scikit-image scikit-learn astropy opencv-python tensorflow pandas matplotlib
```

## Usage

**Preprocess all images** (generates masks for all four methods):
```bash
cd src
python data_prep.py --data
```

**Compare masks** (computes IoU/Dice, saves to `results/segmentation_comparison.csv`):
```bash
python data_prep.py --compare
```

**Train the SCNN**:
```bash
python train_models.py
```
Edit `data_type` at the bottom of `train_models.py` to switch between datasets.

**Visualize a single galaxy**:
```bash
python demo.py
```

**Watershed dataset utility** (if generating watershed splits separately):
```bash
python add_ws_data.py
```

## Repository structure

```
radio-galaxy-clf/
├── src/
│   ├── data_prep.py        # preprocessing pipeline (all four methods)
│   ├── train_models.py     # SCNN training
│   ├── add_ws_data.py      # watershed dataset assembly utility
│   └── demo.py             # single-galaxy visualization
├── notebooks/
│   └── analysis_plots.ipynb  # figures from the paper
├── reference/
│   ├── data_prep.py        # original Brand et al. (2023) preprocessing
│   ├── train_models.py     # original Brand et al. (2023) training code
│   └── README.md
└── .gitignore
```

## Citation

If you use this code, please also cite the original Brand et al. (2023) paper
and dataset, as our SCNN architecture and baseline thresholding method are
taken directly from their work.

## Acknowledgments

We thank Dr. Shyamal Mitra and peer mentors Vivek Abraham, Joel Deville, and
Sana Kohli for their guidance through the Geometry of Space stream at UT Austin.
