# Plant Disease Classification

Two reproducible Jupyter notebooks for image classification on a local [PlantVillage](https://github.com/spMohanty/PlantVillage-Dataset) export:

- `PDD_Classification.ipynb` loads a class-folder training split into memory, performs a **stratified, seeded validation split**, and trains a custom CNN. It also demonstrates SVM/XGBoost models on CNN features.
- `plant diseases.ipynb` consumes explicit `train/` and `valid/` class folders with Keras generators, checks class alignment and path overlap, and reports validation-only metrics. An optional EfficientNet run is included.

No dataset, trained model, or metric result is committed to this repository.

## Data setup

The dataset is intentionally not committed because of size/licensing constraints. Set `PLANTVILLAGE_ROOT` to the **extracted dataset root**:

```bash
export PLANTVILLAGE_ROOT=/absolute/path/to/PlantVillage
```

Supported layouts include either `ROOT/train/<class>/*` and `ROOT/valid/<class>/*`, or the nested PlantVillage augmented layout:

```text
ROOT/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/train/<class>/*
ROOT/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/valid/<class>/*
```

If the data is absent, the notebooks stop with a clear `FileNotFoundError`; they do not create or infer data. Supported image extensions are `.jpg`, `.jpeg`, `.png`, `.bmp`, and `.webp`.

## Install and run

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
export PLANTVILLAGE_ROOT=/absolute/path/to/PlantVillage
jupyter lab
```

Open a notebook and run cells from top to bottom. Useful controls are `PLANT_DISEASE_SEED` (default `42`), `PLANT_DISEASE_EPOCHS`, `PLANT_DISEASE_BATCH_SIZE`, and `PLANT_DISEASE_OUTPUT_DIR` (default `outputs/`). The EfficientNet notebook does **not** download ImageNet weights by default; set `PLANT_DISEASE_IMAGENET_WEIGHTS=1` only when network access and the corresponding weights are available.

## Static validation

Run this lightweight check without loading TensorFlow or training a model:

```bash
python validate_notebooks.py
```

It parses every code cell with Python AST, rejects stale notebook outputs and execution counts, checks for common non-portable paths, and runs Ruff `E9/F821` when Ruff is installed.

## Evaluation and limitations

- The first notebook's holdout is a validation split, not an independent test set. The second notebook evaluates the supplied validation split; it does not pretend that an unlabeled image directory is a test set.
- Seeds and deterministic TensorFlow operations improve repeatability but cannot guarantee bit-for-bit results across hardware and TensorFlow builds.
- Metrics are produced only after a user supplies real data and completes training; this repository makes no performance claim.
- These are educational/research artifacts, not production or medical decision systems. Check licensing, class balance, duplicate images, and external generalization before drawing conclusions.
