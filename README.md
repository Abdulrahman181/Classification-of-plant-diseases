# Plant disease classification notebooks

This repository contains two Jupyter notebooks:

- `PDD_Classification.ipynb` — a CNN and classifier-feature-extraction workflow.
- `plant diseases.ipynb` — a CNN workflow with training, validation, and prediction cells.

The repository does **not** include image data, trained weights, a dependency lockfile, or a configured runtime. A clean clone therefore is not sufficient to run either workflow end to end. No model-performance claim is made here.

## Preparing the data

Both notebooks contain dataset paths that must be changed to match your own local or hosted environment:

- In `PDD_Classification.ipynb`, set `root_dir` to the dataset location expected by that notebook. Its data-loading code reads class folders and image files below the `train` directory.
- In `plant diseases.ipynb`, set `train_dir`, `valid_dir`, and `test_dir` to directories arranged as `split/class_name/image_file`.

The training, validation, and test splits should be disjoint. The latter notebook's validation generator now uses rescaling only, matching its test generator, rather than applying training-time augmentation to validation images. Confirm the number and names of dataset classes match the model's output layer before training.

The notebooks do not download data or configure Kaggle credentials. If the data is hosted on Kaggle, obtain it through your own authorized account and follow Kaggle's current credential-handling guidance; do not commit API credentials or credential files.

## Runtime dependencies and validation

The notebooks import TensorFlow/Keras, NumPy, pandas, scikit-learn, matplotlib, seaborn, tqdm, and Pillow; `PDD_Classification.ipynb` also imports OpenCV and XGBoost. Dependency versions are not specified in this repository, so a compatible environment must be selected and recorded for reproducible training. These ML dependencies and the datasets were not available in the repository's local test environment; notebook syntax checks do not validate model execution, data loading, or accuracy.
