# Plant disease classification notebooks

This repository contains two exploratory Jupyter notebooks:

- `PDD_Classification.ipynb` — a CNN with CNN-feature SVM and XGBoost experiments.
- `plant diseases.ipynb` — a CNN/EfficientNet workflow using directory/dataframe generators.

**This is a research/educational project, not a clinical or agronomic diagnostic tool.** It has not been validated for field use, and its outputs must not guide treatment, pesticide application, quarantine, or other consequential decisions. Seek qualified expert assessment. No performance or external-validation claim is made here.

## Data and setup

No dataset, trained weights, dependency lockfile, or configured ML runtime is included. A clone alone cannot run training. Obtain data only through an authorized source and verify its terms; this repository does not assert a dataset license.

For `PDD_Classification.ipynb`, set `PLANT_DISEASE_DATA_DIR` to the local dataset root containing `train/<class_name>/<image>`, or edit the relative default (`data/plantvillage`). The notebook reads up to 100 supported `.jpg`, `.jpeg`, and `.png` files from each class folder, reports missing/unreadable data, and does not change the process working directory. It creates deterministic, class-stratified, image-level train/validation/test partitions (approximately 70/15/15). Candidate selection uses validation data; the final cell evaluates only the explicitly selected candidate on the held-out partition.

That split is **not group-aware**. The data has no plant, session, or source identifiers in this repository, so duplicate/near-duplicate images or multiple photographs of one plant could cross partitions. Verify and deduplicate/group at source before interpreting any result. The held-out partition comes from the supplied training directory, not a separately verified external dataset.

For `plant diseases.ipynb`, the default split folders are `data/plantvillage/{train,valid,test}`. Override them with `PLANT_DISEASE_DATA_DIR` or `PLANT_DISEASE_TRAIN_DIR`, `PLANT_DISEASE_VALID_DIR`, and `PLANT_DISEASE_TEST_DIR`. Missing or overlapping split directories fail early. It expects class folders for training and validation; its test dataset is currently created without labels and is not used to report a test metric. The custom CNN gets normalized `[0,1]` inputs, while the EfficientNet branch keeps `[0,255]` inputs for its built-in preprocessing; both use rescale-free validation for that branch. Optional sample-image inference only runs when `PLANT_DISEASE_SAMPLE_IMAGE` points to a local image; the helper checks the file, converts it to RGB, and verifies output/class dimensions.

The notebooks do not download datasets or configure Kaggle credentials. The second notebook saves a JSON class-label/input schema next to each model so output indices can be interpreted; it contains no data paths or metrics and is ignored with generated model files by default. Never commit API keys, credential files, private data, or locally generated model files.

## Dependencies and reproducibility

The notebook imports include TensorFlow/Keras, NumPy, pandas, scikit-learn, matplotlib, seaborn, tqdm, Pillow, OpenCV, and XGBoost. This repository intentionally does not claim a compatible pinned environment: framework/platform compatibility has not been tested here. Select and record versions and hardware/runtime for your own execution, and validate the complete workflow against your data. The CI checks below use only the Python standard library; they do not install ML frameworks or execute training.

## Checks

Run the dependency-free repository checks from the clone root:

```bash
python3 scripts/check_notebooks.py
python3 -m unittest discover -s tests -v
```

They validate notebook JSON and Python syntax, enforce that saved cell outputs/execution counts are absent, and check the train/validation/test usage that can be verified statically. They do not load images, install/train models, test accuracy, or establish generalization. Clear notebook outputs before committing; they may contain private images, filenames, paths, or data previews.
