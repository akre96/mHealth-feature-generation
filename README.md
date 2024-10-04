# mHealth Feature Generation
Functions for the loading, parsing, and generation of features for data from Apple HealthKit.

## Folder Overview
- `LICENSE`
- `README.md`: This file
- `mhealth_feature_generation`
    - `circadian_model.py`: Class for a cosine based model that can be fit to mHealth data streams
    - `data_cleaning.py`: Functions to help clean up common issues in HealthKit exports like overlapping entries
    - `dataloader.py`: Helps load XML exports of HealthKit data and select other formats
    - `simple_features.py`: Core of the package. Generates features from HealthKit data streams prior to a provided timestamp
    - `simple_features_daily.py`: Wrapper around `simple_features.py` functions to generate daily features
    - `generate_features.py`: Wrapper around simple_features that will generate a set of features given HealthKit Data and a timestamp
- `poetry.lock`
- `pyproject.toml`
- `tests`: Unit tests


## Usage
This is not intended to be a standalone set of code. I import functions from this as part of routine analyses done on Apple HealthKit data in relation to depression. For instructions on how to use the functions please reach out as it may depend on use case.

Wrapper functions that use many of the feature generation processes serve as examples and convenience functions. These include:
1. Daily aggregation of HealthKit data: `collectAllDailyFeatures()` in `simple_features_daily.py`
2. Aggregation for a user defined duration before a timestamp: `generateHKFeatures()` in `generate_features.py`