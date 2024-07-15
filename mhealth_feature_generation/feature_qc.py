import pandas as pd
import numpy as np


def healthKitQCFillNan(
    df,
    watch_on_threshold: float = 0.8,
    duration_threshold: float = 0.8,
    no_na_features: list = [],
) -> pd.DataFrame:
    """
    Filters a DataFrame based on the given threshold values and fills in missing values.

    Args:
        df (pd.DataFrame): The input DataFrame.
        watch_on_threshold (float, optional): The threshold value for the 'QC_watch_on_percent' column. Defaults to 0.8.
        duration_threshold (float, optional): The threshold value for the 'duration_percent' column. Defaults to 0.8.
        no_na_features (list, optional): The list of features to check for missing values. Defaults to [].

    Returns:
        pd.DataFrame: The filtered DataFrame.

    """
    qc_df = df.copy().reset_index()
    qc_df = qc_df[qc_df.QC_watch_on_percent >= watch_on_threshold]
    print(f"Watch on threshold removed {df.shape[0] - qc_df.shape[0]} rows")
    qc_df["QC_duration_percent"] = (
        qc_df["QC_duration_days"] / qc_df["QC_expected_duration_days"]
    )
    nrows = qc_df.shape[0]
    qc_df = qc_df[qc_df.QC_duration_percent >= duration_threshold]
    print(f"Duration threshold removed {nrows - qc_df.shape[0]} rows")

    # drop any activity value > 10**6
    qc_df.loc[
        qc_df.ActiveEnergyBurned_sum / qc_df.QC_duration_days > 10**5,
        ["ActiveEnergyBurned_sum", "ActiveEnergyBurned_mean"],
    ] = np.nan

    # drop any step value > 10**5
    qc_df.loc[
        qc_df.StepCount_sum / qc_df.QC_duration_days > 10**5, "StepCount_sum"
    ] = np.nan

    # Fill in missing values for sleep
    fill_in_sleep_index = qc_df[
        ~qc_df.sleep_sleepDuration_day_median.isna()
    ].index
    sleep_cat_cols = [
        "sleep_sleep_day_count",
        "sleep_bedrest_day_count",
        "sleep_Asleep_count",
        "sleep_Asleep_mean",
        "sleep_Asleep_sum",
        "sleep_Awake_count",
        "sleep_Awake_mean",
        "sleep_Awake_sum",
        "sleep_InBed_count",
        "sleep_InBed_mean",
        "sleep_InBed_sum",
        "sleep_CategoryValueUnknown_count",
        "sleep_CategoryValueUnknown_mean",
        "sleep_CategoryValueUnknown_sum",
    ]
    qc_df[sleep_cat_cols] = qc_df[sleep_cat_cols].astype(float)
    qc_df.loc[fill_in_sleep_index, sleep_cat_cols] = qc_df.loc[
        fill_in_sleep_index, sleep_cat_cols
    ].fillna(0.0)

    # Drop rows with missing values for main features
    if len(no_na_features) > 0:
        nrows = qc_df.shape[0]
        qc_df = qc_df.dropna(subset=no_na_features)
        print(
            f"Dropped {nrows - qc_df.shape[0]} rows with missing values for main features"
        )

    # If count of vital < QC_duration_days * duration_threshold, set vital agg to nan
    vital_cols = [
        "HeartRate_",
        "HeartRateVariabilitySDNN_",
        "RespiratoryRate_",
        "OxygenSaturation_",
    ]
    for col in vital_cols:
        agg_cols = [c for c in qc_df.columns if c.startswith(col)]
        qc_df.loc[
            qc_df[f"{col}count"].fillna(0)
            < qc_df.QC_duration_days * duration_threshold,
            agg_cols,
        ] = np.nan

    print(
        f"QC removed {df.shape[0] - qc_df.shape[0]} rows ({100 - 100*round(qc_df.shape[0]/df.shape[0], 2)}%)"
    )
    return qc_df
