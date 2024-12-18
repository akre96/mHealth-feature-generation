# Wrapper function for feature generation
"""
Generate HealthKit features for a given user within a specified duration around a timestamp.

Parameters:
user_hk (DataFrame): The HealthKit data for the user.
user_id (str): The unique identifier for the user.
timestamp (datetime): The reference timestamp around which features are generated.
duration (durationType): The duration around the timestamp for which features are generated.

Returns:
DataFrame: A DataFrame containing the generated features.

The function performs the following steps:
1. Extracts data around the given timestamp and duration.
2. Aggregates various sleep-related features.
3. Aggregates active duration features for specified types.
4. Aggregates audio exposure features for different contexts.
5. Aggregates vital signs features for specified types and contexts.
6. Computes Lomb-Scargle features for vital signs.
7. Concatenates all aggregated features into a single DataFrame.
8. Adds quality control (QC) information to the DataFrame.
9. Adds user and survey metadata to the DataFrame.

Note:
- If the extracted data is empty, an empty DataFrame is returned.
- A warning is printed if the end of the window does not match the end of the data.
"""
import pandas as pd
import numpy as np

from .simple_features import (
    aggregateActiveDuration,
    aggregateAudioExposure,
    aggregateDailySleep,
    aggregateSleepCategories,
    aggregateVital,
    calcStartStop,
    getDurationAroundTimestamp,
    processWatchOnPercent,
    processWatchOnTime,
)
from .timedomain_features import (
    getLombScargleFeatures,
)
from .utils import durationType


def generateHKFeatures(user_hk, user_id, timestamp, duration: durationType):
    data = getDurationAroundTimestamp(user_hk, user_id, timestamp, duration)
    if data.empty:
        return pd.DataFrame()

    sleep_aggregations = [
        "sleep_sleepDuration_day",
        "sleep_bedrestDuration_day",
        "sleep_sleepHR_day",
        "sleep_sleepHRV_day",
        "sleep_wakeAfterSleepOnset_day",
        "sleep_sleepEfficiency_day",
        "sleep_sleepOnsetLatency_day",
        "sleep_bedrestOnsetHours_day",
        "sleep_bedrestOffsetHours_day",
        "sleep_sleepOnsetHours_day",
        "sleep_sleepOffsetHours_day",
        "sleep_bedrestNoise_day",
    ]
    sleep_agg = aggregateDailySleep(data, sleep_features=sleep_aggregations)
    sleep_cat_agg = aggregateSleepCategories(data)
    active_duration_aggregations = [
        "ActiveEnergyBurned",
        "BasalEnergyBurned",
        "AppleExerciseTime",
        "StepCount",
    ]
    active_duration_agg = [
        aggregateActiveDuration(data, duration_type, resample="1D", qc=True)
        for duration_type in active_duration_aggregations
    ]
    noise_agg = [
        aggregateAudioExposure(data, resample="1h", context=context)
        for context in ["all", "bedrest"]
    ]
    vital_aggregations = [
        ("HeartRate", (30, 200)),
        ("HeartRateVariabilitySDNN", (0, 1)),
        ("OxygenSaturation", (0.5, 1)),
        ("RespiratoryRate", (0.1, 100)),
    ]
    vital_agg = [
        aggregateVital(
            data,
            vital_type,
            vital_range=vital_range,
            resample="1h",
            context=context,
        )
        for vital_type, vital_range in vital_aggregations
        for context in ["all", "bedrest"]
    ]

    for vital, range in vital_aggregations:
        vital_data = data[data["type"] == vital].copy()
        vital_data["value"] = pd.to_numeric(
            vital_data["value"], errors="coerce"
        )
        vital_data = (
            vital_data.loc[
                vital_data["value"].between(range[0], range[1]),
                ["value", "local_start"],
            ]
            .dropna()
            .drop_duplicates()
        )
        vital_data["hours"] = (
            vital_data["local_start"] - vital_data["local_start"].min()
        ) / pd.Timedelta(hours=1)
        vital_data["hours"] = vital_data["hours"].astype(float)
        vital_data = vital_data.sort_values(by="hours")

        vital_hours = vital_data["hours"].to_numpy()
        signal = vital_data["value"].astype(float).to_numpy()
        lsFeatures = getLombScargleFeatures(time=vital_hours, signal=signal)

        vital_agg += [lsFeatures.add_prefix(vital + "_")]

    hk_metrics = pd.concat(
        [
            sleep_agg,
            sleep_cat_agg,
            *active_duration_agg,
            *noise_agg,
            *vital_agg,
        ],
        axis=1,
    )

    # Add QC info
    start, end = calcStartStop(timestamp, duration)
    if end != timestamp:
        print("Warning: end of window does not match end of data")

    hk_metrics["QC_watch_on_percent"] = processWatchOnPercent(
        data, resample="1h", origin=start, end=timestamp
    )
    hk_metrics["QC_watch_on_hours"] = processWatchOnTime(
        data, resample="1h", origin=start
    )
    hk_metrics["QC_duration_days"] = (
        data["local_start"].max() - data["local_start"].min()
    ) / np.timedelta64(1, "D")
    hk_metrics["QC_ndates"] = data["local_start"].dt.date.nunique()

    hk_metrics["user_id"] = user_id
    hk_metrics["survey_start"] = timestamp
    hk_metrics["QC_expected_duration"] = duration
    hk_metrics["QC_expected_duration_days"] = pd.to_timedelta(
        duration
    ) / pd.Timedelta(days=1)

    return hk_metrics
