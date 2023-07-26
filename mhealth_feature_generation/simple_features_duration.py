""" Preprocessing steps for Apple Health data
Requires HealthKit data from Cassandra to be cleaned / wide format
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Literal
import pingouin as pg
try:
    from simple_features_daily import dailySleepFeatures
    from circadian_model import CircadianModel
except ModuleNotFoundError:
    from .simple_features_daily import dailySleepFeatures
    from .circadian_model import CircadianModel

durationType = pd.Timedelta | Literal["today", "yesterday"] | str


# TODO: Allow variable binning of features during duration
def collectFeatures(
    hk_data: pd.DataFrame,
    user_id: int,
    timestamp: pd.Timestamp,
    duration: durationType,
    watch_on_resample: str = "1h",
) -> pd.DataFrame:
    """Gather health kit data features for a user_id around a timestamp

    Args:
        hk_data (pd.DataFrame): Loaded Apple Health data
        user_id (int): user ID
        timestamp (pd.Timestamp): timestamp to center data around
        duration (pd.Timedelta): duration to collect data before timestamp

    Returns:
        pd.DataFrame: features for user_id around timestamp, 1 row
    """
    subset = getDurationAroundTimestamp(hk_data, user_id, timestamp, duration)
    n_dates = subset["local_start"].dt.date.nunique()
    paee = processActiveDuration(subset, "ActiveEnergyBurned")
    exercise_time = processActiveDuration(subset, "AppleExerciseTime")
    steps = processActiveDuration(subset, "StepCount")
    hr = aggregateVital(subset, "HeartRate", resample="1h", range=(30, 200))
    hrv = aggregateVital(
        subset,
        "HeartRateVariabilitySDNN",
        resample="1h",
        aggregations=["mean", "count"],
    )
    o2sat = aggregateVital(
        subset,
        "OxygenSaturation",
        resample="1h",
        aggregations=["mean", "count"],
    )
    rr = aggregateVital(subset, "RespiratoryRate", resample="1h")
    standing = processEventLog(subset, ["AppleStandHour"])
    sleep = processSleep(subset)

    features = pd.concat(
        [sleep, paee, hr, hrv, rr, o2sat, standing, exercise_time, steps],
        axis=1,
    )

    # Look to see if watch on in last hour for calculating if watch is on for duration < 1hour
    if (type(duration) != str) and (duration < pd.Timedelta("1h")):
        watch_on_subset = getDurationAroundTimestamp(
            hk_data, user_id, timestamp, pd.Timedelta("1h")
        )
        features["watch_on_percent"] = processWatchOnPercent(
            watch_on_subset, resample=watch_on_resample
        )
    else:
        features["watch_on_percent"] = processWatchOnPercent(
            subset, resample=watch_on_resample
        )
    features["user_id"] = user_id
    features["timestamp"] = timestamp
    features["duration"] = duration
    features["n_days"] = n_dates
    return features


def qcWatchData(
    features: pd.DataFrame, watch_on_threshold: float = 80
) -> pd.DataFrame:
    watch_feature_roots = [
        "HeartRate",
        "HeartRateVariabilitySDNN",
        "OxygenSaturation",
        "RespiratoryRate",
        "ActiveEnergyBurned",
        "AppleExerciseTime",
        "AppleStandHour",
        "Sleep",
    ]
    watch_cols = [
        col
        for col in features.columns
        if any([col.startswith(root) for root in watch_feature_roots])
    ]
    qc_features = features.copy()
    duration_cols = [
        col for col in features.columns if col.endswith("duration")
    ]
    value_cols = [
        col for col in features.columns if not col.endswith("duration")
    ]
    # Don't fill 0 for heart rate
    fill_value_cols = [
        c for c in value_cols if not ((c.startswith("HEART") or c.startswith("RESPIRATORY") or c.startswith('OXYGEN')) and not c.endswith("count"))
    ]
    qc_features[fill_value_cols] = qc_features[fill_value_cols].fillna(0)
    qc_features[duration_cols] = qc_features[duration_cols].fillna(
        pd.Timedelta(0)
    )
    qc_features.loc[
        qc_features["watch_on_percent"] < watch_on_threshold, watch_cols
    ] = np.nan
    return qc_features


def getDurationAroundTimestamp(
    hk_data: pd.DataFrame,
    user_id: int,
    timestamp: pd.Timestamp,
    duration: durationType,
):
    """Returns a DataFrame with data from user_id overlapping with duration around a timestamp"""
    # Note returns data that has either a start OR end date within the duration
    # This means that data may be included that is not within the duration

    if duration in ["today", "yesterday"]:
        if duration == "today":
            start = pd.to_datetime(timestamp.date())
        if duration == "yesterday":
            start = pd.to_datetime((timestamp - pd.Timedelta("1d")).date())

        # If duration today and timestamp before 4am then get data from previous date
        # TODO: Make function of last sleep session
        if timestamp.hour < 4:
            start = pd.to_datetime((timestamp - pd.Timedelta("1d")).date())
        end = start + pd.Timedelta("1d")
    elif type(duration) == str:
        duration = pd.Timedelta(duration)
        start = timestamp - duration
        end = timestamp
    else:
        start = timestamp - duration
        end = timestamp

    subset = hk_data[
        (hk_data["user_id"] == user_id)
        & (
            ((hk_data["local_end"] <= end) & (hk_data["local_end"] >= start))
            | (
                (hk_data["local_start"] <= end)
                & (hk_data["local_start"] >= start)
            )
        )
    ]
    return subset


def processWatchOnPercent(
    hk_data: pd.DataFrame, resample: str = "1h"
) -> float:
    hr = hk_data[hk_data.type == "HeartRate"]
    watch_bool = (
        hr["device.name"].str.contains("Apple Watch")
    )
    watch_bool = watch_bool.fillna(False)
    watch_hr = hr.loc[watch_bool, ["local_start", "value"]]
    watch_hr_logs = watch_hr.set_index("local_start").resample(resample).count()
    if watch_hr.empty:
        return 0
    return (
        100
        * watch_hr_logs[watch_hr_logs["value"] > 0].shape[0]
        / watch_hr_logs.shape[0]
    )

def aggregateDailySleep(
    hk_data: pd.DataFrame,
) -> pd.DataFrame:
    sleep_daily = dailySleepFeatures(hk_data)
    sleep_agg = sleep_daily[
            [
            'sleep_bedrestDuration_day',
            'sleep_sleepDuration_day',
            'sleep_sleepEfficiency_day',
            'sleep_sleepOnsetLatency_day',
            'sleep_sleepHR_day',
            'sleep_sleepHRV_day',
            ]
    ].aggregate(["median", "min", "max", "std"]).unstack()
    sleep_agg[
        ('sleep_sleep_day', 'count')
    ] = sleep_daily['sleep_sleepDuration_day'].count()
    sleep_agg[
        ('sleep_bedrest_day', 'count')
    ] = sleep_daily['sleep_bedrestDuration_day'].count()
    sleep_agg = pd.DataFrame(sleep_agg).T
    sleep_agg.columns = [f"{x}_{y}" for (x, y) in sleep_agg.columns]
    return sleep_agg


def processSleep(hk_data: pd.DataFrame) -> pd.DataFrame:
    """Process sleep time interval data from Apple HealthKit

    Aggregation of duration reported in HOURS
    Args:
        hk_data (pd.DataFrame): HealthKit data

    Returns:
        pd.DataFrame: Statistics on sleep time intervals
    """
    sleep = (
        hk_data.loc[
            hk_data.type == "SleepAnalysis",
            ["local_start", "local_end", "body.category.value"],
        ]
        .rename(columns={"body.category.value": "SleepAnalysis"})
        .sort_values(by="local_start")
    )
    sleep["duration"] = sleep["local_end"] - sleep["local_start"]
    sleep_agg = sleep.groupby("SleepAnalysis")["duration"].aggregate(
        ["sum", "mean", "count"]
    )
    sleep_agg["x"] = "x"
    s2 = (
        sleep_agg.reset_index()
        .melt(
            id_vars=["x", "SleepAnalysis"],
            value_vars=["sum", "mean", "count"],
        )
        .pivot_table(
            index="x",
            columns=["SleepAnalysis", "variable"],
            values="value",
            aggfunc="first",
        )
        .reset_index()
    ).drop(columns="x", level=0)
    s2.columns = ["sleep_" + "_".join(col).strip("_") for col in s2.columns]
    duration_cols = [
        col
        for col in s2.columns
        if col.endswith("sum") or col.endswith("mean")
    ]
    s2[duration_cols] = s2[duration_cols].applymap(
        lambda x: pd.Timedelta(x) / pd.Timedelta("1h")
    )
    return s2


def processEventLog(
    hk_data: pd.DataFrame, event_types: List[str]
) -> pd.DataFrame:
    types = hk_data["type"].unique()
    # Check that event_type is in types
    for event_type in event_types:
        if event_type not in types:
            # print(f"ERROR: {event_type} not in {types}")
            return None

    events = hk_data.loc[
        hk_data.type.isin(event_types), ["local_start", "type"]
    ]
    return pd.DataFrame(events.type.value_counts()).T.reset_index(drop=True)


def processActiveDuration(hk_data: pd.DataFrame, hk_type: str) -> pd.DataFrame:
    if hk_type not in ["StepCount", "AppleExerciseTime", "ActiveEnergyBurned"]:
        raise ValueError(
            f"Invalid hk_type: {hk_type}, must be ActiveEnergyBurned, StepCount or AppleExerciseTime"
        )
    steps = (
        hk_data.loc[
            hk_data.type == hk_type,
            ["local_start", "local_end", "value"],
        ]
        .rename(columns={"value": hk_type})
        .sort_values(by="local_start")
    )
    steps["duration"] = steps["local_end"] - steps["local_start"]
    steps["prev_local_end"] = steps["local_end"].shift()
    steps["prev_local_start"] = steps["local_start"].shift()
    steps["overlap"] = (steps["local_start"] < steps["prev_local_end"]) & (
        steps["local_end"] > steps["prev_local_start"]
    )

    # Remove overlapping rows
    # TODO: Review how overlaps are being handled
    if steps["overlap"].any():
        steps = steps[~steps["overlap"]]
        steps.drop(
            columns=["prev_local_end", "prev_local_start", "overlap"],
            inplace=True,
        )

    steps_agg = pd.DataFrame(steps[hk_type].aggregate(["sum", "count"])).T
    steps_agg.columns = [f"{hk_type}_{col}" for col in steps_agg.columns]
    steps_agg[f"{hk_type}_duration"] = pd.to_timedelta(
        steps["duration"].sum()
    ) / pd.Timedelta("1h")

    return steps_agg.reset_index(drop=True)


def aggregateVital(
    hk_data: pd.DataFrame,
    vital_type: str,
    resample="1h",
    standard_aggregations: List[str] = [
        "mean",
        "std",
        "min",
        "max",
        "count",
        "median",
        "skew",
        "kurtosis",
    ],
    linear_time_aggregations: bool = True,
    circadian_model_aggregation: bool = False,
    range: Tuple[float, float] | None = None,
) -> pd.DataFrame:
    vital = hk_data.loc[
        hk_data.type == vital_type,
        ["local_start", "value"],
    ].rename(columns={"value": vital_type})
    if vital.empty:
        return pd.DataFrame()

    if range is not None:
        vital = vital[vital[vital_type].between(range[0], range[1])]
    vital_resamp = vital.set_index("local_start").resample(resample).median()
    vital_agg = pd.DataFrame(vital_resamp.aggregate(standard_aggregations)).T
    vital_agg.columns = [f"{vital_type}_{col}" for col in vital_agg.columns]

    # Add time domain features
    if linear_time_aggregations:
        resamp_nona = vital_resamp.dropna()
        if resamp_nona.shape[0] < 3:
            return vital_agg.reset_index(drop=True)
        time_hours = (resamp_nona.index - resamp_nona.index[0]) / pd.Timedelta("1h")
        regression = pg.linear_regression(time_hours, resamp_nona[vital_type])
        vital_agg[f"{vital_type}_intercept"] = regression["coef"].values[0]
        vital_agg[f"{vital_type}_slope"] = regression["coef"].values[1]
    if circadian_model_aggregation:
        resamp_nona = vital_resamp.dropna()
        if resamp_nona.shape[0] < 3:
            return vital_agg.reset_index(drop=True)
        time_hours = (resamp_nona.index - resamp_nona.index[0]) / pd.Timedelta("1h")
        bounds = (0, [200, 200, 24, 48])
        p0 = [50, 50, 12, 24]
        model = CircadianModel(
            bounds=bounds,
            init_params=p0
        )        
        model.fit(
            time_hours,
            resamp_nona[vital_type],
        )
        params = model.parameters
        vital_agg[f"{vital_type}_circadian_mesor"] = params[0]
        vital_agg[f"{vital_type}_circadian_amplitude"] = params[1]
        vital_agg[f"{vital_type}_circadian_acrophase"] = params[2]
        vital_agg[f"{vital_type}_circadian_period"] = params[3]

    return vital_agg.reset_index(drop=True)
