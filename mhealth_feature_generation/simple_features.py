""" Preprocessing steps for Apple Health data
Requires HealthKit data from Cassandra to be cleaned / wide format
Allows aggregation of features over a defined time period

"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Literal
import pingouin as pg

try:
    from circadian_model import CircadianModel
    from data_cleaning import combineOverlaps
    from data_cleaning import combineOverlapsSleep
except ModuleNotFoundError:
    from .circadian_model import CircadianModel
    from .data_cleaning import combineOverlaps
    from .data_cleaning import combineOverlapsSleep

durationType = pd.Timedelta | Literal["today", "yesterday"] | str


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
    paee = aggregateActiveDuration(subset, "ActiveEnergyBurned")
    exercise_time = aggregateActiveDuration(subset, "AppleExerciseTime")
    steps = aggregateActiveDuration(subset, "StepCount")
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
    standing = countEventLog(subset, ["AppleStandHour"])
    sleep = aggregateSleepCategories(subset)

    features = pd.concat(
        [sleep, paee, hr, hrv, rr, o2sat, standing, exercise_time, steps],
        axis=1,
    )
    start, _ = calcStartStop(timestamp, duration)

    # Look to see if watch on in last hour for calculating if watch is on for duration < 1hour
    if (type(duration) != str) and (duration < pd.Timedelta("1h")):
        watch_on_subset = getDurationAroundTimestamp(
            hk_data, user_id, timestamp, pd.Timedelta("1h")
        )
        features["watch_on_percent"] = processWatchOnPercent(
            watch_on_subset,
            resample=watch_on_resample,
            origin=start,
        )
    else:
        # TODO: Confirm that this calculates from start of duration to end of duration
        features["watch_on_percent"] = processWatchOnPercent(
            subset, resample=watch_on_resample, origin=start
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
        c
        for c in value_cols
        if not (
            (
                c.lower().startswith("heart")  # HR and HRV
                or c.lower().startswith("respiratory")  # RR
                or c.lower().startswith("oxygen")  # SpO2
                or c.lower().startswith("sleep")  # Sleep data
            )
            and not c.endswith("count")  # Counts should be 0 filled
        )
    ]
    qc_features[fill_value_cols] = qc_features[fill_value_cols].fillna(0)
    qc_features[duration_cols] = qc_features[duration_cols].fillna(
        pd.Timedelta(0)
    )
    qc_features.loc[
        qc_features["watch_on_percent"] < watch_on_threshold, watch_cols
    ] = np.nan
    return qc_features


def calcStartStop(
    timestamp: pd.Timestamp,
    duration: durationType,
) -> Tuple[pd.Timestamp, pd.Timestamp]:
    """From a timestamp and duration to look back, generate start and stop
    times for data filtering
    Durations of value "yesterday" or "today" are also possible, used for EMA
    responses referring to those time periods

    Args:
        timestamp (pd.Timestamp): Timestamp for end of data filter window
        duration (durationType): Duration of time to look back

    Raises:
        ValueError: Invalid entry sent for duration variable

    Returns:
        Tuple[pd.Timestamp, pd.Timestamp]: Start and stop timestamps for data
        filtering
    """
    if duration in ["today", "yesterday"]:
        if duration == "today":
            start = pd.to_datetime(timestamp.date())
        elif duration == "yesterday":
            start = pd.to_datetime((timestamp - pd.Timedelta("1d")).date())
        else:
            raise ValueError(f"Invalid duration: {duration}")

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

    return start, end


def getDurationAroundTimestamp(
    hk_data: pd.DataFrame,
    user_id: int,
    timestamp: pd.Timestamp,
    duration: durationType,
):
    """Returns a DataFrame with data from user_id overlapping with duration
    around a timestamp"""
    # Note returns data that has either a start OR end date within the duration
    # This means that data may be included that is not within the duration

    start, end = calcStartStop(timestamp, duration)

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
    hk_data: pd.DataFrame,
    resample: str = "1h",
    origin: pd.Timestamp
    | Literal["epoch", "start", "start_day", "end", "end_day"] = "start",
) -> float:
    """Get % of "resample" bins with heart rate data -> proxy for watch wear

    Args:
        hk_data (pd.DataFrame): Participant healthKit data filtered within a
        time window
        resample (str, optional): resample period. Defaults to "1h".
        origin (pd.Timestamp | Literal[&quot;epoch&quot;, &quot;start&quot;,
            &quot;start_day&quot;, &quot;end&quot;, &quot;end_day&quot;],
            optional): Where to start resampling Defaults to "start".

    Returns:
        float: % of time bins with heart rate data
    """
    hr = hk_data[hk_data.type == "HeartRate"]
    watch_bool = hr["device.name"].str.contains("Apple Watch")
    watch_bool = watch_bool.fillna(False)
    watch_hr = hr.loc[watch_bool, ["local_start", "value"]]
    watch_hr_logs = (
        watch_hr.set_index("local_start")
        .resample(resample, origin=origin)
        .count()
    )
    if watch_hr.empty:
        return 0
    return (
        100
        * watch_hr_logs[watch_hr_logs["value"] > 0].shape[0]
        / watch_hr_logs.shape[0]
    )


def dailySleepFeatures(hk_data: pd.DataFrame, qc:bool = True) -> pd.DataFrame:
    """Calculate standard features on primary sleep metrics from Apple annotations
    Times reported relative to midnight of the prior day, sleep metrics set
    for day after sleep period from 3pm day prior to 3pm day of sleep metric

    Aggregation of duration reported in HOURS
    Args:
        hk_data (pd.DataFrame): HealthKit data

    Returns:
        pd.DataFrame: Metrics on sleep per night
    """
    sleep_data = []
    for uid, data in hk_data.groupby("user_id"):
        sleep = (
            data.loc[
                data.type == "SleepAnalysis",
                ["local_start", "local_end", "value", 'user_id', 'type'],
            ]
            .sort_values(by="local_start")
            .drop_duplicates()
        )
        sleep = combineOverlapsSleep(sleep, value_col="value").drop_duplicates()
        if sleep.empty:
            continue
        hr = data[data["type"] == "HeartRate"].copy()
        hr['value'] = hr['value'].astype(float)
        hrv = data[data["type"] == "HeartRateVariabilitySDNN"].copy()
        hrv['value'] = hrv['value'].astype(float)
        noise = data[data["type"] == "EnvironmentalAudioExposure"].copy()
        noise['value'] = noise['value'].astype(float)
        sleep["value"] = (
            sleep["value"]
            .astype(str)
            .str.replace("HKCategoryValueSleepAnalysis", "")
        )

        sleep["duration"] = sleep["local_end"] - sleep["local_start"]

        # Start at 3pm
        noon = pd.to_datetime(sleep["local_start"].min()).replace(
            hour=15, minute=0, second=0, microsecond=0
        )
        sleep.drop_duplicates()
        in_bed = [
            "InBed",
            "Asleep",
            "AsleepUnspecified",
            "CategoryValueUnknown",
            "Awake",
            "AwakeUnspecified",
            "AsleepCore",
            "AsleepDeep",
            "AsleepREM",
        ]
        asleep = [
            "Asleep",
            "AsleepUnspecified",
            "AwakeUnspecified",
            "CategoryValueUnknown",  # from documentation this indicates asleep unknown stage
            "AsleepCore",
            "AsleepDeep",
            "AsleepREM",
        ]
        awake = ["Awake", "AwakeUnspecified"]
        bedrestOnset = (
            sleep[sleep.value.isin(in_bed)]
            .resample("1D", origin=noon, on="local_start")["local_start"]
            .first()
            .rename("bedrestOnset")
        )
        bedrestOffset = (
            sleep[sleep.value.isin(in_bed)]
            .resample("1D", origin=noon, on="local_start")["local_end"]
            .last()
            .rename("bedrestOffset")
        )
        bedrestDuration = (
            sleep[sleep.value.isin(in_bed)]
            .resample("1D", origin=noon, on="local_start")["duration"]
            .sum()
            .rename("bedrestDuration")
        )
        sleepOnset = (
            sleep[sleep.value.isin(asleep)]
            .resample("1D", origin=noon, on="local_start")["local_start"]
            .first()
            .rename("sleepOnset")
        )

        sleepOffset = (
            sleep[sleep.value.isin(asleep)]
            .resample("1D", origin=noon, on="local_start")["local_end"]
            .last()
            .rename("sleepOffset")
        )

        sleepDuration = (
            sleep[sleep.value.isin(asleep)]
            .resample("1D", origin=noon, on="local_start")["duration"]
            .sum()
            .rename("sleepDuration")
        )
        # remove first awake period if it is before sleep onset and after bedrest onset
        firstAwake = (
            sleep[sleep.value.isin(awake)]
            .resample("1D", origin=noon, on="local_start")[["duration", "local_start"]]
            .first()
        )
        firstAwake = pd.concat(
            [firstAwake, bedrestOnset, sleepOnset],
            axis=1
        )
        firstAwake = firstAwake[
            (firstAwake.local_start > firstAwake.bedrestOnset) & (firstAwake.local_start< firstAwake.sleepOnset)
        ]
        firstAwakeDuration = firstAwake["duration"].rename("firstAwakeDuration")
        awakeDuration = (
            sleep[sleep.value.isin(awake)]
            .resample("1D", origin=noon, on="local_start")["duration"]
            .sum()
            .rename("awakeDuration")
        )
        # Sum awake periods after sleep onset and before sleep offset

        sleep_agg = pd.concat(
            [
                bedrestOnset,
                bedrestOffset,
                bedrestDuration,
                sleepOnset,
                sleepOffset,
                sleepDuration,
                firstAwakeDuration,
                awakeDuration,
            ],
            axis=1,
        )
        sleep_agg['firstAwakeDuration'] = sleep_agg['firstAwakeDuration'].fillna(pd.Timedelta(0))

        # Clean up bedrest onset and offset
        sleep_agg.loc[sleep_agg['bedrestOnset'] > sleep_agg['sleepOnset'], 'bedrestOnset'] = sleep_agg.loc[sleep_agg['bedrestOnset'] > sleep_agg['sleepOnset'],'sleepOnset']
        sleep_agg.loc[sleep_agg['bedrestOffset'] < sleep_agg['sleepOffset'], 'bedrestOffset'] = sleep_agg.loc[sleep_agg['bedrestOffset'] < sleep_agg['sleepOffset'],'sleepOffset']

        sleep_hr, sleep_hrv, sleep_noise = [], [], []
        for i, row in sleep_agg.iterrows():
            sleep_hr.append(
                hr[
                    (hr.local_start >= row.sleepOnset)
                    & (hr.local_start <= row.sleepOffset)
                ]["value"].median()
            )
            sleep_hrv.append(
                hrv[
                    (hrv.local_start >= row.sleepOnset)
                    & (hrv.local_start <= row.sleepOffset)
                ]["value"].median()
            )
            sleep_noise.append(
                noise[
                    (noise.local_start >= row.sleepOnset)
                    & (noise.local_start <= row.sleepOffset)
                ]["value"].median()
            )
        # Starting at 3pm offsets the hour by 15 hours from prior midnight
        hours_offset = 15
        sleep_agg["sleepHR"] = sleep_hr
        sleep_agg["sleepHRV"] = sleep_hrv
        sleep_agg["sleepNoise"] = sleep_noise

        # Convert durations to hours
        sleep_agg["bedrestDuration"] = (
            sleep_agg["bedrestDuration"]
        ).to_numpy() / pd.Timedelta("1 hour")

        sleep_agg["wakeAfterSleepOnset"] = (
            sleep_agg["awakeDuration"] - sleep_agg['firstAwakeDuration']
        ).to_numpy() / pd.Timedelta("1 hour")

        sleep_agg["sleepDuration"] = (
            sleep_agg["sleepDuration"]
        ).to_numpy() / pd.Timedelta("1 hour")
        sleep_agg["awakeDuration"] = (
            sleep_agg["awakeDuration"]
        ).to_numpy() / pd.Timedelta("1 hour")
        sleep_agg["sleepEfficiency"] = (
            sleep_agg["sleepDuration"] / sleep_agg["bedrestDuration"]
        )

        # Set 0 sleep efficiency to NaN
        sleep_agg.loc[sleep_agg.sleepEfficiency == 0, 'sleepEfficiency'] = np.nan

        # No sleep efficiency > 1
        sleep_agg.loc[sleep_agg["sleepEfficiency"] > 1, "sleepEfficiency"] = 1

        sleep_agg["sleepOnsetLatency"] = (
            sleep_agg["sleepOnset"] - sleep_agg["bedrestOnset"]
        ).to_numpy() / pd.Timedelta("1 hour")

        sleep_agg["bedrestOnsetHours"] = hours_offset + (
            sleep_agg["bedrestOnset"] - sleep_agg.index
        ).to_numpy() / pd.Timedelta("1 hour")
        sleep_agg["bedrestOffsetHours"] = hours_offset + (
            sleep_agg["bedrestOffset"] - sleep_agg.index
        ).to_numpy() / pd.Timedelta("1 hour")
        sleep_agg["sleepOnsetHours"] = hours_offset + (
            sleep_agg["sleepOnset"] - sleep_agg.index
        ).to_numpy() / pd.Timedelta("1 hour")
        sleep_agg["sleepOffsetHours"] = hours_offset + (
            sleep_agg["sleepOffset"] - sleep_agg.index
        ).to_numpy() / pd.Timedelta("1 hour")

        sleep_agg = sleep_agg.drop(
            columns=[
                "bedrestOnset",
                "bedrestOffset",
                "sleepOnset",
                "sleepOffset",
                "firstAwakeDuration",
            ]
        )

        sleep_agg = sleep_agg.reset_index()
        sleep_agg["date"] = (
            sleep_agg.local_start + pd.Timedelta("1 day")
        ).dt.date
        sleep_agg = sleep_agg.drop(columns=["local_start"])
        sleep_agg["user_id"] = uid
        sleep_data.append(sleep_agg)

    if len(sleep_data) == 0:
        return pd.DataFrame(columns=["user_id", "date"])
    sleep_df = pd.concat(sleep_data)
    rename = {}

    for col in sleep_df.columns:
        rename[col] = col
        if col not in ["user_id", "date"]:
            rename[col] = f"sleep_{col}_day"
    sleep_df = sleep_df.rename(columns=rename)

    if qc:
        sleep_df = qcSleepFeatures(sleep_df)
    return sleep_df


def qcSleepFeatures(data: pd.DataFrame) -> pd.DataFrame:
    if "sleep_sleepEfficiency_day" in data.columns:
        data.loc[
            data["sleep_sleepEfficiency_day"] == 0, "sleep_sleepEfficiency_day"
        ] = np.nan
    if "sleep_sleepDuration_day" in data.columns:
        data.loc[
            data["sleep_sleepDuration_day"] == 0, "sleep_sleepDuration_day"
        ] = np.nan
    if "sleep_Awake_sum" in data.columns:
        data.loc[data["sleep_Awake_sum"] > 20, "sleep_Awake_sum"] = np.nan
    return data


def qcActivity(data: pd.DataFrame) -> pd.DataFrame:
    types = data["type"].unique()
    if len(types) > 1:
        print('ERROR: Activity data has multiple types', len(types), types)
        return data
    if 'ActiveEnergyBurned' in types:
        data['activity_mins'] = (data['local_end'] - data['local_start']).dt.seconds / 60
        data.loc[data['activity_mins'] == 0, 'value'] = np.nan
        data.loc[data['activity_mins'] == 0, 'activity_mins'] = np.nan
        data['kcal_per_min'] = data['value'].astype(float) / data['activity_mins'].astype(float) / 1000
        data.loc[data['kcal_per_min'] < 0, 'value'] = np.nan
        data.loc[data['kcal_per_min'] > 30, 'value'] = np.nan
        data = data.drop(columns=['activity_mins', 'kcal_per_min'])
    return data

def aggregateAudioExposure(
        hk_data: pd.DataFrame,
        resample: str = "1h",
) -> pd.DataFrame:
    audio_data = hk_data[
        hk_data.type == "EnvironmentalAudioExposure"
    ].copy()
    audio_data['value'] = audio_data['value'].astype(float)
    overlap_combined = combineOverlaps(audio_data, 'value')
    overlap_combined['duration'] = overlap_combined['local_end'] - overlap_combined['local_start']
    resamp = overlap_combined.set_index('local_start')[['body.quantity.count', 'duration', 'value']].resample(resample).median()
    agg = pd.DataFrame(resamp.aggregate({
        'duration': 'sum',
        'value': 'mean',
        'body.quantity.count': 'sum',
    }))
    agg = agg.T.rename(columns={
        'duration': 'audioExposure_hours',
        'value': 'audioExposure_mean',
        'body.quantity.count': 'audioExposure_count',
    })
    agg['audioExposure_entries'] = resamp.value.count()
    agg['audioExposure_hours'] = agg['audioExposure_hours'] / pd.Timedelta('1h')
    return agg


def aggregateDailySleep(
    hk_data: pd.DataFrame,
    sleep_daily: pd.DataFrame | None = None,
    aggs: List = ["median", "min", "max", "std"],
    sleep_features: None | List = None
) -> pd.DataFrame:
    if sleep_daily is None:
        sleep_daily = dailySleepFeatures(hk_data).drop(columns=["user_id", "date"])
        if sleep_features is None:
            sleep_features = [c for c in sleep_daily.columns]
        if sleep_daily.empty:
            return pd.DataFrame()
    sleep_agg = (
        sleep_daily[sleep_features]
        .aggregate(aggs)
        .unstack()
    )
    sleep_agg[("sleep_sleep_day", "count")] = sleep_daily[
        "sleep_sleepDuration_day"
    ].count()
    sleep_agg[("sleep_bedrest_day", "count")] = sleep_daily[
        "sleep_bedrestDuration_day"
    ].count()
    sleep_agg = pd.DataFrame(sleep_agg).T
    sleep_agg.columns = [f"{x}_{y}" for (x, y) in sleep_agg.columns]
    return sleep_agg


def aggregateSleepCategories(hk_data: pd.DataFrame, qc: bool = True) -> pd.DataFrame:
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
            ["local_start", "local_end", "value", 'user_id', 'type'],
        ]
        .rename(columns={"body.category.value": "SleepAnalysis", "value": "SleepAnalysis"})
        .sort_values(by="local_start")
    )
    sleep = combineOverlapsSleep(sleep, "SleepAnalysis")
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
    s2[duration_cols] = s2[duration_cols].map(
        lambda x: pd.Timedelta(x) / pd.Timedelta("1h")
    )
    if qc:
        s2 = qcSleepFeatures(s2)
    return s2


def countEventLog(
    hk_data: pd.DataFrame, event_types: List[str]
) -> pd.DataFrame:
    types = hk_data["type"].unique()
    # Check that event_type is in types
    for event_type in event_types:
        if event_type not in types:
            # print(f"ERROR: {event_type} not in {types}")
            return pd.DataFrame()

    events = hk_data.loc[
        hk_data.type.isin(event_types), ["local_start", "type"]
    ]
    return pd.DataFrame(events.type.value_counts()).T.reset_index(drop=True)


def aggregateActiveDuration(
    hk_data: pd.DataFrame,
    hk_type: str,
    qc: bool = True,
    device: str = "Apple Watch",
    resample: None | str = "1h",
) -> pd.DataFrame:
    if hk_type not in ["StepCount", "AppleExerciseTime", "ActiveEnergyBurned"]:
        raise ValueError(
            f"Invalid hk_type: {hk_type}, must be ActiveEnergyBurned, StepCount or AppleExerciseTime"
        )

    activity = hk_data.loc[
        (hk_data.type == hk_type)
        & (hk_data["device.name"] == device)
        & (hk_data["body.quantity.count"] == 1)
        ,
        ["local_start", "local_end", "value", "type", "user_id"],
    ].sort_values(by="local_start")

    activity['value'] = activity['value'].astype(float)
    if qc:
        activity = qcActivity(activity)

    # Combine overlapping values
    activity = combineOverlaps(activity, "value").rename(
        columns={"value": hk_type}
    )

    activity["duration"] = activity["local_end"] - activity["local_start"]

    if resample is not None:
        activity = activity.set_index("local_start")[[hk_type, 'duration']].resample(resample).median(numeric_only=False).fillna(0)
    # Filter out 0 duration values
    activity = activity[pd.to_timedelta(activity["duration"]) > pd.Timedelta(0)]

    activity_agg = pd.DataFrame(
        activity[hk_type].aggregate(["sum", "count"])
    ).T
    activity_agg.columns = [f"{hk_type}_{col}" for col in activity_agg.columns]
    activity_agg[f"{hk_type}_duration"] = pd.to_timedelta(
        activity["duration"].sum()
    ) / pd.Timedelta("1h")

    # Drop sum on Apple Exercise Time as it is the same as duration
    if hk_type == "AppleExerciseTime" and f"{hk_type}_mean" in activity_agg.columns:
        activity_agg.drop(
            columns=[f"{hk_type}_mean"],
            inplace=True,
        )
    return activity_agg.reset_index(drop=True)


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
    circadian_model_aggregations: bool = False,
    vital_range: Tuple[float, float] | None = None,
) -> pd.DataFrame:
    if vital_type not in [
        "HeartRate",
        "HeartRateVariabilitySDNN",
        "RespiratoryRate",
        "OxygenSaturation",
    ]:
        raise NotImplementedError(f"Vital type {vital_type} not implemented")
    vital = (
        hk_data.loc[
            hk_data.type == vital_type,
            ["local_start", "value"],
        ]
        .rename(columns={"value": vital_type})
        .drop_duplicates()
    )
    if vital.empty:
        return pd.DataFrame()
    vital[vital_type] = vital[vital_type].astype(float)
    if vital_range is not None:
        vital = vital[
            vital[vital_type].between(vital_range[0], vital_range[1])
        ]
    vital_resamp = vital.set_index("local_start").resample(resample).median()
    vital_agg = pd.DataFrame(vital_resamp.aggregate(standard_aggregations)).T
    vital_agg.columns = [f"{vital_type}_{col}" for col in vital_agg.columns]

    # Add time domain features
    if linear_time_aggregations:
        resamp_nona = vital_resamp.dropna()
        if resamp_nona.shape[0] < 3:
            return vital_agg.reset_index(drop=True)
        time_hours = (resamp_nona.index - resamp_nona.index[0]) / pd.Timedelta(
            "1h"
        )
        regression = pg.linear_regression(time_hours, resamp_nona[vital_type])
        vital_agg[f"{vital_type}_intercept"] = regression["coef"].values[0]
        vital_agg[f"{vital_type}_slope"] = regression["coef"].values[1]
    if circadian_model_aggregations:
        resamp_nona = vital_resamp.dropna()
        if resamp_nona.shape[0] < 3:
            return vital_agg.reset_index(drop=True)
        time_hours = (resamp_nona.index - resamp_nona.index[0]) / pd.Timedelta(
            "1h"
        )
        bounds = (0, [200, 200, 24, 48])
        p0 = [50, 50, 12, 24]
        model = CircadianModel(bounds=bounds, init_params=p0)
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
