""" Wrapper functions for simple feature generation from Apple HealthKit data at the daily level
Features are calculated on a daily basis with user_id and date as the primary keys
Features are in format "domain_metric_time" for example "watchOnHours_sum_day"
Time can be "day" or in quarter days with values "h0", "h6", "h12", "h18". Time will also refer to contexts such as "sleep", "resting" etc.

"""
import pandas as pd
import numpy as np
from typing import List, Tuple
from .data_cleaning import combineOverlaps
from .simple_features import (
    aggregateActiveDuration,
    aggregateSleepCategories,
    aggregateVital,
)


def getWatchOnHoursDaily(data: pd.DataFrame) -> pd.DataFrame:
    """Check how many hours a day the watch was worn
    Assumes if there is a heart rate log, the watch was worn
    Returned dataframe has columns [user_id, date, watchOnHours_sum_day]

    Args:
        data (pd.DataFrame): HealthKit data

    Returns:
        pd.DataFrame: Count of hours per day with watch wear
    """
    resample = "1h"
    hr = data[data.type == "HeartRate"]
    if hr.empty:
        return pd.DataFrame(columns=["user_id", "date"])
    watch_hr = hr[["user_id", "local_start", "value"]]
    watch_hr_logs = (
        watch_hr.set_index("local_start")
        .groupby("user_id")
        .resample(resample, origin="start_day")["value"]
        .count()
        > 0
    ).reset_index()
    watch_hr_logs = (
        watch_hr_logs.groupby("user_id")
        .resample("1D", on="local_start", origin="start_day")["value"]
        .sum()
    )
    watch_hr_logs = watch_hr_logs.reset_index().rename(
        columns={"local_start": "date", "value": "watchOnHours_sum_day"}
    )
    watch_hr_logs["date"] = watch_hr_logs["date"].dt.date
    return watch_hr_logs


def qcWatchDataDaily(data: pd.DataFrame, threshold: int = 18) -> pd.DataFrame:
    """Set metrics from watch to NaN if watch was worn for less than threshold hours"""
    if "watchOnHours_sum_day" not in data.columns:
        data["watchOnHours_sum_day"] = getWatchOnHoursDaily(data)[
            "watchOnHours_sum_day"
        ]
    watch_domains = [
        "HeartRate",
        "RespiratoryRate",
        "Oxygen",
        "Sleep",
    ]
    watch_features = [
        c for c in data.columns if c.split("_")[0] in watch_domains
    ]
    print("watch_features", watch_features)

    # Set days with less than 18 hours of watch wear to NaN
    data.loc[data.watchOnHours_sum_day < threshold, watch_features] = np.nan
    return data


def aggregateVitalsDaily(
    data,
    vital_type,
    quarter_day=True,
    standard_aggregations: List = ["mean", "median", "std", "min", "max"],
    resample="1h",
    linear_time_aggregations: bool = False,
    circadian_model_aggregations: bool = False,
    vital_range: Tuple[float, float] | None = None,
) -> pd.DataFrame:
    vital = data[data["type"] == vital_type].copy()
    map_vital = {
        "HeartRate": "hr",
        "HeartRateVariabilitySDNN": "hrv",
        "RespiratoryRate": "rr",
        "OxygenSaturation": "spo2",
    }
    vital_type_short = map_vital[vital_type]
    vital["value"] = vital["value"].astype(float)
    if vital.empty:
        return pd.DataFrame(columns=["user_id", "date"])

    # vital = vital.set_index('time')
    vital_daily = vital.groupby(
        [
            "user_id",
            pd.Grouper(key="local_start", freq="1D", origin="start_day"),
        ]
    ).apply(
        lambda x: aggregateVital(
            hk_data=x,
            vital_type=vital_type,
            resample=resample,
            standard_aggregations=standard_aggregations,
            linear_time_aggregations=linear_time_aggregations,
            circadian_model_aggregations=circadian_model_aggregations,
            vital_range=vital_range,
        )
    )
    vital_daily.columns = [c + "_day" for c in vital_daily.columns]
    vital_daily = vital_daily.reset_index()
    vital_daily = vital_daily.rename(
        columns={
            f"{vital_type_short}_sum_day": f"{vital_type_short}_count_day"
        }
    )
    vital_daily["date"] = vital_daily["local_start"].dt.date
    vital_daily.drop(columns=["level_2", "local_start"], inplace=True)
    return vital_daily
    """
    # Calculate metrics for every 6 hours - TODO: implement as using simple_features functions
    if quarter_day:
        vital_q = (
            resamp_vital.groupby("user_id")
            .resample("6h", on="local_start", origin="start_day")
            .aggregate({"count": "sum", "median": aggregations})
        )
        vital_q.columns = [f"{vital_type}_" + c[1] for c in vital_q.columns]
        vital_q = vital_q.rename(columns={f"{vital_type}_sum": f"{vital_type}_count"})
        vital_q = vital_q.reset_index()
        vital_q["date"] = vital_q.local_start.dt.date
        vital_q["hour"] = vital_q.local_start.dt.hour
        hrq_piv = vital_q.pivot_table(index=["date", "user_id"], columns=["hour"])
        hrq_piv.columns = [f"{c[0]}_h{c[1]}" for c in hrq_piv.columns]
        hrq_piv = hrq_piv[[c for c in hrq_piv.columns if "local_start" not in c]]

        # Fill nan count values with 0
        count_cols = [c for c in hrq_piv.columns if "count" in c]
        hrq_piv[count_cols] = hrq_piv[count_cols].fillna(0)

        # Merge with Daily Feature
        vital_features = (
            vital_daily.reset_index()
            .merge(hrq_piv, on=["date", "user_id"])
            .drop(columns=["local_start", "index"])
        )
    else:
        vital_features = vital_daily.reset_index().drop(
            columns=["local_start", "index"]
        )
    return vital_features
    """


def aggregateSleepCategoriesDaily(hk_data: pd.DataFrame) -> pd.DataFrame:
    """Process sleep time interval data from Apple HealthKit

    Aggregation of duration reported in HOURS
    Args:
        hk_data (pd.DataFrame): HealthKit data

    Returns:
        pd.DataFrame: Statistics on sleep time intervals
    """
    sleep_data = []
    for uid, data in hk_data.groupby("user_id"):
        sleep = data.loc[
            data.type == "SleepAnalysis",
            ["local_start", "local_end", "value", "type", 'user_id'],
        ].sort_values(by="local_start")
        if sleep.empty:
            continue
        sleep["value"] = sleep["value"].str.replace(
            "HKCategoryValueSleepAnalysis", ""
        )
        start_time = pd.to_datetime(sleep["local_start"].min()).replace(
            hour=15, minute=0, second=0, microsecond=0
        )
        sleep['time'] = sleep['local_start']

        sleep_agg = (
            sleep
            .resample("1D", on='time', origin=start_time, group_keys=True)
            .apply(aggregateSleepCategories)
        ).reset_index().rename(columns={'time': 'local_start'})

        if sleep_agg.empty:
            continue

        sleep_agg = sleep_agg.reset_index()
        sleep_agg["date"] = (sleep_agg["local_start"] + pd.Timedelta('1D')).dt.date
        sleep_agg = sleep_agg.drop(columns=["local_start"])
        sleep_agg["user_id"] = uid
        sleep_data.append(sleep_agg)

    if len(sleep_data) == 0:
        return pd.DataFrame(columns=["user_id", "date"])
    return pd.concat(sleep_data).drop(columns=['level_1', 'index'])


def aggregateActiveDurationDaily(
    hk_data: pd.DataFrame, hk_type: str
) -> pd.DataFrame:
    active_data = []
    supported_types = ["StepCount", "ExerciseTime", "ActiveEnergyBurned"]
    if hk_type not in supported_types:
        raise ValueError(
            f"Invalid hk_type: {hk_type}, must be one of {supported_types}"
        )
    for uid, data in hk_data.groupby("user_id"):
        activity = data.loc[
            data.type == hk_type,
            ["local_start", "local_end", "value", "type", "user_id", "device.name", "body.quantity.count"],
        ].sort_values(by="local_start")
        if activity.empty:
            continue
        activity["time"] = activity["local_start"]
        activity_agg = (
            activity.resample(
                "1D", origin="start_day", on="time", group_keys=True
            )
            .apply(lambda x: aggregateActiveDuration(x, hk_type=hk_type))
            .reset_index()
            .rename(columns={"time": "local_start"})
            .drop(columns=["level_1"])
        )
        activity_agg["user_id"] = uid
        activity_agg["date"] = activity_agg.local_start.dt.date

        active_data.append(activity_agg.reset_index(drop=True))
    if len(active_data):
        all_activity_agg = pd.concat(active_data)
    else:
        return pd.DataFrame(columns=["date", "user_id"])
    return all_activity_agg
