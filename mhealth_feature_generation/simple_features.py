""" Simple feature generation from Apple HealthKit data
Features are calculated on a daily basis with user_id and date as the primary keys
Features are in format "domain_metric_time" for example "watchOnHours_sum_day"
Time can be "day" or in quarter days with values "h0", "h6", "h12", "h18". Time will also refer to contexts such as "sleep", "resting" etc.

"""
import pandas as pd
import numpy as np
from typing import List


def getWatchOnHours(data: pd.DataFrame) -> pd.DataFrame:
    """Check how many hours a day the watch was worn
    Assumes if there is a heart rate log, the watch was worn
    Returned dataframe has columns [user_id, date, watchOnHours_sum_day]

    # TODO: Calculate watch on per quarter day and other time contexts

    Args:
        data (pd.DataFrame): HealthKit data

    Returns:
        pd.DataFrame: Count of hours per day with watch wear
    """
    resample = "1h"
    hr = data[data.type == "HeartRate"]
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

def qcWatchData(data: pd.DataFrame, threshold: int = 18) -> pd.DataFrame:
    """ Set metrics from watch to NaN if watch was worn for less than threshold hours
    """
    if "watchOnHours_sum_day" not in data.columns:
        data["watchOnHours_sum_day"] = getWatchOnHours(data)["watchOnHours_sum_day"]
    watch_domains = [
        "hr",
        "rr",
        "hrv",
        "spo2",
        "sleep",
    ]
    watch_features = [
        c for c in data.columns if c.split('_')[0] in watch_domains
    ]
    print('watch_features', watch_features)

    # Set days with less than 18 hours of watch wear to NaN
    data.loc[
        data.watchOnHours_sum_day < threshold,
        watch_features
    ] = np.nan
    return data


def aggregateVitals(
    data,
    vital_type,
    quarter_day=True,
    aggregations: List = ["mean", "median", "std", "min", "max"],
) -> pd.DataFrame:
    if vital_type not in [
        "HeartRate",
        "HeartRateVariabilitySDNN",
        "RespiratoryRate",
        "OxygenSaturation",
    ]:
        raise NotImplementedError(f"Vital type {vital_type} not implemented")

    vital = data[data["type"] == vital_type].copy()
    map_vital = {
        "HeartRate": "hr",
        "HeartRateVariabilitySDNN": "hrv",
        "RespiratoryRate": "rr",
        "OxygenSaturation": "spo2",
    }
    vital_type = map_vital[vital_type]
    vital["value"] = vital["value"].astype(float)

    # Aggregation metrics calculated on hourly resampled data
    resamp_vital = (
        vital.reset_index()
        .groupby("user_id")
        .resample("1h", on="local_start", origin="start_day")["value"]
        .aggregate(["count", "median"])
        .reset_index()
    )

    # Calculate daily metrics
    vital_daily = (
        resamp_vital.groupby("user_id")
        .resample("1D", on="local_start", origin="start_day")
        .aggregate({"count": "sum", "median": aggregations})
    )
    vital_daily.columns = [
        f"{vital_type}_" + c[1] + "_day" for c in vital_daily.columns
    ]
    vital_daily = vital_daily.reset_index()
    vital_daily = vital_daily.rename(
        columns={f"{vital_type}_sum_day": f"{vital_type}_count_day"}
    )
    vital_daily["date"] = vital_daily["local_start"].dt.date

    # Calculate metrics for every 6 hours
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


def processSleep(hk_data: pd.DataFrame) -> pd.DataFrame:
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
            ["local_start", "local_end", "value"],
        ].sort_values(by="local_start")
        sleep["value"] = sleep["value"].str.replace("HKCategoryValueSleepAnalysis", "")

        sleep["duration"] = sleep["local_end"] - sleep["local_start"]
        sleep_agg = (
            sleep.set_index("local_end")
            .groupby("value")
            .resample("1D", origin="start_day")["duration"]
            .aggregate(["sum", "mean", "count"])
        )
        if sleep_agg.empty:
            continue
        duration_cols = [
            col
            for col in sleep_agg.columns
            if col.endswith("sum") or col.endswith("mean")
        ]
        sleep_agg[duration_cols] = sleep_agg[duration_cols].applymap(
            lambda x: pd.Timedelta(x) / pd.Timedelta("1h")
        )
        sleep_agg = sleep_agg.reset_index()
        sleep_agg["date"] = sleep_agg["local_end"].dt.date
        sleep_agg = sleep_agg.drop(columns=["local_end"])
        sleep_piv = sleep_agg.pivot_table(
            index="date", columns=["value"], values=duration_cols
        )
        sleep_piv.columns = [f"sleep{c[1]}_{c[0]}_day" for c in sleep_piv.columns]
        sleep_piv["user_id"] = uid
        sleep_data.append(sleep_piv)

    return pd.concat(sleep_data)


def processActiveDuration(hk_data: pd.DataFrame, hk_type: str) -> pd.DataFrame:
    active_data = []
    supported_types = ["StepCount", "ExerciseTime", "ActiveEnergyBurned"]
    if hk_type not in supported_types:
        raise ValueError(
            f"Invalid hk_type: {hk_type}, must be one of {supported_types}"
        )
    qc_ranges = {
        "ActiveEnergyBurned": {
            "ratePerSecond": [0, 5],
        },
    }
    for uid, data in hk_data.groupby("user_id"):
        activity = (
            data.loc[
                data.type == hk_type,
                ["local_start", "local_end", "value"],
            ]
            .rename(columns={"value": hk_type})
            .sort_values(by="local_start")
        )
        activity["duration"] = activity["local_end"] - activity["local_start"]
        try:
            activity["ratePerSecond"] = activity[hk_type] / activity["duration"].dt.total_seconds()
        except ZeroDivisionError:
            nonzero_act = activity.duration.dt.total_seconds() != 0
            activity["ratePerSecond"] = np.nan
            activity.loc[
                nonzero_act,
                "ratePerSecond"
            ] = activity.loc[nonzero_act, hk_type] / activity.loc[nonzero_act, "duration"].dt.total_seconds()
        activity["prev_local_end"] = activity["local_end"].shift()
        activity["prev_local_start"] = activity["local_start"].shift()
        activity["overlap"] = (activity["local_start"] < activity["prev_local_end"]) & (
            activity["local_end"] > activity["prev_local_start"]
        )

        for metric, range in qc_ranges[hk_type].items():
            activity = activity[
                activity[metric].between(range[0], range[1], inclusive="both")
            ]

        # Remove overlapping rows
        # TODO: Review how overlaps are being handled
        if activity["overlap"].any():
            activity = activity[~activity["overlap"]]
            activity.drop(
                columns=["prev_local_end", "prev_local_start", "overlap"],
                inplace=True,
            )

        activity_agg = pd.DataFrame(
            activity.set_index("local_start")[hk_type]
            .resample("1D", origin="start_day")
            .aggregate(["sum", "count"])
        )
        activity_agg.columns = [f"{hk_type}_{col}_day" for col in activity_agg.columns]
        activity_agg[f"{hk_type}_duration_day"] = pd.to_timedelta(
            activity["duration"].sum()
        ) / pd.Timedelta("1h")
        activity_agg["user_id"] = uid
        activity_agg["date"] = activity_agg.index.date

        active_data.append(activity_agg.reset_index(drop=True))
    return pd.concat(active_data)
