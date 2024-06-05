"""
This file contains functions for data cleaning and manipulation related
to user health data.

Functions:
- combineOverlaps: Combines overlapping activities in the user_hk_data
    DataFrame based on the specified value column.
- combineOverlapsSleep: Combines and splits overlapping sleep durations
    in the given user health data.
"""
import pandas as pd
import numpy as np


def combineOverlaps(
    user_hk_data: pd.DataFrame, value_col: str
) -> pd.DataFrame:
    """
    Combines overlapping activities in the user_hk_data DataFrame based on the specified value column.

    Args:
        user_hk_data (pd.DataFrame): The DataFrame containing the user's activity data.
        value_col (str): The name of the column representing the values to be combined.

    Returns:
        pd.DataFrame: The DataFrame with overlapping activities combined and NaN values dropped.

    """
    activity = (
        user_hk_data.copy()
        .drop_duplicates(
            subset=[
                "local_start",
                "user_id",
                "local_end",
                value_col,
                "type",
            ],
            keep="last",
        )
        .sort_values(by="local_start")
    ).reset_index(drop=True)
    activity["prev_local_end"] = activity["local_end"].shift()
    activity["duration"] = (
        activity["local_end"] - activity["local_start"]
    ) / pd.Timedelta("1m")
    activity["prev_local_start"] = activity["local_start"].shift()
    activity["overlap"] = (
        activity["local_start"] < activity["prev_local_end"]
    ) & (activity["local_end"] > activity["prev_local_start"])
    # Drop if local_end before prev_local_end
    activity.loc[
        activity.overlap & (activity.local_end < activity.prev_local_end),
        value_col,
    ] = np.nan
    has_overlap = activity[activity.overlap].index

    # Combines values using time weighting
    for overlap_ind in has_overlap:
        overlap_rows = activity.loc[[overlap_ind - 1, overlap_ind], :]
        if overlap_rows[value_col].isna().any():
            continue
        total_time = (
            overlap_rows["local_end"].max() - overlap_rows["local_start"].min()
        ) / pd.Timedelta("1m")
        weighted_value = total_time * (
            overlap_rows[value_col].sum() / overlap_rows["duration"].sum()
        )
        activity.loc[overlap_ind, value_col] = weighted_value
        activity.loc[overlap_ind, "local_start"] = overlap_rows[
            "local_start"
        ].min()
        activity.loc[overlap_ind, "local_end"] = overlap_rows[
            "local_end"
        ].max()
        activity.loc[overlap_ind, "duration"] = total_time
        activity.loc[overlap_ind - 1, value_col] = np.nan

    activity.drop(
        columns=["prev_local_end", "prev_local_start", "overlap"],
        inplace=True,
    )
    return activity.dropna(subset=[value_col])


def combineOverlapsSleep(
    user_hk_data: pd.DataFrame, value_col: str
) -> pd.DataFrame:
    """
    Combines and splits overlapping sleep durations in the given user health data.

    Args:
        user_hk_data (pd.DataFrame): The user health data containing sleep durations.
        value_col (str): The name of the column in the user health data that contains sleep values.

    Returns:
        pd.DataFrame: The modified user health data with combined and split sleep durations.

    """
    # Function code goes here
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
    activity = (
        user_hk_data.copy()
        .drop_duplicates(
            subset=[
                "local_start",
                "user_id",
                "local_end",
                value_col,
                "type",
            ],
            keep="last",
        )
        .sort_values(by="local_start")
        .reset_index(drop=True)
    )
    activity["prev_stage"] = activity[value_col].shift()
    activity["prev_local_end"] = activity["local_end"].shift()
    activity["duration"] = (
        activity["local_end"] - activity["local_start"]
    ) / pd.Timedelta("1m")
    activity["prev_local_start"] = activity["local_start"].shift()
    activity["overlap"] = (
        activity["local_start"] < activity["prev_local_end"]
    ) & (activity["local_end"] > activity["prev_local_start"])

    activity["combine_overlap"] = (
        activity.prev_stage == activity[value_col]
    ) & activity.overlap
    activity["split_overlap"] = (
        activity.prev_stage != activity[value_col]
    ) & activity.overlap
    combine_overlap = activity[activity.combine_overlap].index

    # Combines duration if same type
    for overlap_ind in combine_overlap:
        overlap_rows = activity.loc[[overlap_ind - 1, overlap_ind], :]
        activity.loc[overlap_ind, "local_start"] = overlap_rows[
            "local_start"
        ].min()
        activity.drop(overlap_ind - 1, inplace=True)

    # Split duration if different types
    split_overlap = activity[activity.split_overlap].index
    for overlap_ind in split_overlap:
        overlap_rows = activity.loc[[overlap_ind - 1, overlap_ind], :]
        asleep_cat = overlap_rows[value_col].isin(asleep)
        # If both asleep or both not sleep, merge with last
        if asleep_cat.all() or not asleep_cat.any():
            activity.loc[overlap_ind, "local_end"] = overlap_rows[
                "local_end"
            ].max()
            activity.drop(overlap_ind - 1, inplace=True)
        # If one asleep and one not, split prioritize last
        else:
            activity.loc[overlap_ind - 1, "local_end"] = activity.loc[
                overlap_ind, "local_start"
            ]

    keep_last = activity[
        ~(activity.combine_overlap)
        & activity.overlap
        & ~(activity.split_overlap)
    ].index

    # Keep last value if different types
    for keep_ind in keep_last:
        # Leave InBed as it is from the phone
        if not activity.loc[keep_ind - 1, value_col] == "InBed":
            activity.drop(keep_ind - 1, inplace=True)

    activity.drop(
        columns=[
            "prev_local_end",
            "prev_local_start",
            "overlap",
            "combine_overlap",
            "duration",
        ],
        inplace=True,
    )
    return activity
