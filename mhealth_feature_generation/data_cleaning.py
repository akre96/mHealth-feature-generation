import pandas as pd


def combineOverlaps(
    user_hk_data: pd.DataFrame, value_col: str
) -> pd.DataFrame:
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
        .reset_index()
    )
    activity["prev_local_end"] = activity["local_end"].shift()
    activity["duration"] = (
        activity["local_end"] - activity["local_start"]
    ) / pd.Timedelta("1m")
    activity["prev_local_start"] = activity["local_start"].shift()
    activity["overlap"] = (
        activity["local_start"] < activity["prev_local_end"]
    ) & (activity["local_end"] > activity["prev_local_start"])
    has_overlap = activity[activity.overlap].index

    # Combines values using time weighting
    for overlap_ind in has_overlap:
        overlap_rows = activity.loc[[overlap_ind - 1, overlap_ind], :]
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
        activity.drop(overlap_ind - 1, inplace=True)

    activity.drop(
        columns=["prev_local_end", "prev_local_start", "overlap"],
        inplace=True,
    )
    return activity


def combineOverlapsSleep(
    user_hk_data: pd.DataFrame, value_col: str
) -> pd.DataFrame:
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
        .reset_index()
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
    combine_overlap = activity[activity.combine_overlap].index

    # Combines duration if same type
    for overlap_ind in combine_overlap:
        overlap_rows = activity.loc[[overlap_ind - 1, overlap_ind], :]
        activity.loc[overlap_ind, "local_start"] = overlap_rows[
            "local_start"
        ].min()
        activity.drop(overlap_ind - 1, inplace=True)

    keep_last = activity[~(activity.combine_overlap) & activity.overlap].index

    # Keep last value if different types
    for keep_ind in keep_last:
        # Leave InBed as it is from the phone
        if not activity[keep_ind - 1][value_col] == "InBed":
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
