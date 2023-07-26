"""_summary_ Link generated daily mHealth features to self-report data
sensor data must have columns for user_id and date, self-reports must have survey, question, response, and duration columns
Duration used to specify time period prior to self-report to use for feature aggregation
"""

import pandas as pd
from pathlib import Path
from typing import Union, Optional

def aggregateDailyFeaturesToSelfReportWindow(
        daily_sensor_data: pd.DataFrame,
        feature_cols: list[str],
        self_report_data: pd.DataFrame,
) -> pd.DataFrame:
    """ Link generated daily mHealth features to self-report data

    Args:
        daily_sensor_data (pd.DataFrame): Output from `simple_features.py` or other daily features with user_id and date columns
        feature_cols (list[str]): columns in daily_sensor_data with mHealth features
        self_report_data (pd.DataFrame): Self-reported answers, must have survey, question, response, and duration columns

    Returns:
        pd.DataFrame: self_report_data augmented with aggregated mHeath features
    """
    self_report_data = self_report_data.copy()
    if ["user_id", "date"] not in daily_sensor_data.columns:
        raise ValueError("daily_sensor_data must have user_id and date columns")
    
    


