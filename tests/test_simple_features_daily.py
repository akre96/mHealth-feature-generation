import pandas as pd
from mhealth_feature_generation.simple_features import dailySleepFeatures
from mhealth_feature_generation.simple_features_daily import (
    aggregateActiveDurationDaily,
    aggregateVitalsDaily,
    aggregateSleepCategoriesDaily,
)


def test_dailySleepFeatures_basic():
    test_data = pd.read_excel(
        "tests/test_sleep_data.xlsx", sheet_name="1_sleep_period_2_days"
    )
    sleep_feats = dailySleepFeatures(test_data)
    print(sleep_feats)

    assert sleep_feats.shape[0] == 2
    assert (sleep_feats["sleep_bedrestDuration_day"] == 8).all()
    assert (sleep_feats["sleep_sleepDuration_day"] == 4).all()
    assert (sleep_feats["sleep_sleepEfficiency_day"] == 0.5).all()
    assert (sleep_feats["sleep_bedrestOnsetHours_day"] == 22).all()
    assert (sleep_feats["sleep_bedrestOffsetHours_day"] == 30).all()
    assert (sleep_feats["sleep_sleepOnsetHours_day"] == 26).all()
    assert (sleep_feats["sleep_sleepOffsetHours_day"] == 30).all()


def test_dailySleepFeatures_multiple_sleep():
    test_data = pd.read_excel(
        "tests/test_sleep_data.xlsx", sheet_name="2_sleep_period_1_day"
    )
    sleep_feats = dailySleepFeatures(test_data)
    print(sleep_feats[["sleep_bedrestOnsetHours_day"]])

    assert sleep_feats.shape[0] == 1
    assert (sleep_feats["sleep_bedrestDuration_day"] == 10).all()
    assert (sleep_feats["sleep_sleepDuration_day"] == 5).all()
    assert (sleep_feats["sleep_sleepEfficiency_day"] == 0.5).all()
    assert (sleep_feats["sleep_bedrestOnsetHours_day"] == 22).all()
    assert (sleep_feats["sleep_bedrestOffsetHours_day"] == 32).all()
    assert (sleep_feats["sleep_sleepOnsetHours_day"] == 26).all()
    assert (sleep_feats["sleep_sleepOffsetHours_day"] == 32).all()


def test_dailySleepCategories():
    test_data = pd.read_excel(
        "tests/test_sleep_data.xlsx", sheet_name="1_sleep_period_2_days"
    )
    sleep_feats = aggregateSleepCategoriesDaily(test_data)
    print(sleep_feats)
    assert sleep_feats.shape[0] == 2
    assert sleep_feats['date'].min() == pd.to_datetime('January 2, 2023').date()
    assert (sleep_feats['sleep_Asleep_count'] == 1).all()
    assert (sleep_feats['sleep_Asleep_mean'] == 4).all()
    assert (sleep_feats['sleep_InBed_mean'] == 1).all()


def test_aggregateActiveDaily():
    test_data = pd.read_excel(
        "tests/test_sleep_data.xlsx", sheet_name="2_activity_overlap"
    )
    agg = aggregateActiveDurationDaily(test_data, hk_type="ActiveEnergyBurned")
    print(
        agg[
            [
                "ActiveEnergyBurned_sum",
                "ActiveEnergyBurned_count",
                "ActiveEnergyBurned_duration",
            ]
        ]
    )
    assert agg.shape[0] == 1
    assert agg.ActiveEnergyBurned_sum.values[0] == 150
    assert agg.ActiveEnergyBurned_count.values[0] == 1
    assert agg.ActiveEnergyBurned_duration.values[0] == 1.5


def test_aggregateVitalsDaily():
    test_data = pd.read_excel(
        "tests/test_sleep_data.xlsx", sheet_name="6_heart_rate"
    )
    agg = aggregateVitalsDaily(
        test_data,
        vital_type="HeartRate",
        quarter_day=False,
        standard_aggregations=["mean", "min", "max"],
        linear_time_aggregations=True,
        circadian_model_aggregations=True,
        vital_range=(0, 300),
    )
    print(agg)
    assert agg.shape[0] == 2
