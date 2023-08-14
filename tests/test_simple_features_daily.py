import pandas as pd
from mhealth_feature_generation.simple_features_daily import dailySleepFeatures


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
    assert (sleep_feats["sleep_bedrestOnsetHours_day"] == 10).all()
    assert (sleep_feats["sleep_bedrestOffsetHours_day"] == 18).all()
    assert (sleep_feats["sleep_sleepOnsetHours_day"] == 14).all()
    assert (sleep_feats["sleep_sleepOffsetHours_day"] == 18).all()


def test_dailySleepFeatures_multiple_sleep():
    test_data = pd.read_excel(
        "tests/test_sleep_data.xlsx", sheet_name="2_sleep_period_1_day"
    )
    sleep_feats = dailySleepFeatures(test_data)
    print(sleep_feats)

    assert sleep_feats.shape[0] == 1
    assert (sleep_feats["sleep_bedrestDuration_day"] == 10).all()
    assert (sleep_feats["sleep_sleepDuration_day"] == 5).all()
    assert (sleep_feats["sleep_sleepEfficiency_day"] == 0.5).all()
    assert (sleep_feats["sleep_bedrestOnsetHours_day"] == 10).all()
    assert (sleep_feats["sleep_bedrestOffsetHours_day"] == 20).all()
    assert (sleep_feats["sleep_sleepOnsetHours_day"] == 14).all()
    assert (sleep_feats["sleep_sleepOffsetHours_day"] == 20).all()
