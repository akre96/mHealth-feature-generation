import pandas as pd
from mhealth_feature_generation.simple_features import (
    aggregateAudioExposure,
    combineOverlaps,
    aggregateVital
)


def test_audio_exposure():
    test_data = pd.read_excel(
        "tests/test_environment_data.xlsx", sheet_name="audio_exposure"
    )
    cleaned = combineOverlaps(test_data, "value")
    audio_feats = aggregateAudioExposure(cleaned)
    print(audio_feats)
    duration_mins = round(audio_feats["audioExposure_hours"].values[0] * 60)
    assert audio_feats.shape[0] == 1
    assert audio_feats["audioExposure_mean"].values[0] == 65
    assert audio_feats["audioExposure_count"].values[0] == 101
    assert audio_feats["audioExposure_entries"].values[0] == 2
    assert duration_mins == 31

def test_hr_context():
    test_data = pd.read_excel(
        "tests/test_vital_data.xlsx", sheet_name="1_test_hr_context"
    )
    print(test_data)
    nonsleep_rest_hr = aggregateVital(
        test_data,
        vital_type='HeartRate',
        resample='1h',
        standard_aggregations=['mean', 'count'],
        linear_time_aggregations=False,
        circadian_model_aggregations=False,
        context='non-sleep rest'
    )
    print(nonsleep_rest_hr)
    assert nonsleep_rest_hr['HeartRate_nonsleep-rest_count'].values[0] == 1
    assert nonsleep_rest_hr['HeartRate_nonsleep-rest_mean'].values[0] == 75

    # Test that active heart rate is = 100
    active_hr = aggregateVital(
        test_data,
        vital_type='HeartRate',
        resample='1h',
        standard_aggregations=['mean', 'count'],
        linear_time_aggregations=False,
        circadian_model_aggregations=False,
        context='active'
    )
    print(active_hr)
    assert active_hr['HeartRate_active_mean'].values[0] == 100
    assert active_hr['HeartRate_active_count'].values[0] == 1

    # test that sleep heart rate is 40
    sleep_hr = aggregateVital(
        test_data,
        vital_type='HeartRate',
        resample='1h',
        standard_aggregations=['mean', 'count'],
        linear_time_aggregations=False,
        circadian_model_aggregations=False,
        context='sleep'
    )
    print(sleep_hr)
    assert sleep_hr['HeartRate_sleep_mean'].values[0] == 40
    assert sleep_hr['HeartRate_sleep_count'].values[0] == 1