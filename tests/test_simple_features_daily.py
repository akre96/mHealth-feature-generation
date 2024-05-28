import pandas as pd
from mhealth_feature_generation.simple_features import (
    dailySleepFeatures,
    processWatchOnPercent,
    processWatchOnTime,
)
from mhealth_feature_generation.simple_features_daily import (
    aggregateActiveDurationDaily,
    aggregateEnvironmentDaily,
    aggregateVitalsDaily,
    aggregateSleepCategoriesDaily,
)
from numpy.testing import assert_almost_equal


def test_dailySleepFeatures_basic():
    test_data = pd.read_excel(
        "tests/test_sleep_data.xlsx", sheet_name="1_sleep_period_2_days"
    )
    sleep_feats = dailySleepFeatures(test_data)
    print(sleep_feats)

    assert sleep_feats.shape[0] == 2
    assert (
        sleep_feats["sleep_bedrestDuration_day"] == 6
    ).all()  # Bedrest does not include unlabeled time
    assert (sleep_feats["sleep_sleepDuration_day"] == 4).all()
    assert (sleep_feats["sleep_sleepEfficiency_day"] == 4 / 6).all()
    assert (sleep_feats["sleep_bedrestOnsetHours_day"] == 22).all()
    assert (sleep_feats["sleep_bedrestOffsetHours_day"] == 30).all()
    assert (sleep_feats["sleep_sleepOnsetHours_day"] == 26).all()
    assert (sleep_feats["sleep_sleepOffsetHours_day"] == 30).all()


def test_dailySleepFeatures_multiple_sleep():
    test_data = pd.read_excel(
        "tests/test_sleep_data.xlsx", sheet_name="2_sleep_period_1_day"
    )
    sleep_feats = dailySleepFeatures(test_data)
    print(sleep_feats)
    print(sleep_feats[["sleep_wakeAfterSleepOnset_day"]])

    assert sleep_feats.shape[0] == 1
    assert (sleep_feats["sleep_bedrestDuration_day"] == 7).all()
    assert (sleep_feats["sleep_sleepDuration_day"] == 5).all()
    assert (sleep_feats["sleep_sleepEfficiency_day"] == (5 / 7)).all()
    assert (sleep_feats["sleep_bedrestOnsetHours_day"] == 22).all()
    assert (sleep_feats["sleep_bedrestOffsetHours_day"] == 32).all()
    assert (sleep_feats["sleep_sleepOnsetHours_day"] == 26).all()
    assert (sleep_feats["sleep_sleepOffsetHours_day"] == 32).all()
    assert (sleep_feats["sleep_wakeAfterSleepOnset_day"] == 1).all()


def test_dailySleepCategories():
    test_data = pd.read_excel(
        "tests/test_sleep_data.xlsx", sheet_name="1_sleep_period_2_days"
    )
    sleep_feats = aggregateSleepCategoriesDaily(test_data)
    print(sleep_feats)
    assert sleep_feats.shape[0] == 2
    assert (
        sleep_feats["date"].min() == pd.to_datetime("January 2, 2023").date()
    )
    assert (sleep_feats["sleep_Asleep_count"] == 1).all()
    assert (sleep_feats["sleep_Asleep_mean"] == 4).all()
    assert (sleep_feats["sleep_InBed_mean"] == 1).all()


def test_aggregateActiveDaily():
    test_data = pd.read_excel(
        "tests/test_sleep_data.xlsx", sheet_name="2_activity_overlap"
    )
    test_data["device.name"] = "Apple Watch"
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
    assert agg.ActiveEnergyBurned_sum.values[0] == (150/4184)
    assert agg.ActiveEnergyBurned_count.values[0] == 1
    assert agg.ActiveEnergyBurned_duration.values[0] == 1.5


def test_aggregateVitalsDaily():
    test_data = pd.read_excel(
        "tests/test_sleep_data.xlsx", sheet_name="6_heart_rate"
    )
    agg = aggregateVitalsDaily(
        test_data,
        vital_type="HeartRate",
        standard_aggregations=["mean", "min", "max"],
        linear_time_aggregations=True,
        circadian_model_aggregations=True,
        vital_range=(0, 300),
    )
    print(agg)
    assert agg.shape[0] == 2


def test_processWatchOnPercent():
    # Create a mock subset of heart rate logs
    hr_subset = pd.DataFrame(
        {
            "local_start": [
                "2022-01-01 01:00:01",
                "2022-01-01 02:00:01",
                "2022-01-01 03:00:01",
                "2022-01-01 04:00:01",
                "2022-01-01 05:00:01",
            ],
            "value": [70, 80, 90, 100, 110],
            "type": [
                "HeartRate",
                "HeartRate",
                "HeartRate",
                "HeartRate",
                "HeartRate",
            ],
            "device.name": [
                "Apple Watch",
                "Apple Watch",
                "Apple Watch",
                "Apple Watch",
                "Apple Watch",
            ],
        }
    )

    hr_subset["local_start"] = pd.to_datetime(hr_subset["local_start"])
    # Calculate the watch on percentage
    watch_on_percent = processWatchOnPercent(
        hr_subset,
        resample="1h",
        origin="2022-01-01 00:00:00",
        end="2022-01-01 06:00:00",
    )

    # Confirm that the watch on percentage is calculated from the beginning of the duration until the end
    print(watch_on_percent)
    print(processWatchOnTime(hr_subset, origin="2022-01-01 00:00:00"))
    assert_almost_equal(watch_on_percent, 100 * (5 / 7))


def test_audio_exposure_daily():
    test_data = pd.read_excel(
        "tests/test_environment_data.xlsx", sheet_name="audio_exposure"
    )
    audio_feats = aggregateEnvironmentDaily(
        test_data, "EnvironmentalAudioExposure"
    )
    print(audio_feats)
    duration_mins = round(audio_feats["audioExposure_hours"].values[0] * 60)
    assert audio_feats.shape[0] == 1
    assert audio_feats["audioExposure_mean"].values[0] == 65
    assert audio_feats["audioExposure_count"].values[0] == 101
    assert audio_feats["audioExposure_entries"].values[0] == 2
    assert duration_mins == 31
