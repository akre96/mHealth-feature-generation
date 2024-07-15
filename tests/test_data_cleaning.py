import pandas as pd
from mhealth_feature_generation.data_cleaning import (
    combineOverlaps,
    combineOverlapsSleep,
)


def test_combineOverlapSleep():
    test_data = pd.read_excel(
        "tests/test_sleep_data.xlsx", sheet_name="1_sleep_period_1_day_overlap"
    )
    comb = combineOverlapsSleep(test_data, "value")
    print(comb)
    assert comb.shape[0] == 2
    assert comb[comb.value == "Asleep"].shape[0] == 1
    assert comb[comb.value == "InBed"].shape[0] == 1

    # Appropriately combined Asleep periods
    assert (
        comb[comb.value == "Asleep"]["local_start"].values[0]
        == test_data[test_data.value == "Asleep"]["local_start"].min()
    )
    assert (
        comb[comb.value == "Asleep"]["local_end"].values[0]
        == test_data[test_data.value == "Asleep"]["local_end"].max()
    )


def test_combineOverlaps():
    test_data = pd.read_excel(
        "tests/test_sleep_data.xlsx", sheet_name="2_activity_overlap"
    )

    combined = combineOverlaps(test_data, value_col="value")
    print("Original")
    print(test_data[["local_start", "local_end", "value"]])
    print("Combined")
    print(combined[["local_start", "local_end", "value"]])
    assert combined.shape[0] == 1

    duration = (combined["local_end"] - combined["local_start"]).values[
        0
    ] / pd.Timedelta("1h")
    print("Duration", duration)
    assert duration == 1.5

    val = combined["value"].values[0]
    print("Value", val)
    assert val == 150
