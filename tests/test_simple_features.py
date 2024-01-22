import pandas as pd
from mhealth_feature_generation.simple_features import (
    aggregateAudioExposure,
    combineOverlaps,
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
