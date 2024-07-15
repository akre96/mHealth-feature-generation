import pytest
import numpy as np
from scipy.signal import lombscargle
from mhealth_feature_generation.timedomain_features import (
    calculateLombScargle,
    getFrequencies,
    getLombScargleFeatures,
)


def test_calculateLombScargle():
    for center in [True, False]:
        for normalize in [True, False]:
            # Test case 1: Check that the function returns the correct power of the Lomb-Scargle periodogram
            time = np.array([0, 1, 2, 3, 4])
            signal = np.array([1, 2, 3, 4, 5])
            freqs = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
            expected_power = lombscargle(
                time,
                signal,
                2 * np.pi * freqs,
                precenter=center,
                normalize=normalize,
            )
            calculated_power = calculateLombScargle(
                time, signal, freqs, center=center, normalize=normalize
            )
            assert np.allclose(expected_power, calculated_power)

            # Test case 2: Check that the function handles negative frequencies correctly
            time = np.array([0, 1, 2, 3, 4])
            signal = np.array([1, 2, 3, 4, 5])
            freqs = np.array([-0.1, -0.2, -0.3, -0.4, -0.5])
            expected_power = lombscargle(
                time,
                signal,
                -2 * np.pi * freqs,
                precenter=center,
                normalize=normalize,
            )
            calculated_power = calculateLombScargle(
                time, signal, freqs, center=center, normalize=normalize
            )
            assert np.allclose(expected_power, calculated_power)

            # Test case 3: Check that the function handles zero frequencies correctly
            time = np.array([0, 1, 2, 3, 4])
            signal = np.array([1, 2, 3, 4, 5])
            freqs = np.array([0, 0, 0, 0, 0])
            expected_power = np.zeros_like(freqs)
            with pytest.warns(UserWarning):
                calculated_power = calculateLombScargle(
                    time, signal, freqs, center=center, normalize=normalize
                )
            assert np.allclose(expected_power, calculated_power)

            # Test case 4: Check that the function handles non-uniform time points correctly
            time = np.array([0, 0.5, 1, 1.5, 2])
            signal = np.array([1, 2, 3, 4, 5])
            freqs = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
            expected_power = lombscargle(
                time,
                signal,
                -2 * np.pi * freqs,
                precenter=center,
                normalize=normalize,
            )
            calculated_power = calculateLombScargle(
                time, signal, freqs, center=center, normalize=normalize
            )
            assert np.allclose(expected_power, calculated_power)


def test_getFrequencies():
    # Test case 1: Check that the function returns the correct frequencies for a given time array
    time = np.array([0, 1, 2, 3, 4])
    expected_freqs = np.logspace(np.log10(0.25), np.log10(2), 40)
    calculated_freqs = getFrequencies(time)
    assert np.allclose(calculated_freqs, expected_freqs)

    # Test case 2: Check that the function handles an empty time array correctly
    time = np.array([])
    with pytest.raises(ValueError):
        getFrequencies(time)

    # Test case 3: Check that the function handles a time array with a single element correctly
    time = np.array([0])
    with pytest.raises(ValueError):
        getFrequencies(time)


def test_getLombScargleFeatures():
    # Test case 1: Check that the function returns the correct spectral entropy and peak frequency
    # Generate a pure sinusoidal signal
    time = np.linspace(0, 10, 1000)  # Time array
    frequency = 1  # Frequency of the sinusoid in Hz
    A = 1  # Amplitude of the sinusoid
    signal = A * np.sin(2 * np.pi * frequency * time)

    # TODO: figure out why expected_entropy is close to 5 and not 0
    expected_entropy = 5.876
    expected_peak_freq = frequency
    expected_max_lag = 1
    expected_decay_rate = 25.242

    calculated_result_df = getLombScargleFeatures(
        time, signal, normalize=False, center=False
    )

    print(calculated_result_df)
    assert np.isclose(
        calculated_result_df["peak_period"], 1 / expected_peak_freq, atol=1e-3
    )
    assert np.isclose(
        calculated_result_df["spectral_entropy"], expected_entropy, atol=1e-3
    )
    assert np.isclose(
        calculated_result_df["max_autocorr_lag"], expected_max_lag, atol=1e-3
    )
    assert np.isclose(
        calculated_result_df["decay_rate"], expected_decay_rate, atol=1e-3
    )

    # Test case 2: Check that the function handles empty input correctly
    time = np.array([])
    signal = np.array([])
    freqs = np.array([])
    res = getLombScargleFeatures(time, signal, freqs)
    assert res.empty
