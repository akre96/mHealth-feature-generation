""" This module contains the functions to compute simple FFT features from a non uniform time series signal.
"""

import numpy as np
import pandas as pd
from scipy.signal import lombscargle
from scipy.optimize import curve_fit
import warnings
from typing import Optional


def calculateLombScargle(
    time: np.ndarray,
    signal: np.ndarray,
    freqs: np.ndarray,
    normalize: bool = True,
    center: bool = True,
) -> np.ndarray:
    """
    Calculate the Lomb-Scargle periodogram of a time series.

    Parameters:
    time (numpy.ndarray): Time points of the time series.
    signal (numpy.ndarray): Signal values at the given time points.
    freqs (numpy.ndarray): Frequencies at which to calculate the periodogram.
    normalize (bool, optional): Whether to normalize the periodogram. Defaults to True.
    center (bool, optional): Whether to center the signal. Defaults to True.

    Returns:
    numpy.ndarray: periodogram at the specified frequencies.
    """
    # Check if input arrays are not None
    if time is None or signal is None or freqs is None:
        raise ValueError("Input arrays cannot be None")

    # Check if input arrays are not empty
    if len(time) == 0 or len(signal) == 0 or len(freqs) == 0:
        raise ValueError("Input arrays cannot be empty")

    # Calculate the Lomb-Scargle periodogram
    angular_freqs = 2 * np.pi * freqs
    try:
        # Handle the case where all frequencies are zero
        if np.all(freqs == 0):
            warnings.warn(
                "All frequencies are zero. Returning zero pgram.", UserWarning
            )
            return np.zeros_like(freqs)

        pgram = lombscargle(
            time, signal, angular_freqs, normalize=normalize, precenter=center
        )
    except Exception as e:
        raise RuntimeError(
            "Error in calculating Lomb-Scargle periodogram"
        ) from e

    return pgram


def spectralEntropy(pgram):
    """
    Calculate the spectral entropy of the pgram spectrum.

    Parameters:
    pgram (numpy.ndarray): periodogram of a time series.

    Returns:
    float: Spectral entropy of the periodogram
    """
    # Check if the input power array is not None
    if pgram is None:
        raise ValueError("Input pgram array cannot be None")

    # Check if the input pgram array is not empty
    if len(pgram) == 0:
        raise ValueError("Input pgram array cannot be empty")
    # Normalize the PSD
    psd_normalized = pgram / np.sum(pgram)

    # Calculate spectral entropy
    spectral_entropy = -np.sum(
        psd_normalized * np.log(psd_normalized + np.finfo(float).eps)
    )  # Adding eps to avoid log(0)

    return spectral_entropy


def peakFrequency(pgram, freqs):
    """
    Calculate the peak frequency of the pgram spectrum.

    Parameters:
    pgram (numpy.ndarray): Periodogram of a time series.

    Returns:
    float: Peak frequency of the periodogram
    """
    # Check if the input pgram array is not None
    if pgram is None:
        raise ValueError("Input pgram array cannot be None")

    # Check if the input pgram array is not empty
    if len(pgram) == 0:
        raise ValueError("Input pgram array cannot be empty")

    # Find the peak frequency
    peak_frequency = freqs[np.argmax(pgram)]
    return peak_frequency


def getFrequencies(time, max_num_freqs=10000):
    """
    Calculates the frequencies for a given time array using logarithmic spacing.

    Parameters:
        time (array-like): An array-like object containing time values.

    Returns:
        numpy.ndarray: An array of frequencies calculated using logarithmic spacing.

    Notes:
        - The frequency range is determined by the median time interval and the time span.
        - The number of frequencies is calculated based on the time span and the median time interval.
        - If the calculated number of frequencies exceeds 10,000, a warning is issued and the number is set to 10,000.
        - Informed by VanderPlas 2018.
    """
    if len(time) <= 1:
        raise ValueError("Input time array cannot be less than 2 elements")
    time_range = np.max(time) - np.min(time)
    freq_lower_limit = 1 / time_range
    # Median time interval
    time_interval = np.median(np.diff(time))
    if time_interval < 0:
        print(time_interval)
        print(time)
        raise ValueError("Input time array must be sorted in increasing order")
    freq_upper_limit = 2 / time_interval
    T = np.max(time) - np.min(time)
    n_0 = 5
    n_eval = n_0 * T * np.ceil(freq_upper_limit)
    if n_eval > max_num_freqs:
        n_eval = max_num_freqs
    n_eval = int(n_eval)
    # Logarithmic spacing of frequencies
    freqs = np.logspace(
        np.log10(freq_lower_limit), np.log10(freq_upper_limit), n_eval
    )
    return freqs


def calculateAutoCorrelation(pgram, signal):
    psd = pgram / np.sum(pgram)
    autocorr = np.fft.irfft(psd, n=len(signal))
    return autocorr


def MaxAutocorrelationLag(autocorr):
    max_autocorr_lag = np.argmax(autocorr) + 1
    return max_autocorr_lag


def exponential_decay(x, a, b):
    return a * np.exp(-b * x)


def autocorrDecayRate(autocorr):
    try:
        # only use the first half of the autocorrelation
        autocorr = autocorr[: len(autocorr) // 2]
        popt, _ = curve_fit(
            exponential_decay, np.arange(len(autocorr)), autocorr
        )
    except RuntimeError:
        return np.nan
    return popt[1]


def getLombScargleFeatures(
    time: np.ndarray,
    signal: np.ndarray,
    freqs: Optional[np.ndarray] = None,
    normalize: bool = True,
    center: bool = True,
    max_num_freqs: int = 10000,
) -> pd.DataFrame:
    """
    Calculate the Lomb-Scargle periodogram, spectral entropy, and peak frequency of a time series.

    Args:
        time (numpy.ndarray): Time points of the time series.
        signal (numpy.ndarray): Signal values at the given time points.
        freqs (numpy.ndarray, optional): Frequencies at which to calculate the periodogram.
            If not provided, frequencies will be calculated using the `getFrequencies` function.
        normalize (bool, optional): If True, normalize the periodogram. Default is True.
        center (bool, optional): If True, center the data around its mean. Default is True.

    Returns:
        pandas.DataFrame: A dataframe with the following columns:
            - spectral_entropy: The spectral entropy of the time series (float).
            - peak_frequency: The frequency with the highest periodogram value (float).
            - max_autocorr_lag: The lag with the maximum autocorrelation value (int).
            - decay_rate: The decay rate of the autocorrelation function (float).
    """
    if len(time) <= 5:
        return pd.DataFrame()
    # Calculate frequencies if not provided
    if freqs is None:
        freqs = getFrequencies(time, max_num_freqs)
    # Calculate the Lomb-Scargle periodogram
    pgram = calculateLombScargle(time, signal, freqs, normalize, center)

    # Calculate the spectral entropy
    spectral_entropy = spectralEntropy(pgram)

    # Calculate the peak frequency
    peak_frequency = peakFrequency(pgram, freqs)

    # Calculate the autocorrelation array
    autocorr = calculateAutoCorrelation(pgram, signal)
    # check if null or inf in autocorr
    if np.any(np.isnan(autocorr)) or np.any(np.isinf(autocorr)):
        max_autocorr_lag = np.nan
        decay_rate = np.nan
    else:
        # Calculate the maximum autocorrelation lag
        max_autocorr_lag = MaxAutocorrelationLag(autocorr)
        # Calculate the decay rate
        decay_rate = autocorrDecayRate(autocorr)

    # Create single row dataframe with features
    features = pd.DataFrame(
        {
            "spectral_entropy": [spectral_entropy],
            "peak_period": [1 / peak_frequency],
            "max_autocorr_lag": [max_autocorr_lag],
            "decay_rate": [decay_rate],
        }
    )

    return features
