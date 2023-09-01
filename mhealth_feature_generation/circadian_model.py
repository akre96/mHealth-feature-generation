""" Model Circadian Rhythm with different input data streams

"""
from typing import List, Tuple
import numpy as np
import pandas as pd
from scipy.optimize import least_squares


class CircadianModel:
    """Fit 4 parameter circadian rhythm model to data"""

    def __init__(
        self,
        bounds: Tuple = (0, [200, 200, 24, 48]),
        init_params: List = [50, 50, 12, 24],
    ):
        """Initialize model

        Args:
            bounds (Tuple, optional): bounds on parameters, set with heart rate
                in mind. Defaults to (0, [200, 200, 24, 48]).
            init_params (List, optional): initial parameter guesses, set with
                heart rate in mind. Defaults to [50, 50, 12, 24].
        """
        self.bounds = bounds
        self.p0 = init_params
        self.parameters = [np.nan] * 4

    @staticmethod
    def circ_cosine(
        t: np.ndarray,
        mesor: float,
        amplitude: float,
        acrophase: float,
        period: float,
    ) -> np.ndarray:
        """Simple cosine function for circadian rhythm modeling

        Args:
            t (np.ndarray): time points
            mesor (float): average value
            amplitude (float): amplitude of cosine
            acrophase (float): time of peak
            period (float): length of cycle

        Returns:
            float: cosine function output
        """
        y = mesor + amplitude * np.cos(
            ((2 * np.pi * (t - acrophase)) / period)
        )
        return y

    def calculate_residuals(self, parameters, x, y):
        return self.circ_cosine(x, *parameters) - y

    @staticmethod
    def print_cosine_params(params):
        print(
            " Mesor",
            params[0],
            "\n",
            "Amplitude",
            params[1],
            "\n",
            "Acrophase",
            params[2],
            "\n",
            "Period",
            params[3],
            "\n",
        )

    def fit(self, x, y):
        lsq_fit = least_squares(
            self.calculate_residuals, self.p0, args=(x, y), bounds=self.bounds
        )
        self.parameters = lsq_fit.x
        return lsq_fit

    def predict(self, x):
        return self.circ_cosine(x, *self.parameters)

    def circ_model_summary(self, data) -> List:
        """Use in pd.DataFrame.apply to return model fit paramters.
        group data by subject id and date

        Args:
            data ([pd.DataFrame]): Dataframe with only one subject and one day
                of data

        Returns:
            List: model parameters (4), average cost, number of samples, and
                hours with data
        """
        # Get time of day
        data["time"] = (
            data.pacific_time
            - pd.to_datetime(
                data.iloc[0].pacific_time.date(), utc=True
            ).tz_convert("US/Pacific")
        ) / np.timedelta64(1, "h")

        # Number of hours with heart rate logs
        hours = data.pacific_time.dt.hour.nunique()

        # Fit model
        res_lsq = self.fit(data["time"], data["original_value"])
        n = data["original_value"].dropna().shape[0]
        # Return paramters + stats
        return [*res_lsq.x, res_lsq.cost / n, n, hours]
