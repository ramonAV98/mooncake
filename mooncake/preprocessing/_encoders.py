import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd


class CyclicalDates(BaseEstimator, TransformerMixin):
    """Encodes datetime features by projecting them into a unit circle.

    Parameters
    ----------
    day : bool, default=True
        Whether to transform day of month

    month : bool, default=True
        Whether to transform month number

    dayofweek : bool, default=False
        Whether to transform day of week

    References
    ----------
    https://ianlondon.github.io/blog/encoding-cyclical-features-24hour-time/
    """

    def __init__(self, day=True, month=True, dayofweek=False):
        self.day = day
        self.month = month
        self.dayofweek = dayofweek
        self.dtype = float

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        """Adds cyclical columns to ``X``

        Parameters
        ----------
        X : datetime pd.Series
            Datetime pandas series with datetime accessor

        Returns
        -------
        X_out : ndarray of shape (n_samples, n_encoded_features)
            Transformed input
        """
        if not hasattr(X, 'dt'):
            raise ValueError('X must be a datetime pandas Series with '
                             'datetime attribute `dt`, i.e., X.dt')
        X = X.copy()
        datetime_accessor = X.dt
        allowed_datetimes = ['day', 'month', 'dayofweek']
        projections = []
        for datetime in allowed_datetimes:
            if getattr(self, datetime):
                x = getattr(datetime_accessor, datetime)
                proj = self._unit_circle_projection(x)
                projections.append(proj)
        projections = np.hstack(projections)
        return projections

    def get_feature_names(self):
        """Get output feature names for transformation

        Returns
        -------
        feature_names_out : list of str
            Transformed feature names.
        """
        feature_names_out = []
        allowed_datetimes = ['day', 'month', 'dayofweek']
        for datetime in allowed_datetimes:
            if getattr(self, datetime):
                names = [datetime + '_sine', datetime + '_cos']
                feature_names_out.extend(names)
        return np.array(feature_names_out)

    def _unit_circle_projection(self, x):
        """Projects ``x`` into two dimensions using sine and cosine transforms

        Returns
        -------
        np ndarray of shape (n, 2) where n equals len(x)
        """
        x_sine = self._sine_transform(x)
        x_cosine = self._cosine_transform(x)
        return np.array([x_sine, x_cosine]).reshape(-1, 2)

    def _sine_transform(self, x):
        """Sine fourier transformation on x
        """
        return np.sin((2 * np.pi * x) / x.max())

    def _cosine_transform(self, x):
        """Cosine fourier transformation on x
        """
        return np.cos((2 * np.pi * x) / x.max())

    def _more_tags(self):
        return {'stateless': True}


class TimeIndex(BaseEstimator, TransformerMixin):
    """Encodes datetime features with a time index.

    Parameters
    ---------
    start_idx : int
        Integer (including 0) where the time index will start
    """

    def __init__(self, start_idx=0):
        self.start_idx = start_idx
        self.dtype = int

    def fit(self, X, y=None):
        if not hasattr(X, 'dt'):
            raise ValueError('X must be a datetime pandas Series with '
                             'datetime attribute `dt`, i.e., X.dt')

        X = X.sort_values()
        real_dates = X.tolist()
        time_idx = range(self.start_idx, len(X) + self.start_idx)
        self.mapping_ = dict(zip(real_dates, time_idx))
        return self

    def transform(self, X):
        if not hasattr(X, 'dt'):
            raise ValueError('X must be a datetime pandas Series with '
                             'datetime attribute `dt`, i.e., X.dt')

        # The output of the transformer should be 2D
        return X.map(self.mapping_).values.reshape(-1, 1)

    def inverse_transform(self, X):
        if isinstance(X, pd.DataFrame):
            if X.shape[1] != 1:
                raise ValueError(
                    'Inverse transform only accepts pandas Series or pandas'
                    'DataFrame with a single column'
                )
            X = X.iloc[:, 0]
        inverse_map = {v: k for k, v in self.mapping_.items()}
        return X.map(inverse_map).values.reshape(-1, 1)

    def _more_tags(self):
        return {'stateless': True}