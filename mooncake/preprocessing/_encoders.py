import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.validation import check_is_fitted


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
        if isinstance(X, pd.DataFrame):
            X = self._dataframe_to_series(X)

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

    def _dataframe_to_series(self, X):
        if X.shape[1] != 1:
            raise ValueError(
                'CyclicalDates transformer only allows pandas Series or '
                'DataFrames with a single column.'
            )
        return X.iloc[:, 0]

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

    def __init__(self, start_idx=0, extra_timestamps=10, freq='D'):
        self.start_idx = start_idx
        self.extra_timestamps = extra_timestamps
        self.freq = freq
        self.dtype = int

    def fit(self, X, y=None):
        if not hasattr(X, 'dt'):
            raise ValueError('X must be a datetime pandas Series with '
                             'datetime attribute `dt`, i.e., X.dt')

        X = X.sort_values()
        real_dates = X.tolist()
        time_idx = range(self.start_idx, len(X) + self.start_idx)
        self.mapping_ = dict(zip(real_dates, time_idx))
        if self.extra_timestamps > 0:
            self._add_extra_timestamps()
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

    def _add_extra_timestamps(self):
        # Extra date range
        max_timestamp = max(self.mapping_)
        extra_date_range = pd.date_range(
            max_timestamp, periods=self.extra_timestamps, freq=self.freq,
            inclusive='right')
        extra_date_range.freq = None

        # Extra time index.
        # Notice :func:`range()` starts at ``max_time_index + 1`` since it
        # needs to be right inclusive.
        max_time_index = max(self.mapping_.values())
        extra_time_index = range(
            max_time_index + 1,
            max_time_index + len(extra_date_range)
        )

        extra_mapping = dict(zip(extra_date_range, extra_time_index))
        self.mapping_.update(extra_mapping)

    def _more_tags(self):
        return {'stateless': True}


class MultiColumnLabelEncoder(BaseEstimator, TransformerMixin):
    """Label encoder by columns.

    For each column, a sklearn :class:`LabelEncoder` is fitted and applied.

    Used for transforming nominal data (e.g, Holidays, IDs, etc) to a integer
    scale that goes from 0 to n-1 where n is the number of unique values inside
    the column.

    Parameters
    ----------
    columns : list
        Columns to be transformed

    Attributes
    ----------
    mapping_ : dict, str -> LabelEncoder object
        Dictionary mapping from column name to its corresponding fitted
        :class:LabelEncoder object
    """

    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        """Obtains a LabelEncoder object for each column for later
        transformation.

        Each LabelEncoder object contains all the necessary
        information to perform the mapping between the nominal data and its
        corresponding numerical value.

        Parameters
        ----------
        X : pd.DataFrame
            DataFrame to be fitted

        y : None
            This param exists to match sklearn interface, but it should never
            be used.

        Returns
        -------
        self : Fitted transformer
        """
        self.mapping_ = {}  # mapping from column to LabelEncoder object
        for col in self.columns:
            label_enc = LabelEncoder()
            label_enc.fit(X[col])
            self.mapping_[col] = label_enc
        return self

    def transform(self, X):
        """Maps every value inside ``X`` to its numerical counterpart.

        Parameters
        ----------
        X : pd.DataFrame
            Dataframe having init ``columns``

        Returns
        -------
        X : pd.DataFrame
            Copy of ``X`` with ``columns`` numerically encoded
        """
        check_is_fitted(self)
        X = X.copy()
        for col in self.columns:
            label_enc = self.mapping_[col]
            X[col] = label_enc.transform(X[col])
        X[self.columns] = X[self.columns].astype('category')
        return X

    def inverse_transform(self, X):
        """Undos the numerical encoding.

        Parameters
        ----------
        X : pd.DataFrame

        Returns
        -------
        inverse transformed X
        """
        check_is_fitted(self)
        X = X.copy()
        for col in self.columns:
            if col not in self.columns:
                continue
            label_enc = self.mapping_[col]
            X[col] = label_enc.inverse_transform(X[col])
        return X

    def _flip_dict(self, d):
        return {v: k for k, v in d.items()}
