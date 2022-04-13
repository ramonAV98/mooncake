"""
It is encouraged that every cross-validator inherits from sklearn _BaseKFold
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection._split import _BaseKFold

from ..utils.checks import check_is_datetime
from ..utils.data import loc_group
from ..utils.datetime import datedelta, set_date_on_freq


def _loc_between(X, start_date, end_date, date_column):
    """Private function for locating rows between dates

    Parameters
    ----------
    start_date: str or timestamp.
    end_date: str or timestamp.
    """
    period = (start_date, end_date)
    return X.loc[X[date_column].between(*period)]


def train_test_split(X, test_start, test_end, date_column, freq,
                     train_start=None, train_end=None, sequence_length=None):
    """Split dataframe into train and test subsets

    Parameters
    ----------
    X : pd.DataFrame
        Dataframe to split.

    test_start : str or Timestamp
        Test start date.

    test_end : str or Timestamp
        Test end date.

    date_column : str
        Date column

    freq : str, default=None
        Time series frequency. If ``sequence_length`` is not None, this param
        cannot be None.

    train_start : str or Timestamp
        Train start date. If None, the minimum date inside ``date_column`` is
        used

    train_end : str or Timestamp, default=None
        If None, ``train_end`` is given its maximum possible value which is
        one time delta unit less than ``test_start``

    sequence_length : int, default=None
        If not None, the last ``sequence_length`` samples are attached to the
        test data before ``test_start`` (useful for sequential models).
    """
    if date_column not in X:
        raise ValueError(f'Date column {date_column} not found in X')
    check_is_datetime(X[date_column])

    X = X.copy()
    # Map dates to their correct freq
    test_start = set_date_on_freq(test_start, freq)
    test_end = set_date_on_freq(test_end, freq)

    # Train subset
    if train_start is None:
        train_start = X[date_column].min()
    if train_end is None:
        train_end = datedelta(test_start, -1, freq=freq)
    X_train = _loc_between(X, train_start, train_end, date_column)

    # Test subset
    if sequence_length is not None:
        sequence_start = datedelta(
            train_end, -(sequence_length - 1), freq=freq
        )
        X_test = _loc_between(X, sequence_start, test_end, date_column)
    else:
        X_test = _loc_between(X, test_start, test_end, date_column)

    X_train.reset_index(drop=True, inplace=True)
    X_test.reset_index(drop=True, inplace=True)
    return X_train, X_test


class WalkForwardCV(_BaseKFold):
    """Cross-validator for grouped Time Series

    Provides train/test indices to split time series data samples
    that are observed at fixed time intervals, in train/test sets.
    In each split, test indices must be higher than before, and thus shuffling
    in cross validator is inappropriate.

    Parameters
    ----------
    group_ids : list of str
         List of column names identifying a time series

    max_train_size : int, default=None
        Maximum size for a single training set.

    test_size : int, default=None
        Used to limit the size of the test set. Defaults to
        ``n_samples // (n_splits + 1)``, which is the maximum allowed value.
    """

    def __init__(self, group_ids, n_splits=5, max_train_size=None,
                 test_size=None):
        super().__init__(n_splits, shuffle=False, random_state=None)
        self.group_ids = group_ids
        self.max_train_size = max_train_size
        self.test_size = test_size

    def split(self, X, y=None, groups=None):
        """
        Parameters
        ----------
        X : Slice dataset
            Slice version of training dataset. Contains both training and
            validation/test data

        y : None
            Always ignored, exists for compatibility.

        groups : None
            Always ignored, exists for compatibility.

        Returns
        -------
        train-test indices : list of (train, test) tuples
        """
        groups_count = X.decoded_index.groupby(
            self.group_ids, sort=False).size().to_dict()
        indices = dict.fromkeys(
            range(self.n_splits),
            (np.array([], dtype=int), np.array([], dtype=int))
        )
        j = 0
        for groups, count in groups_count.items():
            dummy_array = np.zeros((count, 1))
            tscv = TimeSeriesSplit(
                n_splits=self.n_splits,
                max_train_size=self.max_train_size,
                test_size=self.test_size
            )
            for i, (train_idx, test_idx) in enumerate(tscv.split(dummy_array)):
                i_split_train, i_split_test = indices[i]
                i_split_train = np.append(i_split_train, train_idx + j)
                i_split_test = np.append(i_split_test, test_idx + j)
                indices[i] = i_split_train, i_split_test
            j += count
        return list(indices.values())

    def plot_cv_indices(self, X, y, ax, group_id=None, lw=10):
        """Create a sample plot for indices of a cross-validation object.

        Parameters
        ----------
        X : Slice dataset
            Slice version of training dataset. Contains both training and
            validation/test data

        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            The target variable to try to predict

        ax : matplotlib ax
            Ax to plot on

        Returns
        -------
        ax : matplotlib ax
        """
        import matplotlib.pyplot as plt

        if group_id is not None:
            group_idx = loc_group(X.decoded_index, X.group_ids, group_id).index
            X = X[group_idx.tolist()]
            y = y[group_idx.tolist()]

        # Generate the training/testing visualizations for each CV split
        for ii, (tr, tt) in enumerate(self.split(X=X, y=y)):
            # Fill in indices with the training/test groups
            indices = np.array([np.nan] * len(X))
            indices[tt] = 1
            indices[tr] = 0

            # Visualize the results
            cmap_cv = plt.cm.coolwarm
            ax.scatter(range(len(indices)), [ii + .5] * len(indices),
                       c=indices, marker='_', lw=lw, cmap=cmap_cv,
                       vmin=-.2, vmax=1.2)
        # Formatting
        yticklabels = list(range(self.n_splits))
        ax.set(yticks=np.arange(self.n_splits) + .5, yticklabels=yticklabels,
               ylabel="CV iteration")
        title = '{}'.format(type(self).__name__)
        if group_id is not None:
            title += ' for group id {}'.format(group_id)
        else:
            title += ' for all groups'
        ax.set_title(title, fontsize=15)
        return ax
