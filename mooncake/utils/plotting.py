import matplotlib.pyplot as plt

from .data import (check_group_ids,
                   loc_group,
                   check_is_datetime,
                   check_group_id_presence)


class _GroupPlotter:
    """Plotter for DataFrames containing grouped timeseries.

    Parameters
    ----------
    group_ids : list of str
        List of column names identifying a time series. This means that the
        `group_ids` identify a sample together with the `timestamp_column`.

    timestamp_column : str
        Name of datetime column

    column_to_plot : str or None, default=None
        Column to plot. If None, it is expected `X` contains a single column
        (the one to be plotted) apart from the `group_ids`.
    """
    def __init__(self, group_ids, timestamp_column, column_to_plot=None):
        self.group_ids = group_ids
        self.timestamp_column = timestamp_column
        self.column_to_plot = column_to_plot

    def plot(self, X, group_id, ax=None):
        """Plots a single group.
        """
        if ax is None:
            fig, ax = plt.subplots()

        group = loc_group(X, self.group_ids, group_id)
        group = self._set_timestamp_index(group)
        column_to_plot = self._validate_column_to_plot(group)
        group.plot(y=column_to_plot, style='.-', ax=ax)
        return ax

    def _set_timestamp_index(self, X):
        return X.set_index(self.timestamp_column)

    def _validate_column_to_plot(self, X):
        """Validate there exists a valid column to plot.
        """
        non_group_ids_columns = self._get_non_group_ids_columns(X)

        if len(non_group_ids_columns) > 1:
            if self.column_to_plot is None:
                raise ValueError('`column_to_plot` must be specified when X '
                                 'contains multiple columns.')
            else:
                column_to_plot = self.column_to_plot

        else:
            column_to_plot = non_group_ids_columns[0]

        return column_to_plot

    def _get_non_group_ids_columns(self, X):
        """Obtains all columns that are not part of the `group_ids`.
        """
        non_group_ids_columns = [x for x in X if x not in self.group_ids]
        if not non_group_ids_columns:
            raise
        return non_group_ids_columns


def plot_group_series(X, group_ids, timestamp_column, group_id,
                      column_to_plot=None, ax=None):

    """Plots a single group.

    Parameters
    ----------
    X : pd.DataFrame
        DataFrames containing grouped timeseries.

    group_ids : list of str
        List of column names identifying a time series. This means that the
        `group_ids` identify a sample together with the `timestamp_column`.

    timestamp_column : str
        Name of datetime column

    group_id : str or tuple
        Group id to plot.

    column_to_plot : str or None, default=None
        Column to plot. If None, it is expected `X` contains a single column
        (the one to be plotted) apart from the `group_ids`.

    ax : matplotlib axis

    Returns
    -------
    ax : matplotlib axis
    """
    check_group_ids(X, group_ids)
    check_is_datetime(X[timestamp_column])
    check_group_id_presence(X, group_ids, group_id)
    plotter = _GroupPlotter(group_ids, timestamp_column, column_to_plot)
    ax = plotter.plot(X, group_id, ax)
    return ax





