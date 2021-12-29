"""mooncake utilities.

Should not have any dependency on other mooncake packages.
"""

import datetime as dt
import inspect

import numpy as np
import pandas as pd
from sklearn.compose._column_transformer import make_column_selector


def group_index_generator(X, group_ids, sl, ids_to_yield=None):
    """The given slice (``sl`` param) is applied to each group separately

    Parameters
    ----------
    X : pd.DataFrame

    group_ids : list of str
        Group ids identifying the groups

    sl : Slice object

    ids_to_yield : list, default=None
        List of groups whose index will bi yield. If None, all groups are used

    Yields
    ------
    tuple (group_id, index)
    """
    if not isinstance(sl, slice):
        name = type(sl).__name__
        raise ValueError(
            f'sl parameter must be a slice object. Instead got {name}'
        )
    if ids_to_yield is None:
        ids_to_yield = X.groupby(group_ids).groups.keys()
    X_slice = X.groupby(group_ids).apply(lambda x: x.iloc[sl]).loc[
        ids_to_yield]
    for id_ in ids_to_yield:
        yield id_, X_slice.loc[id_].index


def safe_math_eval(string):
    """Evaluates simple math expression

    Since built-in eval is dangerous, this function limits the possible
    characters to evaluate.

    Parameters
    ----------
    string : str

    Returns
    -------
    evaluated ``string`` : float
    """
    allowed_chars = "0123456789+-*(). /"
    for char in string:
        if char not in allowed_chars:
            raise ValueError("Unsafe eval")
    return eval(string, {"__builtins__": None}, {})


class column_selector(make_column_selector):
    """Creates a callable to select columns to be used with
    :class:`ColumnTransformer`

    Parameters
    ----------
    pattern_include : str or list, default=None
        A selection of columns to include.

        - If None, column selection will not be performed based on this param.
        - If list, the elements will be joined using the regex '|' operator.
            Columns matching the resulting regex will be selected
        - if str, the pattern is used as regex and columns matching will
            be selected

    pattern_exclude : str or list, default=None
        A selection of columns to exclude.

        - If None, column selection will not be performed based on this param.
        - If list, the elements will be joined using the regex '|' operator.
            Columns matching the resulting regex will be omitted from
            selection.
        - If str, the pattern is used as regex and columns matching will be
            omitted from selection.

    dtype_include : column dtype or list of column dtypes, default=None
        A selection of dtypes to include.

    dtype_exclude : column dtype or list of column dtypes, default=None
        A selection of dtypes to exclude.

    Returns
    -------
    selector : callable
        Callable for column selection to be used by a
        :class:`ColumnTransformer`.
    """

    def __init__(self, pattern_include=None, pattern_exclude=None,
                 dtype_include=None, dtype_exclude=None):
        super().__init__(
            pattern=self._to_regex(pattern_include),
            dtype_include=dtype_include,
            dtype_exclude=dtype_exclude
        )
        self.pattern_exclude = self._to_regex(pattern_exclude)

    def __call__(self, X):
        cols = pd.Series(super().__call__(X))
        if self.pattern_exclude is not None:
            cols = pd.Series(cols)
            cols = cols[-cols.str.contains(self.pattern_exclude, regex=True)]
        return cols.tolist()

    def _to_regex(self, x, join_with='|'):
        if isinstance(x, list):
            return join_with.join(x)
        return x


def undo_sliding_window(X, step=1, reverse=False):
    """Inverse transformation for sliding windows data structure

    Parameters
    ----------
    X : np.ndarray
        ndarray containing arrays in sliding format

    step : int
        step used for constructing the sliding windows

    reverse : bool
        If True, the inverse transfrom is performed fromm end to start.
        Otherwise, from start to end.

    Returns
    -------
    np.ndarray

    Examples
    --------
    # Undo sliding window with sequence_length=3 and step=1
    >>> X = np.array([[1, 2, 3], [2, 3, 4], [3, 4, 5]])
    >>> undo_sliding_window(X, step=1)
    array([[1],
           [2],
           [3],
           [4],
           [5]])
    """
    if not len(X.shape) > 1:
        raise ValueError(
            'X must have 2 or more dimensions of any size.'
            'Instead got shape: {}'.format(X.shape)
        )
    np_X = np.array(X)
    if len(np_X.shape) == 2:
        np_X = np.expand_dims(np_X, axis=2)
    a = np.concatenate(np_X[:, :step, :])
    b = np_X[-1, step:, :]
    inverse = np.concatenate((a, b))
    return inverse if not reverse else inverse[::-1]


def loc_group(X, group_ids, id):
    """Auxiliary for locating rows in dataframes with one or multiple group_ids

    Parameters
    ----------
    X : pd.DataFrame
        Dataframe to filter

    group_ids: tuple
        Tuple of columns names

    id : tuple
        Id of the wanted group

    Returns
    -------
    pd.DataFrame
    """
    # Broadcasted numpy comparison
    return X[(X[group_ids].values == id).all(1)].copy()


def set_date_on_freq(d, freq='W'):
    """"Maps ``d`` to the given ``freq``

    Parameters
    ----------
    d : str or Timestamp
    freq : str, {'D', 'W', 'M'}, default='W'

    Returns
    -------
    Timestamp

    Example
    -------
    >>> date = '2020-01-01'
    >>> freq = 'W'
    >>> set_date_on_freq(date, freq)
    Timestamp('2020-01-05 00:00:00')

    Raises
    ------
    ValueError if ``freq`` is not from {'D', 'W', 'M'}
    """
    if freq == 'D':
        return d
    elif freq == 'W':
        return next_weekday(d)
    elif freq == 'M':
        return last_day_of_month(d)
    else:
        raise ValueError('frequency {} not recognized'.format(freq))


def last_day_of_month(d):
    """Returns the last day of the month from the given date ``d``.

    Parameters
    ----------
    d : str or Timestamp

    Returns
    -------
    last day of month : Timestamp

    Example
    -------
    >>> d = '2020-01-01'
    >>> last_day_of_month(d)
    Timestamp('2020-01-31 00:00:00')
    """
    # Just find the first day of the next month and then remove a day
    d = pd.Timestamp(d)
    return pd.Timestamp(dt.date(d.year + (d.month == 12),
                                (d.month + 1 if d.month < 12 else 1),
                                1) - dt.timedelta(1))


def date_on_freq(date, freq):
    """Validates if date is on the given frequency.

    In other words, checks if ``date`` matches any ``freq`` timestamp.

    Parameters
    ----------
    date : str or Timestamp
    freq : str

    Returns
    -------
    bool

    Example
    -------
    >>> date = '2020-01-31'
    >>> freq = 'M'
    >>> date_on_freq(date, freq)
    True
    """
    date = pd.Timestamp(date)
    if freq == 'W':
        return True if date.dayofweek == 6 else False
    if freq == 'M':
        # If tomorrow's month is not the same as today's month,
        # then that means today is the last day of the current month.
        todays_month = date.month
        tomorrows_month = (date + dt.timedelta(days=1)).month
        return True if tomorrows_month != todays_month else False


def next_weekday(d, weekday=6):
    """Finds the next closest requested weekday given a particular date ``d``.

    If ``d`` is already the wanted weekday, the same ``d`` is returned.

    Parameters
    ----------
    d : str or timestamp
        Day for which the next requested weekday will be searched.

    weekday : int, default=6 (sunday)
        Wanted weekday

    Returns
    -------
    next weekday : Timestamp

    Examples
    --------
    >>> d = '2021-01-01'
    >>> next_weekday(d)
    Timestamp('2021-01-03 00:00:00')

    >>> d = '2021-01-01' # already weekday=4 (friday)
    >>> next_weekday(d, weekday=4)
    Timestamp('2021-01-01 00:00:00')
    """
    d = pd.Timestamp(d)
    if d.weekday() == weekday:
        return d
    days_ahead = weekday - d.weekday()
    if days_ahead <= 0:  # Target day already happened this week
        days_ahead += 7
    return d + dt.timedelta(days_ahead)


def datedelta(date, n=0, freq='W'):
    """Auxiliary for computing date offsets using a frequency identifier.

    Parameters
    ----------
    date : str or Timestamp

    n : int
        Magnitude of the delta operation.
        Can also be negative for substraction.

    freq : str, {'D', 'W', 'M', 'Q', 'Y'}
        Units to be used

    Example
    -------
    >>> date = '2020-01-01'
    >>> freq = 'W'
    >>> datedelta(date, n=-1, freq='D')
    Timestamp('2019-12-31 00:00:00')

    Returns
    -------
    Timestamp
    """
    if n == 0:
        date_sign = 1
    else:
        date_sign = abs(n) // n
    freq = freq.lower()
    if freq == 'y':
        dtOff = pd.DateOffset(years=abs(n))
    elif freq == 'q':
        dtOff = pd.DateOffset(quarters=abs(n))
    elif freq == 'm':
        dtOff = pd.DateOffset(months=abs(n))
    elif freq == 'w':
        dtOff = pd.DateOffset(weeks=abs(n))
    elif freq == 'd':
        dtOff = pd.DateOffset(days=abs(n))
    else:
        raise ValueError(
            "The freq parameter not one of the following: "
            "{'Y', 'Q', 'M', 'W', 'D'}"
        )
    return pd.to_datetime(date) + date_sign * dtOff


def add_prefix(d, prefix, sep='__'):
    """Adds prefix to keys in d.

    Parameters
    ----------
    d : dict

    prefix : str
        Prefix to be added to each key in d

    sep : str
        Separator between prefix and original key

    Returns
    -------
    dict
    """
    return {prefix + sep + k: v for k, v in d.items()}


def get_init_params(cls):
    """Inspects given class using the inspect package and extracts the
    parameter attribute from its signature

    Parameters
    ----------
    cls : class

    Returns
    -------
    init params : list
    """
    if not inspect.isclass(cls):
        raise ValueError('cls param has to be of type class. Instead '
                         'got {}'.format(type(cls)))
    return inspect.signature(cls.__init__).parameters
