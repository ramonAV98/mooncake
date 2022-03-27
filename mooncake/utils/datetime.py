import datetime as dt

import pandas as pd


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
