import inspect

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
            raise ValueError("Unsafe eval character: {}".format(char))
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
        if self.pattern_exclude is not None and not cols.empty:
            cols = pd.Series(cols)
            cols = cols[-cols.str.contains(self.pattern_exclude, regex=True)]
            return cols.tolist()
        return cols.tolist()

    def _to_regex(self, x, join_with='|'):
        if isinstance(x, list):
            return join_with.join(x)
        return x


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
