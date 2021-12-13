"""
Module for transformers whose fit and transform methods are done by subsets of
the data. That is, each subset (along any axis) is fitted to its own
transformer.
"""

import pandas as pd
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.validation import check_is_fitted

from ..utils import loc_group


def _get_column_settings(X, column_transformer):
    """Private function that retrieves columns configuration from
    ``column_transformer`` object for later use in transform and
    inverse_transform methods.

    Parameters
    ----------
    X : pd.DataFrame
        pandas DataFrame fitted to ``column_transformer``

    column_transformer : sklearn ColumnTransformer object
        fitted sklearn ColumnTransformer object

    Returns
    -------
    columns_settings : dict
        dictionary with keys 'columns_by_names', 'transformed_dtypes' and
        'original_dtypes'. Description for each:

        - columns_by_name : columns grouped by transformer names
        - transformed_dtypes : dtypes for transformed columns.
        - original_dtypes : dtypes prior to transformation

    Notes
    -----
    sklearn 1.0+ seems to have a nicer way to access this information.
    """
    columns_by_name = {}
    transformed_dtypes = {}
    original_dtypes = {}
    for name, transformer, features in column_transformer.transformers_:

        # One dimensional transformers have a single str as features
        # instead of a list of str like the other cases.
        if isinstance(features, str):
            features = [features]

        # Obtain transformed columns/features (``feature_names``)
        if hasattr(transformer, 'get_feature_names'):
            feature_names = transformer.get_feature_names().tolist()
        elif transformer == 'passthrough':
            feature_names = X.columns[features].tolist()
            features = feature_names
        else:
            feature_names = features

        # Obtain transformed dtypes (``t_dtypes``)
        if hasattr(transformer, 'dtype'):
            t_dtypes = {x: transformer.dtype for x in feature_names}
        else:
            t_dtypes = X[feature_names].dtypes.astype('str').to_dict()

        # Obtain original dtypes (``o_dtypes``)
        if (hasattr(transformer, 'inverse_transform')
                or transformer == 'passthrough'):
            if hasattr(transformer, 'get_feature_names'):
                o_dtypes = X[features].dtypes.astype('str').to_dict()
            else:
                o_dtypes = X[feature_names].dtypes.astype('str').to_dict()
        else:
            o_dtypes = {}

        # Save
        columns_by_name.update({name: (feature_names, features)})
        transformed_dtypes.update(t_dtypes)
        original_dtypes.update(o_dtypes)

    return {
        'columns_by_name': columns_by_name,
        'transformed_dtypes': transformed_dtypes,
        'original_dtypes': original_dtypes
    }


def _set_dtypes(X, columns_settings, transformed=True):
    """Private function that sets correct dtypes in X.

    Parameters
    ----------
    X : pd.DataFrame
        DataFrame where dtypes will be set

    columns_settings : dict
        Columns setting as returned by private function _get_columns_settings

    transformed : bool
        Whether to use transformed or original dtypes
    """
    # Retrieve dtypes
    if transformed:
        dtypes = columns_settings['transformed_dtypes']
    else:
        dtypes = columns_settings['original_dtypes']

    # Only use columns present in X
    X_dtypes = {
        column: dtype for column, dtype in dtypes.items() if column in X
    }
    X = X.astype(X_dtypes, copy=False)
    return X


def _inverse_transform_columns(X, transformer, columns_to_inv, columns):
    """Private function that performs an inverse transformation on the given
    ``columns_to_inv`` in X.

    An empty DataFrame is returned when inverse transformation is not
    possible.

    Parameters
    ----------
    X : pd.DataFrame
        DataFrame to inverse transform

    transformer : transformer estimator or 'passthrough'

    columns_to_inv : list of str
        Columns to be inverse transformed

    columns : list of str
        Name of resulting inverse transformed columns.

    Returns
    -------
    X_inv : pd.DataFrame
        pandas DataFrame with inverse transformed columns. An empty DataFrame
        is returned when inverse transformation is not possible.
    """
    if hasattr(transformer, 'inverse_transform'):
        missing_columns = [c for c in columns_to_inv if c not in X]
        if missing_columns:
            X_inv = pd.DataFrame()  # Empty DataFrame
        else:
            X_inv = pd.DataFrame(
                transformer.inverse_transform(X[columns_to_inv]),
                columns=columns
            )
    elif transformer == 'passthrough':
        columns_to_inv = [c for c in columns_to_inv if c in X]
        X_inv = X[columns_to_inv].copy()
    else:
        X_inv = pd.DataFrame()  # Empty DataFrame
    return X_inv.reset_index(drop=True)


def _column_transformer_inverse_transform(
        X,
        column_transformer,
        columns_settings
):
    """Best effort for inverse transformation of a ColumnTransformer instance

    Parameters
    ----------
    X : pd.DataFrame
        DataFrame to be inverse transformed

    column_transformer : sklearn ColumnTransformer object
        Fitted sklearn ColumnTransformer object

    columns_settings : dict
        Columns setting as returned by private function _get_columns_settings
    """
    inv_transforms = []
    columns_by_name = columns_settings['columns_by_name']
    for name, (columns_to_inv, columns) in columns_by_name.items():
        # Each transformer only inv transforms the columns it
        # originally fitted and is retrieved using its given ``name``.
        transformer = column_transformer.named_transformers_[name]
        x_inv = _inverse_transform_columns(
            X, transformer, columns_to_inv, columns)
        inv_transforms.append(x_inv)
    return pd.concat(inv_transforms, axis=1)


class GroupTransformer(BaseEstimator, TransformerMixin):
    """Transformer that transforms by groups

    For each group, a sklearn :class:`ColumnTransformer` is fitted and
    applied.

    Notes
    -----
    The order of the columns in the transformed feature matrix follows the
    order of how the columns are specified in the transformers list. Since the
    passthrough kwarg is set, columns not specified in the transformers list
    are added at the right to the output.

    Parameters
    ----------
    transformers : list of 3tuples
        List of (name, transformer, columns) tuples specifying the transformer
        objects to be applied to subsets of the data.

    group_ids : list of str
         List of column names identifying a group. This means
         that the ``group_ids`` identify a sample together with a time
         dimension.

    Attributes
    ----------
    mapping_ : dict, str -> ColumnTransformer object
        Dictionary mapping from group_id to its corresponding fitted
        ColumnTransformer object
    """

    def __init__(self, transformers, group_ids):
        self.transformers = transformers
        self.group_ids = group_ids

    def fit(self, X, y=None):
        """Fits a sklearn ColumnTransformer object to each group inside ``X``.

        In other words, each group in ``X`` gets assigned its own
        :class:`ColumnTransformer` instance which is then fitted to the data
        inside such group. All :class:`ColumnTransformer` objects are
        instantiated with the same ``transformers``.

        Parameters
        ----------
        X : pd.DataFrame
            Dataframe having init ``group_ids`` column(s).

        y : None
            This param exists to match sklearn interface, but it should never
            be used.

        Returns
        -------
        self (object): Fitted transformer.
        """
        self._check_group_ids(X)
        self.mapping_ = {}  # mapping from group_id to ColumnTransformer object
        groups = X.groupby(self.group_ids).size().index.tolist()
        for i, id_ in enumerate(groups):
            ct = ColumnTransformer(
                self.transformers, remainder='passthrough', sparse_threshold=0
            )
            group = loc_group(X, self.group_ids, id_)
            ct.fit(group)
            self.mapping_[id_] = ct
            if i == 0:
                # Use first iteration to obtain the columns settings
                self._columns_settings = _get_column_settings(X, ct)
        return self

    def transform(self, X):
        """Transforms every group in X using its corresponding
        :class:`ColumnTransformer`.

        Parameters
        ----------
        X : pd.DataFrame
            Dataframe having init ``group_ids`` column(s).

        Returns
        -------
        pd.DataFrame: Transformed dataframe.
        """
        check_is_fitted(self)
        self._check_group_ids(X)
        holder = []  # holder for all transformed groups
        transformed_order = list(self._columns_settings['transformed_dtypes'])
        for id_, column_transformer in self.mapping_.items():
            group = loc_group(X, self.group_ids, id_)
            if group.empty:
                continue
            group_transformed = column_transformer.transform(group)
            if group_transformed.shape[1] != len(transformed_order):
                continue
            group_transformed = pd.DataFrame(
                group_transformed, columns=transformed_order
            )
            holder.append(group_transformed)
        X_out = pd.concat(holder).reset_index(drop=True)
        X_out = _set_dtypes(X_out, self._columns_settings)
        return X_out

    def inverse_transform(self, X, keep_all_columns=False):
        """Inverse transformation.

        If X contains additional previously unseen column, they will
        be lost unless ``keep_all_columns`` is True. Also, transformed columns
        whose corresponding transformer does not have implemented an
        :meth:`inverse_transform` method will not appear after calling this
        inverse transformation. This causes that the resulting DataFrame
        ``X_out`` won't be equal to the original X, that is, the
        expression X = f-1(f(X)) wont be satisfied.

        Parameters
        ----------
        X : pd.DataFrame
            Dataframe to be inverse transformed.

        keep_all_columns : bool

        Returns
        -------
        X_inv : pd.DataFrame
            Inverse transformed dataframe
        """
        check_is_fitted(self)
        self._check_group_ids(X)
        all_inv = []  # Holder of inv transformations for all groups
        for id_, column_transformer in self.mapping_.items():
            group = loc_group(X, self.group_ids, id_)
            if group.empty:
                continue
            inv_group = _column_transformer_inverse_transform(
                group, column_transformer, self._columns_settings)
            if keep_all_columns:
                left_columns = list(set(group) - set(inv_group))
                inv_group = pd.concat(
                    (inv_group, group[left_columns].reset_index(drop=True)),
                    axis=1,
                )
            all_inv.append(inv_group)

        # Insert all inv transformed data into a pandas DataFrame and set
        # correct dtypes.
        X_inv = pd.concat(all_inv, axis=0).reset_index(drop=True)
        X_inv = _set_dtypes(X_inv, self._columns_settings, transformed=False)
        return X_inv

    def _check_group_ids(self, X):
        """Checks group_id columns are present in X
        """
        msg = 'group_id column {} not found in X'
        for col in self.group_ids:
            if col not in X:
                raise ValueError(msg.format(col))


class DataframeColumnTransformer(BaseEstimator, TransformerMixin):
    """Wrapper for sklearn :class:`ColumnTransformer` that returns pandas
    DataFrames instead of numpy arrays in transform and inverse_transform
    methods.

    Parameters
    ----------
     transformers : list of 3tuples
        List of (name, transformer, columns) tuples specifying the transformer
        objects to be applied to subsets of the data.

    Attributes
    ----------
    column_transformer_ : sklearn :class:`ColumnTransformer` object
    """

    def __init__(self, transformers):
        self.transformers = transformers

    def fit(self, X, y=None):
        self.column_transformer_ = ColumnTransformer(
            self.transformers, sparse_threshold=0, remainder='passthrough'
        )
        self.column_transformer_.fit(X)
        self._columns_settings = _get_column_settings(
            X, self.column_transformer_
        )
        return self

    def transform(self, X):
        """Transform X separately by each transformer and concatenate results
        in a single pandas DataFrame

        Parameters
        ----------
        X : pd.DataFrame
            DataFrame to be transformed by subsets

        Returns
        -------
        X_out : pd.DataFrame
            Horizontally stacked results of transformers
        """
        arr = self.column_transformer_.transform(X)  # numpy array
        transformed_order = list(self._columns_settings['transformed_dtypes'])
        X_out = pd.DataFrame(arr, columns=transformed_order)
        X_out = _set_dtypes(X_out, self._columns_settings)
        return X_out

    def inverse_transform(self, X):
        """Inverse transforms X separately by each transformer and concatenate
        results in a single pandas DataFrame.

        Transformed columns whose corresponding transformer does not have
        implemented an :meth:`inverse_transform` method will not appear
        after calling this inverse transformation. This causes that the
        resulting DataFrame ``X_out`` won't be equal to the original X, that
        is, the expression X = f-1(f(X)) wont be satisfied.


        Parameters
        ----------
        X : pd.DataFrame
            DataFrame to be inverse transformed by subsets

        Returns
        -------
        X_out : pd.DataFrame
        """
        X_inv = _column_transformer_inverse_transform(
            X, self.column_transformer_, self._columns_settings)
        X_inv = _set_dtypes(X_inv, self._columns_settings, transformed=False)
        return X_inv


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
