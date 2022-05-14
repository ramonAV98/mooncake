import pandas as pd
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.compose import ColumnTransformer as SkColumnTransformer
from sklearn.utils.validation import check_is_fitted

from mooncake.utils.checks import check_group_ids
from mooncake.utils.data import loc_group
from ._components import ComponentsGetter
from ._inverse_transformer import ColumnTransformerInverseTransformer
from ...utils.data import numpy_2d_to_pandas


def _instantiate_sklearn_column_transformer(transformers):
    """Instantiates :class:`sklearn.compose.ColumnTransformer`.
    """
    return SkColumnTransformer(
        transformers=transformers, remainder='passthrough', sparse_threshold=0)


def _to_pandas(arrays, components_getter, transformed=True,
               inverse_transformer=None):
    """Converts collection of 2-D arrays to a pandas DataFrame.
    """
    # Columns and dtypes for output pandas DataFrame.
    columns_order = components_getter.get_columns_order(transformed)
    dtypes = components_getter.get_columns_dtypes(transformed)

    # Check for non invertible columns.
    if inverse_transformer is not None:
        # Non invertible columns do not appear in final pandas DataFrame, so
        # they must be removed from the conversion.
        non_invertible = inverse_transformer.get_non_invertible_columns()
        for column in non_invertible:
            columns_order.remove(column)
            # It is possible the same column name appears more than once in
            # the `non_invertible` list. While it is not problem for the list
            # `columns_order` to have duplicates it is for the dict `dtypes`.
            # Additionally, if there still exists a duplicate in
            # `columns_order`, it is not deleted from `dtypes`.
            if column in dtypes and column not in columns_order:
                dtypes.pop(column)

    if not isinstance(arrays, list):
        arrays = [arrays]

    return numpy_2d_to_pandas(arrays, columns_order, dtypes)


class GroupColumnTransformer(BaseEstimator, TransformerMixin):
    """Transformer that transforms by groups.

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
        check_group_ids(X, self.group_ids)

        # Mapping from group_id to ColumnTransformer object.
        self.mapping_ = {}

        groups = X.groupby(self.group_ids).groups
        for i, group_id in enumerate(groups):
            ct = _instantiate_sklearn_column_transformer(self.transformers)
            group = loc_group(X, self.group_ids, group_id)
            ct.fit(group)
            self.mapping_[group_id] = ct

            if i == 0:
                self._components_getter = ComponentsGetter(ct, group)
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
        check_group_ids(X, self.group_ids)

        arrays = []
        for group_id, column_transformer in self.mapping_.items():
            group = loc_group(X, self.group_ids, group_id)
            if not group.empty:
                transformed_group = column_transformer.transform(group)
                arrays.append(transformed_group)

        return _to_pandas(arrays, self._components_getter)

    def inverse_transform(self, X):
        """Inverse transformation.

        If X contains additional previously unseen column, they will
        be lost unless ``keep_all_columns`` is True. Also, transformed columns
        whose corresponding transformer does not have implemented an
        :meth:`inverse_transform` method will not appear after calling this
        inverse transformation. This causes that the resulting DataFrame
        ``X_out`` might not be equal to the original X, that is, the
        expression X = f-1(f(X)) wont be satisfied.

        Parameters
        ----------
        X : pd.DataFrame
            Dataframe to be inverse transformed.


        Returns
        -------
        X_inv : pd.DataFrame
            Inverse transformed dataframe
        """
        check_is_fitted(self)
        check_group_ids(X, self.group_ids)

        arrays = []  # Holder of inv transformations for all groups.
        for group_id, column_transformer in self.mapping_.items():
            group = loc_group(X, self.group_ids, group_id)
            if not group.empty:
                inverse_transformer = ColumnTransformerInverseTransformer(
                    column_transformer, self._components_getter)
                inv_group = inverse_transformer.inverse_transform(group)
                arrays.append(inv_group)
        return _to_pandas(
            arrays, self._components_getter, transformed=False,
            inverse_transformer=inverse_transformer)


class ColumnTransformer(BaseEstimator, TransformerMixin):
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
        self.column_transformer_ = _instantiate_sklearn_column_transformer(
            self.transformers)
        self.column_transformer_.fit(X)
        self._components_getter = ComponentsGetter(self.column_transformer_, X)
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
        arr = self.column_transformer_.transform(X)
        return _to_pandas(arr, self._components_getter)

    def inverse_transform(self, X):
        """Inverse transforms X separately by each transformer and concatenate
        results in a single pandas DataFrame.

        Transformed columns whose corresponding transformer does not have
        implemented an :meth:`inverse_transform` method will not appear
        after calling this inverse transformation. Hence, it is possible the
        resulting DataFrame ``X_out``  is not equal to the original X, that
        is, the expression X = f-1(f(X)) wont be satisfied.


        Parameters
        ----------
        X : pd.DataFrame
            DataFrame to be inverse transformed by subsets

        Returns
        -------
        X_out : pd.DataFrame
        """
        inverse_transformer = ColumnTransformerInverseTransformer(
            self.column_transformer_, self._components_getter)
        inv_X = inverse_transformer.inverse_transform(X)
        return inverse_transformer, _to_pandas(inv_X, self._components_getter, transformed=False,
                          inverse_transformer=inverse_transformer)
