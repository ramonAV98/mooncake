import pandas as pd
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.compose import ColumnTransformer as SkColumnTransformer
from sklearn.utils.validation import check_is_fitted

from ..utils.data import loc_group
from ..utils.checks import check_group_ids


class _ColumnTransformerConfig:
    """Easy to use class for accessing commonly used info from a sklearn
    :class:`ColumnTransformer` instance.

    Parameters
    ----------
    column_transformer : fitted sklearn ColumnTransformer.

    X : pd.DataFrame
        Dataframe fitted to the `column_transformer` instance.
    """

    def __init__(self, column_transformer, X):
        check_is_fitted(column_transformer)
        self.column_transformer = column_transformer
        self.X = X

    def _yield_transformers_triplet(self):
        """Yields triplet (name, transformer, features) from
        sklearn `column_transformer` instance.

        Yields
        ------
        (name, transformer, features)
        """
        for triplet in self.column_transformer.transformers_:
            name, transformer, features = triplet

            if not isinstance(features, list):
                features = [features]

            yield name, transformer, features

    def get_dtypes(self):
        """Obtains original and transformed dtypes.

        Returns
        -------
        dtypes : dict
            Dictionary with keys 'transformed_dtypes' and 'original_dtypes'.
        """
        transformed_dtypes = {}
        original_dtypes = {}
        for name, transformer, features in self._yield_transformers_triplet():
            # Notation warning: `features` aka 'features_in' refers to the
            # column names prior to transformation. Conversely, 'feature_names'
            # aka 'features_out' refers to the columns names after
            # transformation.

            feature_names = self._get_feature_names(transformer, features)

            # Transformed dtypes
            transformed_dtypes.update(
                self._get_transformed_dtypes(transformer, feature_names))

            # Original dtypes
            original_dtypes.update(
                self._get_original_dtypes(transformer, feature_names, features)
            )

        return {
            'transformed_dtypes': transformed_dtypes,
            'original_dtypes': original_dtypes
        }

    def get_transformer_by_name(self, name):
        """Obtains transformer instance by its name.

        Parameters
        ----------
        name : str

        Returns
        -------
        transformer
        """
        return self.column_transformer.named_transformers_[name]

    def get_columns_by_name(self, name):
        """Obtains original columns by transformer name.

        Parameters
        ----------
        name : str

        Returns
        -------
        columns : list
        """
        for name_, transformer, features in self._yield_transformers_triplet():
            if name_ == name:
                return features
        raise ValueError('Name {} not found'.format(name))

    def get_transformed_columns_by_name(self, name):
        """Obtains transformed columns by transformer name.

        Parameters
        ----------
        name : str

        Returns
        -------
        columns : list
        """
        for name_, transformer, features in self._yield_transformers_triplet():
            if name_ == name:
                transformed_columns = self._get_feature_names(transformer,
                                                              features)
                return transformed_columns
        raise ValueError('Name {} not found'.format(name))

    def get_transformed_order(self):
        """Obtains columns order for transformed X.

        Returns
        -------
        columns : list
        """
        transformed_order = []
        for name, transformer, features in self._yield_transformers_triplet():
            transformed_columns = self.get_transformed_columns_by_name(name)
            transformed_order.extend(transformed_columns)
        return transformed_order

    def _get_feature_names(self, transformer, features):
        if hasattr(transformer, 'get_feature_names'):
            feature_names = transformer.get_feature_names().tolist()
        elif transformer == 'passthrough':
            feature_names = self.X.columns[features].tolist()
        else:
            feature_names = features
        return feature_names

    def _get_transformed_dtypes(self, transformer, feature_names):
        if hasattr(transformer, 'dtype'):
            dtypes = {x: transformer.dtype for x in feature_names}
        else:
            dtypes = self.X[feature_names].dtypes.astype('str').to_dict()

        return dtypes

    def _get_original_dtypes(self, transformer, feature_names, features):
        if (hasattr(transformer, 'inverse_transform')
                or transformer == 'passthrough'):
            if hasattr(transformer, 'get_feature_names'):
                dtypes = self.X[features].dtypes.astype('str').to_dict()
            else:
                dtypes = self.X[feature_names].dtypes.astype('str').to_dict()
        else:
            dtypes = {}

        return dtypes


class _ColumnTransformerInverseTransformer:
    """Effort for inverse transformation of a ColumnTransformer instance.

    Parameters
    ----------
    column_transformer : sklearn ColumnTransformer
        Fitted sklearn :class:`ColumnTransformer` instance.

    columns_config : _ColumnsConfig
    """

    def __init__(self, column_transformer, columns_config):
        check_is_fitted(column_transformer)
        self.column_transformer = column_transformer
        self.columns_config = columns_config

    def inverse_transform(self, X):
        """Inverse transforms X.

        Parameters
        ----------
        X : pd.DataFrame
            DataFrame to be inverse transformed.

        Returns
        -------
        X_inv : pd.DataFrame
        """
        inv_transforms = []
        # columns_by_name = columns_settings['columns_by_name']
        # for name, (columns_to_inv, columns) in columns_by_name.items():
        for triplet in self.column_transformer.transformers_:
            name, transformer, features = triplet

            # Notice each transformer only inv transforms the columns it
            # originally fitted.

            # Columns to be inverse transformed.
            columns_to_inv = self.columns_config. \
                get_transformed_columns_by_name(name)

            # Original columns (names after inverse transformation).
            columns = self.columns_config.get_columns_by_name(name)

            x_inv = self._inverse_transform_columns(
                X, transformer, columns_to_inv, columns)
            inv_transforms.append(x_inv)

        X_inv = pd.concat(inv_transforms, axis=1)
        return X_inv

    def _inverse_transform_columns(self, X, transformer, columns_to_inv,
                                   columns):
        """Performs an inverse transformation on the given ``columns_to_inv``
        in X.

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
            pandas DataFrame with inverse transformed columns. An empty
            DataFrame is returned when inverse transformation is not possible.
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


def _set_dtypes(X, columns_config, transformed=True):
    """Private function that sets correct dtypes in X.

    Parameters
    ----------
    X : pd.DataFrame
        DataFrame where dtypes will be set

    columns_config

    transformed : bool
        Whether to use transformed or original dtypes
    """
    dtypes = columns_config.get_dtypes()
    columns_dtypes = (dtypes['transformed_dtypes']
                      if transformed else dtypes['original_dtypes'])

    # Only use columns present in X.
    X_dtypes = {
        column: dtype for column, dtype
        in columns_dtypes.items() if column in X
    }
    X = X.astype(X_dtypes, copy=False)
    return X


class GroupTransformer(BaseEstimator, TransformerMixin):
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
            ct = SkColumnTransformer(
                self.transformers, remainder='passthrough', sparse_threshold=0)
            group = loc_group(X, self.group_ids, group_id)
            ct.fit(group)
            self.mapping_[group_id] = ct
            if i == 0:
                # Use first iteration to obtain the columns settings.
                self._columns_config = _ColumnTransformerConfig(ct, group)
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

        holder = []  # Holder for all transformed groups.
        transformed_order = self._columns_config.get_transformed_order()
        for group_id, column_transformer in self.mapping_.items():
            group = loc_group(X, self.group_ids, group_id)
            if group.empty:
                continue

            # Transform group.
            group_transformed = column_transformer.transform(group)

            # TODO: explanation of this condition
            if group_transformed.shape[1] != len(transformed_order):
                continue

            # Save transformed group.
            group_transformed = pd.DataFrame(
                group_transformed, columns=transformed_order)
            holder.append(group_transformed)

        X_out = pd.concat(holder).reset_index(drop=True)
        X_out = _set_dtypes(X_out, self._columns_config)
        return X_out

    def inverse_transform(self, X, keep_all_columns=False):
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

        keep_all_columns : bool

        Returns
        -------
        X_inv : pd.DataFrame
            Inverse transformed dataframe
        """
        check_is_fitted(self)
        check_group_ids(X, self.group_ids)

        holder = []  # Holder of inv transformations for all groups.
        for group_id, column_transformer in self.mapping_.items():
            group = loc_group(X, self.group_ids, group_id)
            if group.empty:
                continue

            # Inverse transform group.
            inverse_transformer = _ColumnTransformerInverseTransformer(
                column_transformer, self._columns_config)
            inv_group = inverse_transformer.inverse_transform(group)

            if keep_all_columns:
                # If any column was left out.
                left_out_columns = list(set(group) - set(inv_group))
                left_out_df = group[left_out_columns].reset_index(drop=True)
                inv_group = pd.concat((inv_group, left_out_df), axis=1)

            holder.append(inv_group)

        # Insert all inv transformed data into a pandas DataFrame and set
        # correct dtypes.
        X_inv = pd.concat(holder, axis=0).reset_index(drop=True)
        X_inv = _set_dtypes(X_inv, self._columns_config, transformed=False)
        return X_inv


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
        self.column_transformer_ = SkColumnTransformer(
            self.transformers, sparse_threshold=0, remainder='passthrough'
        )
        self.column_transformer_.fit(X)
        self._columns_config = _ColumnTransformerConfig(
            self.column_transformer_, X)
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
        transformed_order = self._columns_config.get_transformed_order()
        X_out = pd.DataFrame(arr, columns=transformed_order)
        X_out = _set_dtypes(X_out, self._columns_config)
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
        inverse_transformer = _ColumnTransformerInverseTransformer(
            self.column_transformer_, self._columns_config)
        X_inv = inverse_transformer.inverse_transform(X)
        X_inv = _set_dtypes(X_inv, self._columns_config, transformed=False)
        return X_inv
