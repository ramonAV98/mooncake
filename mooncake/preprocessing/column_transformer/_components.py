from abc import ABCMeta, abstractmethod

from sklearn.utils.validation import check_is_fitted


class ComponentsGetter:
    """Get method interface for all :class:`ColumnTransformerComponent`.
    """

    def __init__(self, column_transformer, X):
        self.factory = ComponentsFactory(column_transformer, X)

    def get_columns_by_name(self, name, transformed=True):
        columns_by_name_component = self.factory.get_component(
            'columns_by_name')
        return columns_by_name_component.get(
            name=name, transformed=transformed)

    def get_columns_order(self, transformed=True):
        columns_order_component = self.factory.get_component(
            'columns_order')
        return columns_order_component.get(transformed=transformed)

    def get_columns_dtypes(self, transformed=True):
        columns_dtypes_component = self.factory.get_component('columns_dtypes')
        return columns_dtypes_component.get(transformed=transformed)


class ComponentsFactory:
    """"Factory for ColumnTransformerComponent instances.

    Parameters
    ----------
    column_transformer : sklearn.compose.ColumnTransformer
            Fitted sklearn.compose.ColumnTransformer.

    X : pd.DataFrame
            Non transformed dataframe (used for obtaining remainder column
            names).
    """

    def __init__(self, column_transformer, X):
        self.column_transformer = column_transformer
        self.X = X

    def get_component(self, name):
        """Returns a ColumnTransformerComponent instance.

        Parameters
        ----------
        name : str, {'columns_by_name', 'columns_order'}
            Name of component

        Returns
        -------
        component
        """
        components = {
            'columns_by_name': ColumnsByName,
            'columns_order': ColumnsOrder,
            'columns_dtypes': ColumnsDtypes
        }

        return components[name](self.column_transformer, self.X)


class ColumnTransformerComponent(metaclass=ABCMeta):
    """Base class for sklearn ColumnTransformer components.

    Parameters
    ----------
    column_transformer : sklearn.compose.ColumnTransformer
        Fitted sklearn.compose.ColumnTransformer.

    X : pd.DataFrame
        Non transformed dataframe (used for obtaining remainder column names).
    """

    def __init__(self, column_transformer, X):
        check_is_fitted(column_transformer)
        self.column_transformer = column_transformer
        self.X = X

    @abstractmethod
    def get(self, transformed):
        """Every derived class must implement this method.

        Parameters
        ----------
        transformed : bool
            Whether or not to return the transformed version of the component.
        """
        pass

    def generator(self):
        """Yields (name, transformer, features) triplets.

        Meaning of each element:
        - name (str): user-defined name given to the transformer.
        - transformer (transformer): fitted transformer.
        - features (list): list of column names to be fitted.

        For any fitted sklearn ColumnTransformer, such triplets are stored
        inside the transformers_ attribute,
        i.e. `column_transformer.transformers_`.

        Yields
        ------
        tuple : (name, transformer, features) triplets.
        """
        for triplet in self.column_transformer.transformers_:
            name, transformer, features = triplet

            if not isinstance(features, list):
                features = [features]

            yield name, transformer, features

    def get_transformer_by_name(self, name):
        """Returns transformer instance by its name.

        Parameters
        ----------
        name : str

        Returns
        -------
        transformer : sklearn transformer
            Fitted sklearn transformer.
        """
        return self.column_transformer.named_transformers_[name]

    def get_remainder_names(self, remainder_index):
        """Returns names of remainder columns.

        Parameters
        ----------
        remainder_index : list of int
            Index position of remainder columns (same format as sklearn
            remainder output).

        Returns
        -------
        names : list
            Names of remainder columns.
        """
        return self.X.columns[remainder_index].tolist()

    def get_transformed_names(self, name, transformer, features):
        """Returns column names after transformation.
        """
        if name == 'remainder':
            return self.get_remainder_names(features)

        if hasattr(transformer, 'get_feature_names') and features:
            # If `features` is empty, then the transformer was never fitted,
            # so calling `get_feature_names()` will throw NoFittedError. Hence,
            # the condition also checks `features` is non empty.
            return transformer.get_feature_names().tolist()

        return features

    def get_non_transformed_names(self, name, features):
        if name == 'remainder':
            return self.get_remainder_names(features)
        return features


class ColumnsByName(ColumnTransformerComponent):

    def __init__(self, column_transformer, X):
        super().__init__(column_transformer, X)

    def get(self, name, transformed=True):
        """Returns columns given a transformer name.

        Parameters
        ----------
        name : str
            Name of transformer.

        transformed : bool, default=True
            Whether or not to return the transformed column names.

        Returns
        -------
        columns : list
            List of column names.
        """

        for name_, transformer, features in self.generator():
            if name == name_:
                if transformed:
                    return self.get_transformed_names(
                        name, transformer, features)
                else:
                    return self.get_non_transformed_names(name, features)

        raise ValueError(
            'Name "{}" not found in `column_transformer` instance'.format(name)
        )


class ColumnsOrder(ColumnTransformerComponent):
    def __init__(self, column_transformer, X):
        super().__init__(column_transformer, X)

    def get(self, transformed=True):
        """Returns columns order.

        The order of the columns in the transformed dataframe follows the
        order of how the columns are specified in the transformers list.

        Parameters
        ----------
        transformed : bool, default=True
            Whether or not to return the transformed column order.

        Returns
        -------
        columns_order : list
            List of ordered column names.
        """
        columns_order = []
        factory = ComponentsFactory(self.column_transformer, self.X)
        columns_by_name_component = factory.get_component(
            name='columns_by_name')
        for name, transformer, features in self.generator():
            columns = columns_by_name_component.get(
                name=name, transformed=transformed)
            columns_order.extend(columns)
        return columns_order


class ColumnsDtypes(ColumnTransformerComponent):

    def __init__(self, column_transformer, X):
        super().__init__(column_transformer, X)

    def get(self, transformed=True):
        if not transformed:
            return self.X.dtypes.to_dict()
        return self._transformed()

    def _transformed(self):
        # Instantiate `columns_by_name` component.
        factory = ComponentsFactory(self.column_transformer, self.X)
        columns_by_name_component = factory.get_component(
            name='columns_by_name')

        dtypes = {}
        for name, transformer, features in self.generator():
            columns = columns_by_name_component.get(name)
            if hasattr(transformer, 'dtype'):
                d = {x: transformer.dtype for x in columns}
            else:
                d = self.X[columns].dtypes.to_dict()
            dtypes.update(d)
        return dtypes
