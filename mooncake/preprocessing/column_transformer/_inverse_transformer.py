import numpy as np
from sklearn.utils.validation import check_is_fitted


class ColumnTransformerInverseTransformer:
    """Inverse transformation of a ColumnTransformer instance.

    Parameters
    ----------
    column_transformer : sklearn ColumnTransformer
        Fitted sklearn :class:`ColumnTransformer` instance.
    """

    def __init__(self, column_transformer, components_getter):
        check_is_fitted(column_transformer)
        self.column_transformer = column_transformer
        self.components_getter = components_getter

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
        self._non_invertible_columns = []
        inv_transforms = []

        # Notice each transformer only inv transforms the columns it
        # originally fitted.
        for triplet in self.column_transformer.transformers_:
            name, transformer, features = triplet

            # Obtain columns to inverse transform.
            columns_to_inv = self.components_getter.get_columns_by_name(name)

            # Inverse transform.
            columns_inverse_transformed = self._inverse_transform_columns(
                X, transformer, columns_to_inv)

            # Check for non invertible columns.
            if columns_inverse_transformed is None:
                non_invertible = self.components_getter.get_columns_by_name(
                    name, transformed=False)
                self._non_invertible_columns.extend(non_invertible)
            else:
                inv_transforms.append(columns_inverse_transformed)

        # Stack inverse transformations of all columns (axis=1).
        X_inv = np.hstack(inv_transforms)
        return X_inv

    def get_non_invertible_columns(self):
        return self._non_invertible_columns

    def _inverse_transform_columns(self, X, transformer, columns_to_inv):
        """Performs an inverse transformation on the given `columns_to_inv`
        in `X`.

        An empty DataFrame is returned when inverse transformation is not
        possible.

        Parameters
        ----------
        X : pd.DataFrame
            DataFrame to be inverse transformed.

        transformer : transformer estimator or 'passthrough'

        columns_to_inv : list of str
            Columns to be inverse transformed.

        Returns
        -------
        X_inv : 2-D numpy array
            Array with inverse transformed columns. An empty array is returned
            when inverse transformation is not possible.
        """
        missing_columns = [c for c in columns_to_inv if c not in X]

        if hasattr(transformer, 'inverse_transform'):
            if missing_columns:
                return None
            return transformer.inverse_transform(X[columns_to_inv])

        if transformer == 'passthrough':
            # For this case, the inverse transformation can proceed even with
            # missing columns. However, they are added to the non invertible
            # list.
            if missing_columns:
                self._non_invertible_columns.extend(missing_columns)
            columns_to_inv = [c for c in columns_to_inv if c in X]
            return X[columns_to_inv].values
        return None
