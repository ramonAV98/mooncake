from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array
import pandas as pd
import numpy as np


def _identity(X):
    """The identity function."""
    return X


class IdentityTransformer(BaseEstimator, TransformerMixin):
    """Identity transformer.


    Can be inserted into a sklearn Pipeline for duplicating columns. For
    example, this transformer makes it possible to apply two transformations
    to the same by duplicating the existing one.

    Parameters
    ----------
    out_feature : str
        Name of output column.

    cast_to_object : bool, default=False
        Whether or not the returned array in transform method is converted to
        type object.

    dtype : np.dtype, default=None
        Transformer dtype attribute. Used for determining the original data
        dtype. Example: np.dtype('<M8[ns]')

    inverse_func : callable, default=None
        The callable to use for the inverse transformation. This will be
        passed the same arguments as inverse transform, with args and
        kwargs forwarded. If inverse_func is None, then inverse_func
        will be the identity function.
    """

    def __init__(self, out_feature, cast_to_object=False, dtype=None,
                 inverse_func=None):
        self.out_feature = out_feature
        self.cast_to_object = cast_to_object
        self.dtype = dtype
        self.inverse_func = inverse_func

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        check_array(X)

        if isinstance(X, pd.DataFrame):
            X = X.values

        if self.cast_to_object:
            X = X.astype(object)

        return X

    def get_feature_names(self):
        return np.array([self.out_feature])

    def inverse_transform(self, X):
        if self.inverse_func is None:
            return _identity(X)
        return self.inverse_func(X)
