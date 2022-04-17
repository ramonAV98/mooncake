from abc import ABCMeta
from abc import abstractmethod

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.utils import check_consistent_length


class response_function(metaclass=ABCMeta):
    """A generic response function class meant for subclassing.

    `response_function` is a base class to construct specific response function
    classes and instances. It cannot be used directly.
    """

    @abstractmethod
    def fit(self, x, y):
        """Returns response function parameters estimates.

        Parameters
        ----------
        x : array_like
        y : array_like

        Returns
        -------
        parameter_tuple : tuple of floats
            tuple containing the estimates for the response function

        Examples
        --------
        >>> x = [1, 2, 3]
        >>> y = [4, 5, 6]
        >>> alpha, beta = constant_elasticity.fit(x, y)
        """
        pass

    @abstractmethod
    def f(self, x, *args):
        """Evaluates response function at `x`.

        Parameters
        ----------
        x : array_like

        arg1, arg2, arg3,... : floats
            response function parameters

        Returns
        -------
        f(x) : array_like
            response function evaluated at `x`.
        """
        pass

    @abstractmethod
    def elasticity(self, x, *args):
        """Calculates the elasticity at `x` for the response function.

        Parameters
        ----------
        x : array_like

        arg1, arg2, arg3,... : floats
            response function parameters.

        Returns
        -------
        elasticity : array_like
            Elasticities at x.
        """
        pass


class constant_elasticity_gen(response_function):
    r"""A constant elasticity response function.

    Notes
    -----
    The constant elasticity response function is defined as

    .. math::

        f(x) = \alpha x^{\beta}

    for a real number :math:`x`.
    """

    def fit(self, x, y):
        check_consistent_length(x, y)

        # Select positive values.
        # For each tuple (x, y) both elements must be positive
        xy = np.array([x, y])
        x, y = xy[:, np.all(xy > 0, axis=0)]

        # Reshape x into 2d
        x = x.reshape(-1, 1)

        # Transform both x and y to ln
        ln_x, ln_y = map(np.log, (x, y))

        # Fit linear regression and return estimated parameters
        model = LinearRegression().fit(ln_x, ln_y)
        alpha = np.exp(model.intercept_)
        beta = model.coef_
        return alpha.item(), beta.item()

    def f(self, x, *args):
        alpha, beta = args
        return alpha * np.power(x, beta)

    def elasticity(self, x, *args):
        alpha, beta = args
        return beta


constant_elasticity = constant_elasticity_gen()

# Add only the response function object names, not the *_gen class names.
__all__ = [
    'constant_elasticity'
]
