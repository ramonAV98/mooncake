import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted

from ._response_functions import constant_elasticity
from ..nn.base import BaseTrainer
from ..utils import group_index_generator


class DemandResponse(BaseEstimator):
    """Mathematical model to describe the behaviour of economical demand
    through a response function.

    The response function, or response curve, specifies how demand for a
    good varies as a function of any input.

    Parameters
    ----------
    estimator : neural net estimator
        Fitted estimator. It is assumed the target variable for ``estimator``
        is some kind of demand.

    preprocessor : transformer
        Fitted transformer estimator used to preprocess data.

    column : str
        Column name for which the demand model will be based on, e.g., price
        of the product. ``column`` must be a column inside ``X`` when calling
        fit method.

    response_function : str {"linear", "constant_elasticity", "logit"},
    default = "constant_elasticity"
        Response function used to model demand.

    domain : dict, str -> array_like, default=None
        Mapping from group_id to array that specifies the domain for the
        response function for each group. If None, the domain for each group_id
        will be given by the min-max range.

    Attributes
    ----------
    estimates_ : dict, str -> tuple
        Mapping from group id to tuple containing the estimated response
        function coefficients

    domain_ : dict, str -> array
        Mapping from group id to array containing the domain for each group
    """

    def __init__(self, estimator, preprocessor, column,
                 response_function='constant_elasticity', domain=None,
                 idx_by_group=None):
        self.estimator = estimator
        self.column = column
        self.preprocessor = preprocessor
        self.response_function = response_function
        self.domain = domain
        self.idx_by_group = idx_by_group
        self._validate_estimator()
        self._select_response_function()

    def _select_response_function(self):
        if self.response_function == 'constant_elasticity':
            self._response_function = constant_elasticity
        else:
            raise ValueError(
                'The only available response function is "constant"'
            )

    def fit(self, X, y=None):
        """Fits response function on the output given by X.

        Parameters
        ----------
        X : pd.DataFrame
            pandas DataFrame containing the given ``column``.

        y : None
            Compatibility purposes.

        Returns
        -------
        self (object)
        """
        # Retrieve settings
        group_ids = self.estimator.group_ids
        max_prediction_length = self.estimator.max_prediction_length
        target = self.estimator.target

        if self.domain is None:
            self.domain_ = self._compute_domain(X)
        else:
            self._validate_domain()
            self.domain_ = self.domain

        X = X.copy()
        outputs = []
        for i in range(self._n):
            # Insert ith domain element on given ``column`` for each group on
            # the specified index
            if self.idx_by_group is None:
                sl = slice(-max_prediction_length, None)
                index_generator = group_index_generator(X, group_ids, sl)
            else:
                index_generator = self.idx_by_group.items()
            for id_, index in index_generator:
                X.loc[index, self.column] = self.domain_[id_][i]

            # Predict ``X`` after all domain values have been inserted.
            # ``y_hat_i`` contains the predictions of the ith domain element
            # for all groups
            Xt = self.preprocessor.transform(X)
            y_hat_i = self.estimator.predict(
                Xt, raw=False, inverse_transformer=self.preprocessor
            )
            y_hat_i_agg = y_hat_i.groupby(group_ids).agg(
                {target + '_pred': sum}
            )
            outputs.append(y_hat_i_agg)
        self._outputs = pd.concat(outputs, axis=1, ignore_index=True)
        self.estimates_ = self._estimates_per_group()
        return self

    def plot_response_function(self, group_id, ax=None, estimates_title=False):
        """Plots the estimated response function and the actual estimator
        outputs (which were used to fit the response function).

        Parameters
        ----------
        group_id : str
            Group to plot

        ax : matplotlib ax, default=None
            matplotlib axes to plot on

        estimates_title : bool, default=False
            If True, the estimated parameters are displayed inside the ax title
        """
        check_is_fitted(self)
        if ax is None:
            _, ax = plt.subplots()
        estimates = self.estimates_[group_id]
        x = self.domain_[group_id]
        y_estimator = self._outputs.loc[group_id].values
        y_fit = self._response_function.f(x, *estimates)
        ax.scatter(x, y_estimator, label='Estimator output')
        ax.plot(x, y_fit, label='Response function output')
        ax.legend()
        if estimates_title:
            ax.set_title(self.estimates_[group_id])

    def predict(self, x_dict):
        """Computes predictions for all groups

        Parameters
        ----------
        x_dict : dict, str -> float
            Mapping from the group_id to a number in which the response
            function will be evaluated

        Returns
        -------
        predictions : dict, str -> float
            Dictionary containing the response function evaluated at the given
            number for all groups in ``x_dict``
        """
        check_is_fitted(self)
        response_per_group = {}
        for id_, x in x_dict.items():
            estimates = self.estimates_[id_]
            response_per_group[id_] = self._response_function.f(x, *estimates)
        return response_per_group

    def elasticities(self, x_dict):
        """Computes elasticities for all groups

        Parameters
        ----------
        x_dict : dict, str -> float
            Mapping from the group_id to a number in which the elasticity
            will be evaluated

        Returns
        -------
        elasticities : dict, str -> float
            Dictionary containing the elasticities evaluated at the given
            number for all groups
        """
        check_is_fitted(self)
        elasticity_per_group = {}
        if not x_dict:
            if self.response_function != 'constant_elasticity':
                raise ValueError(
                    'Emtpy `x_dict` is only valid when response function is '
                    '"constant_elasticity".'
                )
            x_dict = dict.fromkeys(self.estimates_)
        for id_, x in x_dict.items():
            estimates = self.estimates_[id_]
            elasticity_per_group[id_] = self._response_function.elasticity(
                x, *estimates
            )
        return elasticity_per_group

    def _estimates_per_group(self):
        estimates_per_group = {}
        for id_, x in self.domain_.items():
            if id_ not in self._outputs.index:
                continue
            y = self._outputs.loc[id_].values
            estimates = self._response_function.fit(x, y)
            estimates_per_group[id_] = estimates
        return estimates_per_group

    def _compute_domain(self, X):
        self._n = 15
        agg = {self.column: (min, max)}
        min_max_per_group = X.groupby(self.estimator.group_ids).agg(agg)
        price_domain = min_max_per_group.apply(
            lambda x: np.linspace(x[0], x[1], self._n),
            axis=1
        ).to_dict()
        return price_domain

    def _validate_domain(self):
        if not isinstance(self.domain, dict):
            raise ValueError(
                'domain must be a mapping from group id (str) to an array '
                'containing the numbers in which the response function will be'
                'fitted on. Instead got {}'.format(type(self.domain))
            )
        for id_, arr in self.domain.items():
            if not isinstance(arr, (list, np.array, tuple)):
                raise ValueError(
                    'value for key {} in domain dictionary is not '
                    'array_like. Instead got {}'.format(id_, type(arr))
                )
        lens = list(map(len, self.domain.values()))
        if len(set(lens)) == 1:
            self._n = set(lens).pop()
        else:
            raise ValueError(
                'arrays in domain must all be same length'
            )

    def _validate_estimator(self):
        """Check the estimator attributes
        """
        if not isinstance(self.estimator, BaseTrainer):
            raise ValueError(
                "base_estimator must be an instance of BaseTrainer"
            )
        check_is_fitted(self.estimator)
