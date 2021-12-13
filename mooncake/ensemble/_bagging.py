import itertools
import numbers

import numpy as np
import pandas as pd
from joblib import Parallel
from sklearn.base import BaseEstimator
from sklearn.base import clone
from sklearn.ensemble._base import _set_random_states, _partition_estimators
from sklearn.utils import check_random_state
from sklearn.utils.fixes import delayed
from sklearn.utils.random import sample_without_replacement

from ..nn.base import BaseTrainer

MAX_INT = np.iinfo(np.int32).max


def _generate_bagging_indices(random_state, bootstrap, n_population,
                              n_samples):
    """Draw randomly sampled indices."""
    # Get valid random state
    random_state = check_random_state(random_state)
    # Draw sample indices
    if bootstrap:
        indices = random_state.randint(0, n_population, n_samples)
    else:
        indices = sample_without_replacement(
            n_population, n_samples, random_state=random_state
        )
    return indices


def _parallel_build_estimators(n_estimators, ensemble, X, seeds):
    """Private function used to build a batch of estimators within a job."""
    # Retrieve settings
    n_samples = len(X)
    max_samples = ensemble._max_samples
    bootstrap = ensemble.bootstrap

    # Build estimators
    estimators = []
    for i in range(n_estimators):
        random_state = seeds[i]
        estimator = ensemble._make_estimator(random_state)
        # Sample indices
        indices = _generate_bagging_indices(
            random_state, bootstrap, n_samples, max_samples
        )
        # Fit current estimator
        estimator.fit((X[indices]))
        estimators.append(estimator)
    return estimators


def _parallel_predict_regression(estimators, X, raw, inverse_transformer):
    """Private function used to compute predictions within a job."""
    return [estimator.predict(X, raw, inverse_transformer)
            for estimator in estimators]


class Bagging(BaseEstimator):
    """Bagging regressor

    A Bagging regressor is an ensemble meta-estimator that fits base
    regressors each on random subsets of the original dataset and then
    aggregate their individual predictions by averaging to form a final
    prediction. Such a meta-estimator can typically be used as a way to reduce
    the variance of a black-box  estimator (e.g., a neural network), by
    introducing randomization into its  construction procedure and then making
    an ensemble out of it.

    Parameters
    ----------
    base_estimator : object
        The base estimator from which the ensemble is built.

    n_estimators : int, default=10
        The number of estimators in the ensemble.

    max_samples : int or float, default=1.0
        The number of samples to draw from X to train each base estimator (with
        replacement by default, see `bootstrap` for more details).

        - If int, then draw `max_samples` samples.
        - If float, then draw `max_samples * X.shape[0]` samples.

    bootstrap : bool, default=True
        Whether samples are drawn with replacement. If False, sampling
        without replacement is performed.

    n_jobs : int (default=None)
        Number of workers requested. Passing n_jobs=-1 means requesting all
        available workers for instance matching the number of CPU cores on the
        worker host(s).

    random_state : int default=None
        Controls the random resampling of the original dataset
    """

    def __init__(self, base_estimator, n_estimators=10, max_samples=1.0,
                 bootstrap=True, estimator_params=tuple(), n_jobs=None,
                 random_state=None, verbose=0):
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.estimator_params = estimator_params
        self.bootstrap = bootstrap
        self.max_samples = max_samples
        self.estimators = []
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose
        self._validate_estimator()

    def _validate_estimator(self):
        """Check the base_estimator and the n_estimator attributes.
        """
        if not isinstance(self.n_estimators, int):
            raise ValueError(
                "n_estimators must be an integer, got {0}.".format(
                    type(self.n_estimators)
                )
            )
        if self.n_estimators <= 0:
            raise ValueError(
                "n_estimators must be greater than zero, got {0}.".format(
                    self.n_estimators
                )
            )
        if not isinstance(self.base_estimator, BaseTrainer):
            raise ValueError(
                "base_estimator must be an instance of BaseTrainer"
            )

    def _make_estimator(self, random_state=None):
        """Make and configure a copy of the `base_estimator` attribute.
        Warning: This method should be used to properly instantiate new
        sub-estimators.

        Warning: This method should be used to properly instantiate new
        sub-estimators.
        """
        estimator = clone(self.base_estimator)
        estimator.set_params(
            **{p: getattr(self, p) for p in self.estimator_params}
        )
        if random_state is not None:
            _set_random_states(estimator, random_state)
        return estimator

    def _parallel_args(self):
        return {}

    def __len__(self):
        """Return the number of estimators in the ensemble."""
        return len(self.estimators)

    def __getitem__(self, index):
        """Return the index'th estimator in the ensemble."""
        return self.estimators[index]

    def __iter__(self):
        """Return iterator over estimators in the ensemble."""
        return iter(self.estimators)

    def fit(self, X, y=None):
        """Build a Bagging ensemble of estimators from the training set (X, y).

        Parameters
        ----------
        X : pd.DataFrame
            The training input samples

        y : None.
            This parameters is for sklearn compatibility purposes and
            should always be left in None. The info about the target values is
            already given in the base_estimator object through the dataset
            attributes.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        random_state = check_random_state(self.random_state)

        # Create slice dataset
        X = self.base_estimator.get_dataset(X, sliceable=True)

        # Validate max_samples
        max_samples = self.max_samples
        if not isinstance(max_samples, numbers.Integral):
            max_samples = int(max_samples * len(X))
        if not (0 < max_samples <= len(X)):
            raise ValueError("max_samples must be in (0, n_samples]")
        # Store validated integer row sampling value
        self._max_samples = max_samples

        # Seeds
        seeds = random_state.randint(MAX_INT, size=self.n_estimators)
        self._seeds = seeds

        # Parallel loop
        n_jobs, n_estimators_per_job, starts = _partition_estimators(
            self.n_estimators, self.n_jobs
        )
        all_results = Parallel(n_jobs=n_jobs, verbose=self.verbose)(
            delayed(_parallel_build_estimators)(
                n_estimators_per_job[i],
                self,
                X,
                seeds[starts[i]: starts[i + 1]],
            )
            for i in range(n_jobs)
        )
        # ``all_results`` is a list of lists. The outer len, len(all_results),
        # is equal to n_jobs and each sublist i has len equal to
        # n_estimators_per_job[i]. So, we proceed to flatten ``all_results``
        all_results = list(itertools.chain.from_iterable(all_results))
        self.estimators = all_results
        return self

    def predict(self, X, raw=True, inverse_transformer=None):
        """Predict regression target for X.

        The predicted regression target of an input sample is computed as the
        mean predicted regression targets of the estimators in the ensemble.

        Parameters
        ----------
        X : pd.DataFrame
            Input values

        raw : bool
            - If True, raw predictions in a numpy array are returned
            - If False, an interpretable version of predictions is returned.

        preprocessor : transformer
            If not None, predictions are inverse transformed using the
            inverse_transform method of this object. Only available
            when ``raw`` is False.

        Returns
        -------
        y_hat : np.array or pd.DataFrame
            - If ``raw`` equals True:
                an interpretable version of predictions using a pandas
                DataFrame is returned
            - If False:
                raw predictions in a numpy array are returned

        """
        # Parallel loop
        n_jobs, n_estimators, starts = _partition_estimators(self.n_estimators,
                                                             self.n_jobs)
        all_y_hat = Parallel(n_jobs=n_jobs, )(
            delayed(_parallel_predict_regression)(
                self.estimators[starts[i]: starts[i + 1]],
                X,
                raw,
                inverse_transformer
            )
            for i in range(n_jobs))
        all_y_hat = list(itertools.chain.from_iterable(all_y_hat))

        if raw:
            return sum(all_y_hat) / self.n_estimators
        pk = self.base_estimator.group_ids + [self.base_estimator.time_idx]
        return pd.concat(all_y_hat).groupby(pk).mean().reset_index()
