"""
This module includes the main metrics used in time forcasting, divided by
groups 1,2,3. All of these have the same pattern f (y_true, y_predict)
respecting the sklearn convention. To be compatible with functions related to
sklearn, the function sklearn.metrics.make_scorer must be used, for example:

>>> cross_val_score(model,
...  X_train,
...  y_train,
...  scoring=make_scorer(mean_absolute_error, greater_is_better=False),
...  cv=5,
...  n_jobs=-1)

It is important to clarify that the smaller the errors, the better the model.
While the accuracy, the greater the model, the greater_is_better parameter is
important to define.

More info:
https://scikit-learn.org/stable/modules/model_evaluation.html#implementing
-your-own-scoring-object
"""
import numpy as np


def to_numpy_params(f):
    """Decorator that allows to convert a 2 array-like params of a function to
    2 numpy params.

    Parameters
    ----------
    f : callable, f(array-like, array-like, *args, **kw_args) -> float
        function to convert of the form

    Returns
    -------
    new_f : callable, f(np.ndarray, np.ndarray, *args, **kw_args) -> float:
    """

    def new_f(y_true, y_pred, *args, **kw_args):
        """
        Parameters
        ----------
        y_true : array-like
            True values

        y_pred : array-like
            Predicted values

        Returns
        -------
        float : Metric result

        """
        v1 = np.array(y_true)
        v2 = np.array(y_pred)
        return f(v1, v2, *args, **kw_args)

    return new_f


# Scaled-dependent errors


@to_numpy_params

def mean_absolute_error(y_true, y_pred, sample_weigth=None):
    """.. math:: /(\frac{\sum^N_{t=1} |E_t|}{N}/)

    Parameters
    ----------
    y_true : array-like
        Ground truth (correct) target values.
    y_pred : array-like
        Estimated target values.
    sample_weigth : array-like
        Sample weights.

    Returns
    -------
    float: mean absolute error (lower is better)

    Reference
    ---------
    https://en.wikipedia.org/wiki/Mean_absolute_error
    """
    e = y_true - y_pred
    return np.average(np.abs(e), weights=sample_weigth)


@to_numpy_params
def mean_squared_error(y_true, y_pred, sample_weigth=None):
    """Formula /(\frac{\sum^N_{t=1} E_t^2}{N}/)

    Parameters
    ----------
        y_true (array-like): Ground truth (correct) target values.
        y_pred (array-like): Estimated target values.
        sample_weigth (array-like): Sample weights.

    Returns
    -------
    float: mean squared error (lower is better)

    Reference
    ---------
    https://en.wikipedia.org/wiki/Mean_squared_error
    """
    e = y_true - y_pred
    return np.average(e ** 2, weights=sample_weigth)


@to_numpy_params
def root_mean_squared_error(y_true, y_pred, sample_weigth=None):
    """Formula /(\sqrt{\frac{\sum^N_{t=1} E_t^2}{N}}/)

    Parameters
    ----------
    y_true : array-like
        Ground truth (correct) target values.

    y_pred : array-like
        Estimated target values.

    sample_weigth : array-like
        Sample weights.

    Returns
    -------
    float : root mean squared error (lower is better)

    Reference
    ---------
    https://en.wikipedia.org/wiki/Root-mean-square_deviation
    """
    mse = mean_squared_error(y_true, y_pred, sample_weigth)
    return np.sqrt(mse)


# Percentage errors

@to_numpy_params
def mean_absolute_percentage_error(y_true, y_pred, sample_weigth=None):
    """Formula /(\frac{\sum^N_{t=1} |\frac{E_t}{Y_t}|}{N}}/)

     Parameters
    ----------
    y_true : array-like
        Ground truth (correct) target values.

    y_pred : array-like
        Estimated target values.

    Returns
    -------
    float: mean absolute percentage deviation (lower is better)

    Reference
    ---------
    https://en.wikipedia.org/wiki/Mean_absolute_percentage_error
    """
    e = y_true - y_pred
    ab = np.abs(e / y_true)
    return np.average(ab, weights=sample_weigth) * 100


@to_numpy_params
def mean_absolute_percentage_deviation(y_true, y_pred):
    """Formula .. math:: /(\frac{\sum^N_{t=1} |\frac{E_t}{Y_t}|}{N}}/)

    Parameters
    ----------
    y_true : array-like
        Ground truth (correct) target values.

    y_pred : array-like
        Estimated target values.

    Returns
    -------
    float: mean absolute percentage deviation (lower is better)

    Reference
    ---------
    https://en.wikipedia.org/wiki/Mean_absolute_percentage_error
    """
    e = y_true - y_pred
    return np.sum(e) / np.sum(y_true)


@to_numpy_params
def symmetric_mean_absolute_percentage_error(y_true, y_pred, f_den=1):
    """Formula .. math:: /(\sum^N_{t=1} \frac{|E_t - Y_t|}{|E_t| + |Y_t|}}/)

    Parameters
    ----------
    y_true : array-like
        Ground truth (correct) target values.

    y_pred : array-like
        Estimated target values.

    f_den : float
        denominator factor.

    Returns
    -------
    float: symmetric mean absolute percentage error (lower is better)

    Reference
    ---------
    https://en.wikipedia.org/wiki/Symmetric_mean_absolute_percentage_error
    """
    e = y_true - y_pred
    num = np.abs(e)
    den = (np.abs(y_true) + np.abs(y_pred)) * f_den
    smape = np.sum(num / den) / len(y_true)
    return smape


# Scaled errors

@to_numpy_params
def mean_absolute_scaled_error_mod(y_true, y_pred):
    e = np.average(np.abs(y_true - y_pred))
    mae = np.average(np.abs(y_true[1:] - y_true[:-1]))
    return e / mae


# Accuracy

def accuracy(y_true, y_pred, error=mean_absolute_percentage_error,
             sample_weigth=None):
    """Calculate 1 - error.

    Parameters
    ----------
    y_true : array-like
        Ground truth (correct) target values.
    y_pred : array-like
        Estimated target values.
    error : callable, f(y_true, y_pred, sample_weigth) => float)
        Selected error metric
    sample_weigth : array-like
        Sample weights.

    Returns
    -------
    float
    """
    error_ = error(y_true, y_pred, sample_weigth)
    return 1 - error_


@to_numpy_params
def regression_report(y_true, y_pred):
    """Prints regression report

    Parameters
    ----------
    y_true : array-like
        Ground truth (correct) target values.
    y_pred : array-like
        Estimated target values.

    Returns
    -------
    regression_report : str
    """
    # scaled-dependent errors
    mae = mean_absolute_error(y_true, y_pred)
    rmse = root_mean_squared_error(y_true, y_pred)

    # percentage errors
    mape = mean_absolute_percentage_error(y_true, y_pred)

    # Scaled errors
    mase = mean_absolute_scaled_error_mod(y_true, y_pred)

    verbose = """Scaled-dependent errors\n
                Mean Absolute Error (mae): {}\n
                Root Mean Squared Error (rmse): {}\n\n
                Percentage Errors\n
                Mean Absolute Percentage Error (mape): {}\n
                Scaled Errors
                Mean Absolute Scaled Error (mase): {}
                """.format(mae, rmse, mape, mase)
    return verbose
