"""
Helper functions and classes for users.

They should not be used in skorch directly.
"""

import pickle

import pandas as pd

from sklearn.base import BaseEstimator
from .utils import column_selector
from skorch.callbacks import LRScheduler, GradientNormClipping, EarlyStopping

__all__ = [
    'common_callbacks',
    'column_selector',
    'ModelLog',
    'get_unique_categories'
]


def common_callbacks(lr_scheduler=None, early_stopping=False,
                     gradient_clipping=False, patience=5):
    """Helper for constructing callbacks

    This function only aims to facilitate the creation of three commonly used
    callbacks: LRScheduler, EarlyStopping and GradientNormClipping.
    Please refer to https://skorch.readthedocs.io/en/stable/callbacks.html#
    for a full description of every available callback.

    Parameters
    ----------
    lr_scheduler : dict
        Dictionary containing all the configuration for the chosen learning
        rate scheduler. This includes the `policy` (torch lr_scheduler class
        name, i.e., 'OneCycleLR'), `step_every` (when the scheduler takes a
        step, can be either 'epoch' or 'batch) and every parameter
        for the torch scheduler to be instantiated correctly.

    early_stopping : bool, default=False
        Callback for stopping training when scores don’t improve.
        Thi callback stops training early if a specified the valid loss did not
        improve in patience number of epochs.

    gradient_clipping : bool, default=False
        Clips gradient norm of a module’s parameters.

    patience : int
        Number of epochs to wait for improvement of the monitor value until the
        training process is stopped. This parameter is ignored if
        ``early_stopping`` is False.

    Returns
    -------
    list of tuples
        List of tuples of the form (name, obj)
    """
    callbacks = []
    if lr_scheduler is not None:
        name = 'lr_scheduler'
        obj = LRScheduler(**lr_scheduler)
        callbacks.append((name, obj))
    if early_stopping:
        name = 'early_stopping'
        obj = EarlyStopping(patience=patience)
        callbacks.append((name, obj))
    if gradient_clipping :
        name = 'gradient_clipping'
        obj = GradientNormClipping(1)
        callbacks.append((name, obj))
    return callbacks


class ModelLog:
    """Helper that facilitates saving trained models, datasource params,
    feature engineering, and training data with pickle.

    This class help to read/write python trained models objects.

    Parameters
    ----------
    datasource_params : dict
        Dictionary of the datasource params used to obtain the training data

    feature_engineering: FeatureEngineering
        Feature engineering used to obtain the training data

    training_data: pd.DataFrame
        DataFrame used to train the model

    model: BaseTrainer
        Trained model
    """

    def __init__(self, datasource_params, feature_engineering,
                 training_data, model):
        self.__valid_parameters__(
            datasource_params, feature_engineering, training_data, model
        )
        self.datasource_params = datasource_params
        self.feature_engineering = feature_engineering
        self.training_data = training_data
        self.model = model

    def run(self):
        """TODO: Repeat the process
        """
        pass

    @staticmethod
    def read_model_log(url_path):
        """Read a model log from a file

        Parameters
        ----------
        url_path : str
            File path

        Return
        ------
        ModelLog: the model log

        """
        with open(url_path, 'rb') as f:
            model_log = pickle.load(f)
        return model_log

    def save_model_log(self, url_path):
        """Save the ModelLog to a File

        Parameters
        ----------
        url_path : str
            File path
        """
        with open(url_path, 'wb') as f:
            pickle.dump(self, f)

    def __valid_parameters__(self, datasource_params, feature_engineering,
                             training_data, model):
        if not isinstance(datasource_params, dict):
            raise TypeError("datasource param must be a dict")
        if len(datasource_params) == 0:
            raise ValueError("datasource param must have values")
        # if not isinstance(feature_engineering,  FeatureEngineering):
        #    raise TypeError("feature_engineering must be a FeatureEngineering")
        if not isinstance(training_data, pd.DataFrame):
            raise TypeError("training_data must be pd.DataFrame")
        if len(training_data) == 0:
            raise ValueError("training_data doesnt be empty")
        if not isinstance(model, BaseEstimator):
            raise TypeError("model must be BaseTrainer")
        # if not is_trained(model):
        #   raise ValueError("model must be a trained model")

    def __dict__(self):
        return {"datasource_params": self.datasource_params,
                "feature_engineering": self.feature_engineering,
                "training_data": self.training_data,
                "model": self.model}

    def __eq__(self, other):
        return self.datasource_params == other.datasource_params and \
               self.feature_engineering == other.feature_engineering and \
               self.training_data == other.training_data and \
               self.model == other.model

    def __hash__(self):
        return hash(self.datasource_params) * \
               hash(self.feature_engineering) * \
               hash(self.training_data) * \
               hash(self.model)

    def __repr__(self):
        return """ModelLog(
                        datasource_params: 
                        %s
                        feature_engineering: 
                        %s
                        training_data: 
                        %s
                        model:
                        %s
                      )\n""" % (self.datasource_params,
                                self.feature_engineering.head(),
                                self.training_data.head(),
                                self.model)

    def __str__(self):
        return self.__repr__()


def get_unique_categories(df, pattern_exclude=None):
    """Obtains the unique values from each object dtype column.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe for which the unique categories will be obtained

    pattern_exclude : str or list, default=None
        A selection of columns to exclude.

        - If None, column selection will not be performed based on this param.
        - If list, the elements will be joined using the regex '|' operator.
            Columns matching the resulting regex will be omitted from
            selection.
        - If str, the pattern is used as regex and columns matching will be
            omitted from selection.

    Returns
    -------
    uniques : list of lists
        The ith element of the returned list is another list containing the
        unique elements of the ith column in order of appearance.
    """

    fn = column_selector(
        dtype_include=['object'], pattern_exclude=pattern_exclude
    )
    columns = fn(df)
    uniques = []
    for c in columns:
        if hasattr(df[c].dtypes, 'categories'):
            uniques.append(df[c].dtypes.categories.tolist())
        else:
            uniques.append(df[c].unique().tolist())
    return uniques
