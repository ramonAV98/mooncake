from abc import ABCMeta, abstractmethod
from copy import deepcopy

import numpy as np
import pandas as pd
import skorch.dataset
import torch
from pytorch_forecasting.metrics import MultiHorizonMetric, RMSE
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted
from skorch import NeuralNet
from skorch.callbacks import Callback, LRScheduler
from skorch.helper import predefined_split
from skorch.utils import params_for
from torch.utils.data.dataloader import default_collate

from .datasets import SliceDataset
from ..utils import add_prefix
from ..utils import get_init_params
from ..utils import safe_math_eval
from ..utils import undo_sliding_window


class BaseModule(torch.nn.Module, metaclass=ABCMeta):
    """Base class for pytorch neural network modules.

    .. note::
        This class should not be used directly. Use derived classes instead.
    """

    @classmethod
    @abstractmethod
    def from_dataset(cls, dataset, **kwargs):
        """Create model from dataset

        Parameters
        ----------
        dataset : timeseries dataset

        Returns
        -------
        BaseModule: Model that can be trained
        """
        pass


class BaseTrainer(BaseEstimator, metaclass=ABCMeta):
    """Base class for trainers

    In addition to the parameters listed below, there are parameters
    with specific prefixes that are handled separately. To illustrate
    this, here is an example:

    >>> net = NeuralNet(
    ...    [...],
    ...    optimizer=torch.optim.SGD,
    ...    optimizer__momentum=0.95,
    ...)

    This way, when ``optimizer`` is initialized, :class:`.NeuralNet`
    will take care of setting the ``momentum`` parameter to 0.95.
    (Note that the double underscore notation in
    ``optimizer__momentum`` means that the parameter ``momentum``
    should be set on the object ``optimizer``. This is the same
    semantic as used by sklearn.). Supported prefixes include:
    ['module',
    'iterator_train',
    'iterator_valid',
    'optimizer',
    'criterion',
    'callbacks',
    'dataset']

    .. note::
        This class should not be used directly. Use derived classes instead.
    
    Parameters
    ----------
    module : class
        Neural network class that inherits from :class:`BaseModule`

    group_ids : list of str
        List of column names identifying a time series. This means that the
        group_ids identify a sample together with the time_idx. If you have
        only one time series, set this to the name of column that is constant.

    time_idx : str
        Time index column. This column is used to determine the sequence of
        samples.

    target : str
        Target column. Column containing the values to be predicted.

    max_prediction_length : int
        Maximum prediction/decoder length (choose this not too short as it can
        help convergence)

    max_encoder_length : int
        Maximum length to encode. This is the maximum history length used by
        the time series dataset.

    time_varying_known_reals : list of str
        List of continuous variables that change over time and are known in the
        future (e.g. price of a product, but not demand of a product). If None,
        every numeric column excluding ``target`` is used.

    time_varying_unknown_reals : list of str
        List of continuous variables that change over time and are not known in
        the future. You might want to include your ``target`` here. If None,
        only ``target`` is used.

    static_categoricals : list of str
        List of categorical variables that do not change over time (also known
        as `time independent variables`). You might want to include your
        ``group_ids`` here for the learning algorithm to distinguish between
        different time series. If None, only ``group_ids`` is used.
      
    dataset : class
        The dataset is necessary for the incoming data to work with
        pytorch :class:`DataLoader`. Must inherit from
        :class:`pytorch.utils.data.Dataset`.

    min_encoder_length : int, default=None
        Minimum allowed length to encode. If None, defaults to
        ``max_encoder_length``.

    collate_fn : callable, default=None
        Collate function to be passed to torch
        :class:`~torch.data.utils.Dataloader` instance

    criterion : class, default=None
        The uninitialized criterion (loss) used to optimize the
        module. If None, the root mean squared error (:class:`RMSE`) is used.

    optimizer : class, default=None
        The uninitialized optimizer (update rule) used to optimize the
        module. If None, :class:`Adam` optimizer is used.

    lr : float, default=1e-5
        Learning rate passed to the optimizer.

    max_epochs : int, default=10
        The number of epochs to train for each :meth:`fit` call. Note that you
        may keyboard-interrupt training at any time.

    batch_size : int, default=64
        Mini-batch size. If ``batch_size`` is -1, a single batch with all the
        data will be used during training and validation.

    callbacks: None, “disable”, or list of Callback instances, default=None
        Which callbacks to enable.

        - If callbacks=None, only use default callbacks which include:
            - `epoch_timer`: measures the duration of each epoch
            - `train_loss`: computes average of train batch losses
            - `valid_loss`: computes average of valid batch losses
            - `print_log`:  prints all of the above in nice format

        - If callbacks="disable":
            disable all callbacks, i.e. do not run any of the callbacks.

        - If callbacks is a list of callbacks:
            use those callbacks in addition to the default callbacks. Each
            callback should be an instance of skorch :class:`.Callback`.

        Alternatively, a tuple ``(name, callback)`` can be passed, where
        ``name`` should be unique. Callbacks may or may not be instantiated.
        The callback name can be used to set parameters on specific
        callbacks (e.g., for the callback with name ``'print_log'``, use
        ``net.set_params(callbacks__print_log__keys_ignored=['epoch',
        'train_loss'])``).

    verbose : int, default=1
        This parameter controls how much print output is generated by
        the net and its callbacks. By setting this value to 0, e.g. the
        summary scores at the end of each epoch are no longer printed.
        This can be useful when running a hyperparameters search. The
        summary scores are always logged in the history attribute,
        regardless of the verbose setting.

    device : str, torch.device, default='cpu'
        The compute device to be used. If set to 'cuda', data in torch
        tensors will be pushed to cuda tensors before being sent to the
        module. If set to None, then all compute devices will be left
        unmodified.

    kwargs : dict
       Extra prefixed parameters (see list of supported prefixes above).

    Attributes
    ----------
    callbacks_ : list of (name, obj) tuples
        Callbacks used during training

    dataset_params_ : dict
        Training dataset parameters

    net_ : skorch neural net
        Fitted neural net
    """

    prefixes = [
        'module',
        'iterator_train',
        'iterator_valid',
        'optimizer',
        'criterion',
        'callbacks',
        'dataset'
    ]

    @abstractmethod
    def __init__(self, module, dataset, group_ids, time_idx, target,
                 max_prediction_length, max_encoder_length,
                 time_varying_known_reals, time_varying_unknown_reals,
                 static_categoricals, cv_split=None, min_encoder_length=None,
                 collate_fn=None, criterion=None, optimizer=None, lr=1e-5,
                 max_epochs=10, batch_size=64, callbacks=None, verbose=1,
                 device='cpu', **kwargs):
        self.module = module
        self.dataset = dataset
        self.group_ids = group_ids
        self.time_idx = time_idx
        self.target = target
        self.max_prediction_length = max_prediction_length
        self.max_encoder_length = max_encoder_length
        self.time_varying_known_reals = time_varying_known_reals
        self.time_varying_unknown_reals = time_varying_unknown_reals
        self.static_categoricals = static_categoricals
        self.cv_split = cv_split
        self.min_encoder_length = min_encoder_length
        self.collate_fn = default_collate if collate_fn is None else collate_fn
        self.criterion = RMSE if criterion is None else criterion
        self.optimizer = torch.optim.Adam if optimizer is None else optimizer
        self.lr = lr
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.callbacks = callbacks
        self.verbose = verbose
        self.device = device
        self._check_params()
        self.kwargs = kwargs

    def fit(self, X, y=None):
        """Initialize and fit the neural net estimator.

        If the neural net was already initialized, by calling fit, the
        neural net will be re-initialized

        Parameters
        ----------
        X : pd.DataFrame
            The input data

        y : None
            This parameter only exists for sklearn compatibility and must
            be left in None

        Returns
        -------
        self : trained neural net
        """
        train_dataset = self.get_dataset(X)
        self.dataset_params_ = train_dataset.get_parameters()
        self.callbacks_ = self._init_callbacks(train_dataset)
        self.net_ = self._fit_skorch(train_dataset)
        return self

    def predict(self, X, raw=True, inverse_transformer=None):
        """Calls forward pass of the net for the given input values.

        Parameters
        ----------
        X : pd.DataFrame
            Input values

        raw : bool
            - If True, raw predictions in a numpy array are returned
            - If False, an interpretable version of predictions is returned.

        inverse_transformer : transformer
            If not None, predictions are inverse transformed using the
            :meth:`inverse_transform` method of this object. Only available
            when ``raw`` is False.

        Returns
        -------
        y_hat : np.array or pd.DataFrame
            If ``raw`` is False, an interpretable version of predictions using
            a pandas DataFrame is returned, otherwise, raw predictions in a
            numpy array are returned.

        Notes
        -----
        Predictions (interpretable or not) are sorted by group_ids and date.
        So, for instance, it is possible that the order of the
        non-interpretable predictions does not match the order of the given
        ``X``. Same is true for interpretable predictions; however, in such
        case, a date column is added, so no chance of misinterpretation is
        possible.
        """
        check_is_fitted(self)
        dataset = self.get_dataset(
            X, params=self.dataset_params_, predict_mode=True)
        raw_output = self.net_.predict(dataset)
        if raw:
            return raw_output
        if not hasattr(dataset, 'groups_sizes'):
            raise ValueError(
                "torch dataset does not have group_sizes attribute"
            )
        # Since raw predictions come in a sliding windows format, they must be
        # split in their corresponding groups to undo the sliding window one
        # group at a time. For the following block of code, ``group_sizes``
        # is a mapping from group_ids to their size and ``sliding_output`` is a
        # list of ndarrays where the i element corresponds to the prediction of
        # the i group in a sliding windows format.
        groups_sizes = dataset.groups_sizes
        groups_sizes_values = list(groups_sizes.values())
        sliding_output = np.split(raw_output, np.cumsum(groups_sizes_values))
        non_sliding_output = [
            undo_sliding_window(x) for x in sliding_output if x.size != 0
        ]

        # The next steps are devoted to insert the values inside
        # ``non_sliding_output``, which contains the predictions after the
        # sliding windows format has been removed, into an interpretable pandas
        # DataFrame

        # Keep only the groups whose prediction was possible
        groups_to_keep = list(groups_sizes)
        X = X.set_index(self.group_ids).loc[groups_to_keep].reset_index()

        # Add predictions to pandas DataFrame ``y_hat``
        pk = self.group_ids + [self.time_idx]  # primary key
        X = X.sort_values(pk).reset_index(drop=True)
        y_hat = X.copy()
        g = y_hat.groupby(self.group_ids, sort=False)
        pred_idx = []
        for i, (group, size) in enumerate(groups_sizes.items()):
            n_pred = self.max_prediction_length + size - 1
            tail_idx = g.get_group(group).tail(n_pred).index.tolist()
            pred_idx.extend(tail_idx)
            y_hat.loc[tail_idx, self.target] = non_sliding_output[i]
        # Only keep prediction info
        y_hat = y_hat.loc[pred_idx]

        # Inverse transform
        if inverse_transformer is not None:
            y_hat = inverse_transformer.inverse_transform(y_hat)
            X = inverse_transformer.inverse_transform(X)

        # Recover true values and return only primary key with target column
        columns_to_keep = pk + [self.target]
        y_hat = pd.merge(
            y_hat[columns_to_keep],
            X[columns_to_keep],
            on=pk,
            suffixes=('_pred', '_true')
        )
        return y_hat

    def get_dataset(self, X, params=None, sliceable=False, **kwargs):
        """Constructs torch dataset using the input data ``X``

        Parameters
        ----------
        X : pd.DataFrame
            Input data

        params : dict, default=None
            If given, generates torch dataset using this parameters. Otherwise,
            the parameters are obtained from the object (self) attributes.

        sliceable : bool, default=False
            If True, the sliceable version of the dataset is returned

        **kwargs : key-word arguments
            Additional parameters passed to dataset class. If given,
            kwargs will override values given to ``params``.

        Returns
        -------
        dataset: torch dataset
            The initialized dataset.
        """
        # Return ``X`` if already is a dataset
        if isinstance(X, torch.utils.data.Dataset):
            return X
        if params is not None:
            dataset = self.dataset.from_parameters(params, X, **kwargs)
        else:
            dataset_params = self.get_kwargs_for('dataset')
            dataset_params.update(kwargs)
            dataset = self.dataset(X, **dataset_params)
        if sliceable:
            return SliceDataset(dataset)
        return dataset

    def get_kwargs_for(self, name, prefix=False):
        """Collects __init__ kwargs for an attribute.

        Attributes must be type class and could be, for instance, pytorch
        modules, criteria or data loaders.

        The returned kwargs are obtained by inspecting the __init__ method
        from the passed attribute (e.g., module.__init__()) and from prefixed
        kwargs (double underscore notation, e.g., 'module__something') passed
        at __init__.

        Parameters
        ----------
        name : str
            The name of the attribute whose arguments should be
            returned. E.g. for the module, it should be ``'module'``.

        prefix : bool
            If True, keys will contain ``name`` with double underscore as
            prefix.

        Returns
        -------
        kwargs : dict
        """
        # Collect kwargs from attribute's init
        if hasattr(self, name):
            init_params = get_init_params(getattr(self, name))
            kwargs = {k: v for k, v in vars(self).items() if k in init_params}
        else:
            kwargs = {}
        # Collect prefixed kwargs
        prefixed_kwargs = params_for(prefix=name, kwargs=self.kwargs)
        # Join both
        kwargs.update(prefixed_kwargs)

        if prefix:
            kwargs = add_prefix(kwargs, name)
        return kwargs

    def _get_train_split(self, train_dataset):
        """Obtains the validation technique used during training

        Parameters
        ----------
        train_dataset : torch Dataset
            Training torch dataset

        Returns
        -------
        iterable
        """
        if isinstance(self.cv_split, (pd.DataFrame, torch.utils.data.Dataset)):
            valid_dataset = self.get_dataset(
                X=self.cv_split,
                params=self.dataset_params_
            )
            return predefined_split(valid_dataset)
        elif isinstance(self.cv_split, int):
            if not hasattr(train_dataset, 'timeseries_cv'):
                obj_name = type(train_dataset).__name__
                raise ValueError(
                    'Dataset {} does not have a '
                    'timeseries_cv method'.format(obj_name)
                )
            # ``cv`` is list containing train, validation splits (tuples)
            cv = train_dataset.timeseries_cv(self.cv_split)
            return skorch.dataset.CVSplit(cv=cv)
        elif isinstance(self.cv_split, skorch.dataset.CVSplit):
            return self.cv_split
        else:
            obj_name = type(self.cv_split).__name__
            raise ValueError(
                'Passed `cv_split` {} not supported'.format(obj_name)
            )

    def _fit_skorch(self, train_dataset):
        """Initializes and fits skorch :class: NeuralNet.

        Parameters
        ----------
        train_dataset : torch dataset
            Training dataset

        Returns
        -------
        skorch NeuralNet fitted
        """
        module = self._init_module(train_dataset)
        if self.cv_split is not None:
            train_split = self._get_train_split(train_dataset)
        else:
            train_split = None
        return NeuralNet(
            module=module,
            criterion=self.criterion,
            optimizer=self.optimizer,
            lr=self.lr,
            max_epochs=self.max_epochs,
            batch_size=self.batch_size,
            callbacks=self.callbacks_,
            verbose=self.verbose,
            device=self.device,
            train_split=train_split,
            iterator_train__shuffle=True,
            iterator_train__collate_fn=self.collate_fn,
            iterator_valid__collate_fn=self.collate_fn,
            **self.kwargs
        ).fit(X=train_dataset)

    def _init_module(self, train_dataset):
        """Instantiates pytorch module using object (self) attributes and
        training dataset

        Parameters
        ----------
        train_dataset : torch dataset
            Training dataset. Used as input in ``from_dataset`` method

        Returns
        -------
        module : torch neural net object
            Instantiated neural net
        """
        module_kwargs = self.get_kwargs_for('module')
        module = self.module.from_dataset(train_dataset, **module_kwargs)
        return module.to(self.device)

    def _init_callbacks(self, train_dataset):
        """Evaluates string expressions in callbacks.

        Returns
        -------
        callbacks : list of callbacks
        """
        # Do not alter original callbacks
        callbacks = deepcopy(self.callbacks)

        if self.callbacks is not None:
            # ``iterations`` is the number of batches on each epoch
            iterations = int(np.ceil(len(train_dataset) / self.batch_size))
            for name, obj in callbacks:
                if isinstance(obj, LRScheduler):
                    for key, value in vars(obj).items():
                        if isinstance(value, str):
                            if 'iterations' in value:
                                value = value.replace(
                                    'iterations', str(iterations)
                                )
                                vars(obj)[key] = safe_math_eval(value)
                            elif hasattr(self, value):
                                vars(obj)[key] = getattr(self, value)
        return callbacks

    def _check_params(self):
        """Collection of parameters validations.
        """
        if not issubclass(self.optimizer, torch.optim.Optimizer):
            raise ValueError(
                'optimizer must be a subclass of torch.optim.Optimizer. '
                'Instead got {}'.format(self.optimizer)
            )
        if not issubclass(self.criterion, torch.nn.Module):
            raise ValueError(
                'criterion must be a subclass of torch.nn.Module. Instead got '
                '{}'.format(self.optimizer)
            )
        if self.callbacks is not None:
            for i, tup in enumerate(self.callbacks):
                if not isinstance(tup, tuple):
                    raise ValueError(
                        'callbacks must contain a list of (name, callback) '
                        'tuples. Instead on index {} got {}'.format(i, tup)
                    )
                name, obj = tup
                if not isinstance(obj, Callback):
                    obj_name = obj.__class__.__name__
                    raise ValueError(
                        'callback with name {} is not a skorch Callback '
                        'object. Instead got {}'.format(name, obj_name)
                    )
        if not isinstance(self.criterion(), MultiHorizonMetric):
            raise ValueError(
                'criterion must be a pytorch_forecasting metric. Select one '
                'from module pytorch_forecasting.metrics'
            )
