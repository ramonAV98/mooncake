"""
Module for pytorch_forecasting models
"""

from pytorch_forecasting import TemporalFusionTransformer as TFT

from .base import BaseTrainer
from .datasets import PyTorchForecastingDataset


def output_class(net=None, **res):
    """Auxiliary for pytorch-forecasting modules output

    pytorch-forecasting require an output_class method that gets called at the
    end of every forward pass and deals with all the output information from
    the model. Since skorch only requires the actual prediction to compute the
    loss, that is the only thing we extract.

    Notes
    -----
    Notice this is not actually a class. The name follows from the private
    attribute ``_output_class`` (which is where this function is assigned to)
    in all pytorch_forecasting models.

    Parameters
    ----------
    net : pytorch-forecasting model
        Compatability purposes (equivalent to self)

    **res : dict
        Dictionary containing info about the results

    Returns
    -------
    predictions : torch.tensor
    """
    return res['prediction'].squeeze(-1)


def output_transformer(module, out=None):
    """Auxiliary for pytorch-forecasting modules output

    pytorch-forecasting modules require a pickable callable that takes network
    output and transforms it to prediction space. Since for our purpose,
    predictions already are in the prediction space (we inverse transform
    predictions after training using sklearn), we leave them untouched.

    Parameters
    ----------
    module : pytorch-forecasting model
        Compatability purposes (equivalent to self)

    out : dict
        Dictionary containing info about the results

    Returns
    -------
    predictions : torch.tensor
    """
    if isinstance(module, dict):
        return module['prediction']
    return out['prediction']


class PyForecastTrainer(BaseTrainer):
    """Base class for pytorch_forecasting models that collects common
    methods between them.

    .. note::
        This class should not be used directly. Use derived classes instead.
    """

    def __init__(self, module, group_ids, time_idx, target,
                 max_prediction_length, max_encoder_length,
                 time_varying_known_reals, time_varying_unknown_reals,
                 static_categoricals, cv_split=None, min_encoder_length=None,
                 collate_fn=None, criterion=None, optimizer=None, lr=1e-5,
                 max_epochs=10, batch_size=64, callbacks=None, verbose=1,
                 device='cpu', **kwargs):
        super().__init__(
            module=module,
            dataset=PyTorchForecastingDataset,
            group_ids=group_ids,
            time_idx=time_idx,
            target=target,
            max_prediction_length=max_prediction_length,
            max_encoder_length=max_encoder_length,
            time_varying_known_reals=time_varying_known_reals,
            time_varying_unknown_reals=time_varying_unknown_reals,
            static_categoricals=static_categoricals,
            cv_split=cv_split,
            min_encoder_length=min_encoder_length,
            collate_fn=collate_fn,
            criterion=criterion,
            optimizer=optimizer,
            lr=lr,
            max_epochs=max_epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=verbose,
            device=device,
            **kwargs
        )
        self.loss = self.criterion()

    def interpret_output(self, X):
        """Provides a visual interpretation of the models output that includes
        feature ranking and attention across time index.

        Parameters
        ----------
        X : pd.DataFrame
            Dataframe to predict / Dataframe whose prediction wants to be
            interpreted

        Returns
        -------
        None
        """
        raw_predictions = self._raw_predict(X)
        interpretation = self.net_.module_.interpret_output(
            raw_predictions, reduction="sum"
        )
        self.net_.module_.plot_interpretation(interpretation)

    def plot_prediction(self, X, idx, add_loss_to_title=False, ax=None):
        """Provides a visual view of the model's output.

        Parameters
        ----------
        X : pd.DataFrame
            Dataframe to predict

        idx : int
            Number of sequence to visualize

        add_loss_to_title : bool
            If True, the loss is given inside the ax title

        ax : matplotlib ax
            matplotlib axes to plot on

        Returns
        -------
        None
        """
        raw_predictions, x = self._raw_predict(X, return_x=True)
        self.net_.module_.plot_prediction(
            x, raw_predictions, idx=idx, add_loss_to_title=add_loss_to_title,
            ax=ax
        )

    def _init_module(self, train_dataset):
        """Instantiates pytorch module using object (self) attributes and
        training dataset.

        Complete compatibility with skorch requires assigning two methods:
        `output_transformer` and `output_class`. See their docstring for
        further details

        Returns
        -------
        module : torch neural net object
            Instantiated neural net
        """
        module_kwargs = self.get_kwargs_for('module')
        module_kwargs.update({'output_transformer': output_transformer})
        module = self.module.from_dataset(train_dataset, **module_kwargs)
        module._output_class = output_class
        return module

    def _raw_predict(self, X, return_x=False):
        """Computes raw prediction.

        Used in `interpret_output` and `plot_prediction` methods.

        ``raw_predictions`` is a dictionary containing a lot of info about
        the model's output for any X. They are used for model interpretation
        and plotting output methods. To access raw predictions, the
        _output_class attribute has to be silenced to activate the default
        behaviour inside the TemporalFusionTransformer forward method. Such
        default behaviour along with kwarg mode='raw' will give us the
        raw predictions. At the end, the _output_class attribute is recovered.

        Parameters
        ----------
        X : pd.DataFrame
            Dataframe to predict

        return_x : bool
            If True, the given X is returned in tensor format

        Returns
        -------
        raw predictions : dict
        """
        # Delete _output_class attribute from skorch module
        if hasattr(self.net_.module, '_output_class'):
            module = True
            delattr(self.net_.module, '_output_class')
        else:
            module = False
        if hasattr(self.net_.module_, '_output_class'):
            module_ = True
            delattr(self.net_.module_, '_output_class')
        else:
            module_ = False

        # Generate raw predictions
        ds = self.get_dataset(X)
        raw_predictions = self.net_.module_.predict(
            ds, mode='raw', batch_size=len(ds), return_x=return_x
        )
        # Recover _output_class attribute
        if module:
            self.net_.module._output_class = output_class
        if module_:
            self.net_.module_._output_class = output_class
        return raw_predictions


class TemporalFusionTransformer(PyForecastTrainer):
    """Temporal Fusion Transformer for forecasting time series.

    Implementation of the article Temporal Fusion Transformers for
    Interpretable Multi-horizon Time Series Forecasting. The network
    outperforms DeepAR by Amazon by 36-69% in benchmarks.

    Parameters
    ----------
    group_ids : list of str
        List of column names identifying a time series. This means that the
        ``group_ids`` identify a sample together with the ``date``. If you
        have only one times eries, set this to the name of column that is
        constant.

    date : str
        Date column. This column is used to determine the sequence of samples.
        Ideally, all groups should contain the same dates

    target : str
        Target column. Column containing the values to be predicted.

    max_prediction_length : int
        Maximum prediction/decoder length. Usually this this is defined by
        the difference between forecasting dates.

    max_encoder_length : int, default=None
        Maximum length to encode (also known as `input sequence length`). This
        is the maximum history length used by the time series dataset. If None,
        3 times the ``max_prediction_length`` is used.

    time_varying_known_reals : list of str, default=None
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

    criterion : class, default=None
        The uninitialized criterion (loss) used to optimize the module. If
        None, the :class:`.RMSE` is used.

    optimizer : class, default=None
        The uninitialized optimizer (update rule) used to optimize the
        module. if None, :class:`.Adam` optimizer is used.

    lr : float, default=1e-5
        Learning rate passed to the optimizer.

    max_epochs : int, default=10
        The number of epochs to train for each :meth:`fit` call. Note that you
        may keyboard-interrupt training at any time.

    batch_size : int, default=64
        Mini-batch size. If ``batch_size`` is -1, a single batch with
        all the data will be used during training and validation.

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

    emb_dim : int, default=10
        Dimension of every embedding table

    hidden_size : int, default=16
        Size of the context vector

    hidden_continuous_size : int, default=8
        Hidden size for processing continuous variables

    lstm_layers : int, default=2
        Number of LSTM layers (2 is mostly optimal)

    dropout : float, default=0.1
        Dropout rate

    verbose : int, default=1
        This parameter controls how much print output is generated by
        the net and its callbacks. By setting this value to 0, e.g. the
        summary scores at the end of each epoch are no longer printed.
        This can be useful when running a hyperparameter search. The
        summary scores are always logged in the history attribute,
        regardless of the verbose setting.

    device : str, torch.device, default="cpu"
        The compute device to be used. If set to "cuda", data in torch
        tensors will be pushed to cuda tensors before being sent to the
        module. If set to None, then all compute devices will be left
        unmodified.
    """

    def __init__(self, group_ids, time_idx, target, max_prediction_length,
                 max_encoder_length, time_varying_known_reals,
                 time_varying_unknown_reals, static_categoricals,
                 cv_split=None, min_encoder_length=None,
                 criterion=None, optimizer=None, lr=1e-5, max_epochs=10,
                 batch_size=64, callbacks=None, emb_dim=10, hidden_size=16,
                 hidden_continuous_size=8, lstm_layers=2, dropout=0.1,
                 output_size=1, verbose=1, device='cpu', **kwargs):
        super().__init__(
            module=TFT,
            group_ids=group_ids,
            time_idx=time_idx,
            target=target,
            max_prediction_length=max_prediction_length,
            max_encoder_length=max_encoder_length,
            time_varying_known_reals=time_varying_known_reals,
            time_varying_unknown_reals=time_varying_unknown_reals,
            static_categoricals=static_categoricals,
            cv_split=cv_split,
            min_encoder_length=min_encoder_length,
            collate_fn=PyTorchForecastingDataset.tft_collate_fn,
            criterion=criterion,
            optimizer=optimizer,
            lr=lr,
            max_epochs=max_epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=verbose,
            device=device,
            **kwargs
        )
        self.emb_dim = emb_dim
        self.hidden_size = hidden_size
        self.hidden_continuous_size = hidden_continuous_size
        self.lstm_layers = lstm_layers
        self.dropout = dropout
        self.output_size = output_size
