"""
Module for custom torch datasets.
"""

import inspect
import warnings
from abc import ABCMeta, abstractmethod
from collections import Sequence

import numpy as np
import pandas as pd
import torch
from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.data.timeseries import (
    check_for_nonfinite, _find_end_indices
)
from torch.nn.utils import rnn
from torch.utils.data import Dataset

from ..preprocessing import MultiColumnLabelEncoder


class BaseDataset(Dataset, metaclass=ABCMeta):
    """Base abstract class for PyTorch custom Datasets

    .. note::
        This class should not be used directly. Use derived classes instead.
    """

    @abstractmethod
    def __init__(self, *args, **kwargs):
        pass

    @abstractmethod
    def __getitem__(self, index):
        pass

    @abstractmethod
    def __len__(self):
        pass

    @staticmethod
    def collate_fn(batches):
        """Collate_fn to be give to PyTorch :class:`Dataloader`

        Override this method in case the derived Dataset class needs a custom
        collate fn
        """
        return

    @property
    @abstractmethod
    def groups_sizes(self):
        """Size for each group

        Returns
        -------
        group_sizes : dict, str -> int
        """
        pass

    @classmethod
    def from_parameters(cls, parameters, X, **kwargs):
        """Builds dataset from given parameters

        Parameters
        ----------
        parameters : dict
            Dataset parameters which to use for the new dataset

        X : pd.DataFrame
            Data from which new dataset will be generated

        kwargs : keyword arguments overriding parameters

        Returns
        -------
        Dataset
        """
        parameters.update(kwargs)
        return cls(X, **parameters)

    def get_parameters(self):
        """Get parameters that can be used with :py:meth:`~from_parameters` to
        create a new dataset with the same categorical encoders.

        Returns
        -------
        params : dict
        """
        kwargs = {
            name: getattr(self, name)
            for name in
            inspect.signature(self.__class__.__init__).parameters.keys()
            if name not in ["self", "X", "data"]
        }
        return kwargs


class SeqToSeqDataset(BaseDataset):
    """Dataset for SeqToSeq model

    In brief, this is a light weight version of :class:`TimeSeriesDataset` from
    the pytorch_forecasting library.

    Parameters
    ----------
    X : pd.DataFrame
        Dataframe with time series data. Each row can be identified with
        ``date`` and the ``group_ids``.

    group_ids : list of str
        List of column names identifying a time series. This means that the
        ``group_ids`` identify a sample together with ``date``. If you
        have only one times series, set this to the name of column that is
        constant.

    time_idx : str
        Time index column. This column is used to determine the sequence of
        samples.

    target : str
        Target column. Column containing the values to be predicted.

    max_prediction_length : int
        Maximum prediction/decoder length. Usually this is defined by the
        difference between forecasting dates.

    max_encoder_length : int, default=None
        Maximum length to encode (also known as `input sequence length`). This
        is the maximum history length used by the time series dataset.

    time_varying_known_reals : list of str
        List of continuous variables that change over time and are known in the
        future (e.g. price of a product, but not demand of a product).

    time_varying_unknown_reals : list of str
        List of continuous variables that change over time and are not known in
        the future. You might want to include your ``target`` here.

    static_categoricals : list of str, default=None
        List of categorical variables that do not change over time (also known
        as `time independent variables`). If None, ``group_ids`` will be used.

    min_encoder_length : int, default=None
        Minimum allowed length to encode. If None, defaults to
        ``max_encoder_length``.

    categorical_encoder : MultiColumnLabelEncoder object, default=None

    predict_mode : bool, default=False

    add_missing_sequences : bool, default=False
        The missing_sequences ensure that there is a sequence that finishes on
        every timestep. Warning: might drastically increase the length of the
        dataset.

    Attributes
    ----------
    data : dict of tensors
        Dictionary containing the data in ``X`` divided in five groups:
            - reals: continuous data (includes target)
            - categoricals: encoded categorical data
            - target: target data
            - groups:  encoded group ids
            - time: time index
    """

    def __init__(self, X, group_ids, time_idx, target, max_prediction_length,
                 max_encoder_length, time_varying_known_reals,
                 time_varying_unknown_reals, static_categoricals=None,
                 min_encoder_length=None, categorical_encoder=None,
                 predict_mode=False, add_missing_sequences=False):
        self.group_ids = group_ids
        self.time_idx = time_idx
        self.target = target
        self.max_prediction_length = max_prediction_length
        self.max_encoder_length = max_encoder_length
        self.time_varying_known_reals = time_varying_known_reals
        self.time_varying_unknown_reals = time_varying_unknown_reals
        self.static_categoricals = static_categoricals
        self._validate_segmentation(X)
        self.categorical_encoder = categorical_encoder
        if min_encoder_length is None:
            min_encoder_length = max_encoder_length
        self.min_encoder_length = min_encoder_length
        self.min_prediction_length = max_prediction_length
        self.predict_mode = predict_mode
        self.add_missing_sequences = add_missing_sequences

        # Preprocess X
        X = self._preprocess_X(X)
        self.min_prediction_idx = X[self.time_idx].min()
        self.index = self._construct_index(X)
        self.data = self._X_to_tensors(X)

    def __getitem__(self, idx):
        """Returns a single dataset sample
        """
        # Get continuous and categorical data
        index = self.index.iloc[idx]
        data_cont = self.data["reals"][
                    index.index_start: index.index_end + 1].clone()
        data_cat = self.data["categoricals"][
                   index.index_start: index.index_end + 1].clone()

        # Time index
        time = self.data["time"][
               index.index_start: index.index_end + 1].clone()

        # Determine data window (encoder and decoder lengths)
        sequence_length = len(time)
        assert (
                sequence_length >= self.min_prediction_length
        ), "Sequence length should be at least minimum prediction length"
        # determine prediction/decode length and encode length
        decoder_length = min(
            time[-1] - (self.min_prediction_idx - 1),
            self.max_prediction_length,
            sequence_length - self.min_encoder_length,
        )
        encoder_length = sequence_length - decoder_length
        assert (
                decoder_length >= self.min_prediction_length
        ), "Decoder length should be at least minimum prediction length"
        assert encoder_length >= self.min_encoder_length, (
            "Encoder length should be at least minimum encoder length"
        )

        # Split continuous data into encoder and decoder inputs.
        # ``data_cont`` contains the n time_varying_unknown features
        # at the end (see ``_X_to_tensors`` method). Since the decoder only
        # works with known data, we must exclude them
        n = len(self.time_varying_unknown_reals)
        enc_x = data_cont[:encoder_length, :]
        dec_x = data_cont[-decoder_length:, :-n]

        # Get unique element from each categorical column for embedding input
        emb_x = data_cat.unique(dim=0).squeeze(-1)

        # Get target values. ``y`` contains both encoder and decoder target
        # values. However, only decoder target values, ``dec_y``, are used.
        y = [
            d[index.index_start: index.index_end + 1].clone()
            for d in self.data["target"]
        ]
        dec_y = y[0][encoder_length:]

        # Return (X, y) tuple where X is a dict with all the module inputs.
        # Notice target values ``dec_y`` are also delivered as input since
        # teacher forcing is available
        return dict(emb_x=emb_x, enc_x=enc_x, dec_x=dec_x, dec_y=dec_y), dec_y

    def __len__(self):
        return len(self.index)

    @staticmethod
    def collate_fn(batches):
        """collate_fn for padding variable length sequences

        SeqToSeq only allows encoder input to be variable length.
        """
        emb_x = torch.stack([b[0]["emb_x"] for b in batches]).view(
            len(batches), -1)
        enc_lens = [len(b[0]['enc_x']) for b in batches]
        enc_x_pad = rnn.pad_sequence(
            [b[0]["enc_x"] for b in batches], batch_first=True
        )
        dec_x = torch.stack([b[0]["dec_x"] for b in batches])
        dec_y = torch.stack([b[0]["dec_y"] for b in batches])

        return dict(
            emb_x=emb_x,
            enc_x=enc_x_pad,
            dec_x=dec_x,
            dec_y=dec_y,
            enc_lens=enc_lens,
        ), dec_y

    @property
    def groups_sizes(self):
        """Returns a mapping from group_id to the number of samples it has
        inside the dataset.

        Returns
        -------
        group_sizes : dict, str -> int
        """
        index_start = self.index["index_start"].to_numpy()
        x = pd.DataFrame(
            self.data["groups"][index_start].numpy(), columns=self.group_ids
        )
        x_inv = self.categorical_encoder.inverse_transform(x)
        sizes_dict = x_inv.groupby(
            self.group_ids, sort=False).grouper.size().to_dict()
        return sizes_dict

    def timeseries_cv(self, n):
        val_idx = self.index.groupby('group_id').tail(n).index.values
        train_idx = self.index[~self.index.index.isin(val_idx)].index.values
        return [(train_idx, val_idx)]

    def _validate_segmentation(self, X):
        """Checks presence of columns in X and guarantees
        ``static_categoricals`` include ``group_ids``
        """
        # check presence in ``X``
        for col in self.time_varying_known_reals:
            if col not in X:
                raise ValueError(
                    'column {} from time_varying_known_reals was not found '
                    'in X'.format(col)
                )
        for col in self.time_varying_unknown_reals:
            if col not in X:
                raise ValueError(
                    'column {} from time_varying_unknown_reals was not found '
                    'in X'.format(col)
                )
        # ``static_categoricals always have to include ``group_ids``
        if self.static_categoricals is not None:
            for g in self.group_ids:
                if g not in self.static_categoricals:
                    self.static_categoricals.append(g)
            for col in self.static_categoricals:
                if col not in X:
                    raise ValueError(
                        'column {} from static_categoricals was not found '
                        'in X'.format(col)
                    )
        else:
            self.static_categoricals = self.group_ids

    def _preprocess_X(self, X):
        """Encodes static categoricals
        """
        # Encode static categoricals
        if self.categorical_encoder is None:
            self.categorical_encoder = MultiColumnLabelEncoder(
                self.static_categoricals).fit(X)
        X = self.categorical_encoder.transform(X)

        # Sort and reset
        pk = self.group_ids + [self.time_idx]  # primary key
        X = X.sort_values(pk).reset_index(drop=True)
        return X

    def _construct_index(self, X):
        """Creates index of samples.

        Parameters
        ----------
        X : pd.DataFrame
            Categorically encoded X

        Returns
        -------
        pd.DataFrame: index dataframe
        """
        g = X.groupby(self.group_ids, observed=True)

        df_index_first = g[self.time_idx].transform("nth", 0).to_frame(
            "time_first")
        df_index_last = g[self.time_idx].transform("nth", -1).to_frame(
            "time_last")
        df_index_diff_to_next = -g[self.time_idx].diff(-1).fillna(-1).astype(
            int).to_frame("time_diff_to_next")
        df_index = pd.concat(
            [df_index_first, df_index_last, df_index_diff_to_next], axis=1)
        df_index["index_start"] = np.arange(len(df_index))
        df_index["time"] = X[self.time_idx]
        df_index["count"] = (df_index["time_last"] - df_index[
            "time_first"]).astype(int) + 1
        group_ids = g.ngroup()
        df_index["group_id"] = group_ids

        min_sequence_length = (
                self.min_prediction_length + self.min_encoder_length
        )
        max_sequence_length = (
                self.max_prediction_length + self.max_encoder_length
        )

        # Calculate maximum index to include from current index_start
        max_time = (df_index["time"] + max_sequence_length - 1).clip(
            upper=df_index["count"] + df_index.time_first - 1)

        df_index["index_end"], missing_sequences = _find_end_indices(
            diffs=df_index.time_diff_to_next.to_numpy(),
            max_lengths=(max_time - df_index.time).to_numpy() + 1,
            min_length=min_sequence_length,
        )

        # Add duplicates but mostly with shorter sequence length for start of
        # timeseries. While the previous steps have ensured that we start a
        # sequence on every time step, the missing_sequences
        # ensure that there is a sequence that finishes on every timestep
        if len(missing_sequences) > 0 and self.add_missing_sequences:
            shortened_sequences = df_index.iloc[
                missing_sequences[:, 0]].assign(
                index_end=missing_sequences[:, 1])

            # Concatenate shortened sequences
            df_index = pd.concat([df_index, shortened_sequences], axis=0,
                                 ignore_index=True)

        # Filter out where encode and decode length are not satisfied
        df_index["sequence_length"] = df_index["time"].iloc[
                                          df_index["index_end"]
                                      ].to_numpy() - df_index["time"] + 1

        # Filter too short sequences
        df_index = df_index[
            # Sequence must be at least of minimal prediction length
            lambda x: (x.sequence_length >= min_sequence_length)
                      &
                      # Prediction must be for after minimal prediction index +
                      # length of prediction
                      (x["sequence_length"] + x["time"] >=
                       self.min_prediction_idx + self.min_prediction_length)
        ]

        # Check that all groups/series have at least one entry in the index
        if not group_ids.isin(df_index.group_id).all():
            missing_groups = X.loc[~group_ids.isin(
                df_index.group_id), self.group_ids].drop_duplicates()
            warnings.warn(
                "Min encoder length and/or min_prediction_idx and/or min "
                "prediction length and/or lags are too large for "
                "{} series/groups which therefore are not "
                "present in the dataset index. This means no predictions can "
                "be made for those series. "
                "First 10 removed groups: {}".format(
                    len(missing_groups),
                    list(
                        missing_groups.iloc[:10].to_dict(
                            orient='index').values()
                    )
                )
            )
        assert (
                len(df_index) > 0
        ), ("Min encoder length and/or and/or max encoder length and/or "
            "max prediction length are too large for ALL groups")

        # Predict mode
        if self.predict_mode:
            # Get the rows containing the max sequence length of their group.
            # Note that if a group has multiple max values, all will be
            # returned.
            max_on_each_row = df_index.groupby('group_id')[
                'sequence_length'].transform(max)
            idx = (max_on_each_row == df_index['sequence_length'])
            df_index = df_index.loc[idx].reset_index(drop=True)
        else:
            df_index = df_index.reset_index(drop=True)

        return df_index

    def _X_to_tensors(self, X):
        """Convert data to tensors for faster access with :meth:`__getitem__`.

        Parameters
        ----------
        X : pd.DataFrame
            Categorically encoded X

        Returns
        -------
        dict of tensors
            Dictionary of tensors for continuous, categorical data, groups,
            target and time index
        """
        groups = check_for_nonfinite(
            torch.tensor(X[self.group_ids].to_numpy(np.long),
                         dtype=torch.long), self.group_ids
        )
        time = check_for_nonfinite(
            torch.tensor(X[self.time_idx].to_numpy(np.long),
                         dtype=torch.long), self.time_idx
        )
        categorical = check_for_nonfinite(
            torch.tensor(
                X[self.static_categoricals].to_numpy(np.long),
                dtype=torch.long
            ),
            self.static_categoricals
        )
        target = [
            check_for_nonfinite(
                torch.tensor(
                    X[self.target].to_numpy(dtype=np.float),
                    dtype=torch.float),
                self.target,
            )
        ]
        reals = self.time_varying_known_reals + self.time_varying_unknown_reals
        continuous = check_for_nonfinite(
            torch.tensor(X[reals].to_numpy(dtype=np.float), dtype=torch.float),
            reals
        )

        tensors = dict(
            reals=continuous,
            categoricals=categorical,
            target=target,
            groups=groups,
            time=time
        )
        return tensors


class PyTorchForecastingDataset(TimeSeriesDataSet, BaseDataset):
    """Dataset for pytorch_forecasting models.

    Parameters
    ----------
    X : pd.DataFrame
        Dataframe with time series data. Each row can be identified with
        ``date`` and the ``group_ids``.

    group_ids : list of str
        List of column names identifying a time series. This means that the
        ``group_ids`` identify a sample together with ``date``. If you
        have only one times series, set this to the name of column that is
        constant.

    date : str
        Date column

    target : str
        Target column

    max_prediction_length : int
        Maximum prediction/decoder length. Usually this is defined by the
        difference between forecasting dates.

    max_encoder_length : int, default=None
        Maximum length to encode (also known as `input sequence length`). This
        is the maximum history length used by the time series dataset.

    time_varying_known_reals : list of str
        List of continuous variables that change over time and are known in the
        future (e.g. price of a product, but not demand of a product).

    time_varying_unknown_reals : list of str
        List of continuous variables that change over time and are not known in
        the future. You might want to include your ``target`` here.

    static_categoricals : list of str
        List of categorical variables that do not change over time (also known
        as `time independent variables`). You might want to include your
        ``group_ids`` here for the learning algorithm to distinguish between
        different time series.
    """

    def __init__(self, data, group_ids, date, target, max_prediction_length,
                 max_encoder_length, time_varying_known_reals,
                 time_varying_unknown_reals, static_categoricals,
                 scalers=None, categorical_encoders=None):

        self.date = date  # save date attribute for get_params method to work
        data = _add_time_idx(data, group_ids, date)
        if scalers is None:
            scalers = {k: None for k in time_varying_known_reals}
        if categorical_encoders is None:
            categorical_encoders = {}
        super().__init__(
            data=data,
            time_idx='time_idx',
            target=target,
            group_ids=group_ids,
            max_encoder_length=max_encoder_length,
            max_prediction_length=max_prediction_length,
            time_varying_known_reals=time_varying_known_reals,
            time_varying_unknown_reals=time_varying_unknown_reals,
            static_categoricals=static_categoricals,
            scalers=scalers,
            categorical_encoders=categorical_encoders,
            target_normalizer=None
        )

    @property
    def groups_sizes(self):
        return self.decoded_index.groupby(
            self.group_ids, sort=False).grouper.size().to_dict()

    @staticmethod
    def collate_fn(batches):
        """Collate fn for temporal fusion transformer (TFT)

        This is a modified version of :meth:`TimeSeriesDataset._collate_fn`
        from pytorch_forecasting library to match skorch requirements for data
        delivery.

        Summary of modifications in order to satisfy skorch
        convention:

        - weights are ignored
        - X is a dict of a dict
        - y is reshaped
        """
        batch_size = len(batches)
        # collate function for dataloader
        # lengths
        encoder_lengths = torch.tensor(
            [batch[0]["encoder_length"] for batch in batches],
            dtype=torch.long)
        decoder_lengths = torch.tensor(
            [batch[0]["decoder_length"] for batch in batches],
            dtype=torch.long)

        # ids
        decoder_time_idx_start = (
                torch.tensor(
                    [batch[0]["encoder_time_idx_start"] for batch in batches],
                    dtype=torch.long) + encoder_lengths
        )
        decoder_time_idx = decoder_time_idx_start.unsqueeze(1) + torch.arange(
            decoder_lengths.max()).unsqueeze(0)
        groups = torch.stack([batch[0]["groups"] for batch in batches])

        # features
        encoder_cont = rnn.pad_sequence(
            [batch[0]["x_cont"][:length] for length, batch in
             zip(encoder_lengths, batches)], batch_first=True
        )
        encoder_cat = rnn.pad_sequence(
            [batch[0]["x_cat"][:length] for length, batch in
             zip(encoder_lengths, batches)], batch_first=True
        )

        decoder_cont = rnn.pad_sequence(
            [batch[0]["x_cont"][length:] for length, batch in
             zip(encoder_lengths, batches)], batch_first=True
        )
        decoder_cat = rnn.pad_sequence(
            [batch[0]["x_cat"][length:] for length, batch in
             zip(encoder_lengths, batches)], batch_first=True
        )

        # target scale
        if isinstance(batches[0][0]["target_scale"],
                      torch.Tensor):  # stack tensor
            target_scale = torch.stack(
                [batch[0]["target_scale"] for batch in batches])
        elif isinstance(batches[0][0]["target_scale"], (list, tuple)):
            target_scale = []
            for idx in range(len(batches[0][0]["target_scale"])):
                if isinstance(batches[0][0]["target_scale"][idx],
                              torch.Tensor):  # stack tensor
                    scale = torch.stack(
                        [batch[0]["target_scale"][idx] for batch in batches])
                else:
                    scale = torch.tensor(
                        [batch[0]["target_scale"][idx] for batch in batches],
                        dtype=torch.float)
                target_scale.append(scale)
        else:  # convert to tensor
            target_scale = torch.tensor(
                [batch[0]["target_scale"] for batch in batches],
                dtype=torch.float)

        # target and weight
        if isinstance(batches[0][1][0], (tuple, list)):
            target = [
                rnn.pad_sequence([batch[1][0][idx] for batch in batches],
                                 batch_first=True)
                for idx in range(len(batches[0][1][0]))
            ]
            encoder_target = [
                rnn.pad_sequence(
                    [batch[0]["encoder_target"][idx] for batch in batches],
                    batch_first=True)
                for idx in range(len(batches[0][1][0]))
            ]
        else:
            # TimeSeriesDataset includes a tuple of the form (y, weight)
            # as target. This procedure allows us to keep only the y
            target = []
            for b in batches:
                y = b[1]
                if isinstance(y, (tuple, list)):
                    y = y[0]
                target.append(y)
            target = rnn.pad_sequence(target, batch_first=True)
            encoder_target = rnn.pad_sequence(
                [batch[0]["encoder_target"] for batch in batches],
                batch_first=True)
        x = dict(
            encoder_cat=encoder_cat,
            encoder_cont=encoder_cont,
            encoder_target=encoder_target,
            encoder_lengths=encoder_lengths,
            decoder_cat=decoder_cat,
            decoder_cont=decoder_cont,
            decoder_target=target,
            decoder_lengths=decoder_lengths,
            decoder_time_idx=decoder_time_idx,
            groups=groups,
            target_scale=target_scale,
        )
        y = target.reshape(batch_size, -1)
        # Summary of modifications in order to satisfy skorch convention:
        #   - weights are ignored
        #   - X is a dict of a dict
        #   - y is reshaped
        return {'x': x}, y


class SliceDataset(Sequence, Dataset):
    """Makes ``dataset`` sliceable.

    Helper class that wraps a torch dataset to make it work with
    sklearn. That is, sometime sklearn will touch the input data, e.g. when
    splitting the data for a grid search. This will fail when the input data is
    a torch dataset. To prevent this, use this wrapper class for your
    dataset.

    ``dataset`` attributes are also available from :class:`SliceDataset`
    object (see Examples section).

    Parameters
    ----------
    dataset : torch.utils.data.Dataset
      A valid torch dataset.

    indices : list, np.ndarray, or None (default=None)
      If you only want to return a subset of the dataset, indicate
      which subset that is by passing this argument. Typically, this
      can be left to be None, which returns all the data.

    Examples
    --------
    >>> X = MyCustomDataset()
    >>> search = GridSearchCV(net, params, ...)
    >>> search.fit(X, y)  # raises error
    >>> ds = SliceDataset(X)
    >>> search.fit(ds, y)  # works
    >>> ds.a  # returns 1 since ``X`` attributes are also available from ``ds``

    Notes
    -----
    This class will only return the X value by default (i.e. the
    first value returned by indexing the original dataset). Sklearn,
    and hence skorch, always require 2 values, X and y. Therefore, you
    still need to provide the y data separately.

    This class behaves similarly to a PyTorch
    :class:`~torch.utils.data.Subset` when it is indexed by a slice or
    numpy array: It will return another ``SliceDataset`` that
    references the subset instead of the actual values. Only when it
    is indexed by an int does it return the actual values. The reason
    for this is to avoid loading all data into memory when sklearn,
    for instance, creates a train/validation split on the
    dataset. Data will only be loaded in batches during the fit loop.
    """

    def __init__(self, dataset, indices=None):
        self.dataset = dataset
        self.indices = indices
        self.indices_ = (
            self.indices if self.indices is not None
            else np.arange(len(self.dataset))
        )
        self.ndim = 1

    @property
    def shape(self):
        return (len(self),)

    def transform(self, data):
        """Additional transformations on ``data``.

        Notes
        -----
        If you use this in conjunction with PyTorch
        :class:`~torch.utils.data.DataLoader`, the latter will call
        the dataset for each row separately, which means that the
        incoming ``data`` is a single rows.

        """
        return data

    def __getattr__(self, attr):
        """If attr is not in self, look in self.dataset.

        Notes
        -----
        Issues with serialization were solved with the following discussion:
        https://stackoverflow.com/questions/49380224/how-to-make-classes-with-getattr-pickable
        """
        if 'dataset' not in vars(self):
            raise AttributeError
        return getattr(self.dataset, attr)

    def __len__(self):
        return len(self.indices_)

    def __getitem__(self, i):
        if isinstance(i, (int, np.integer)):
            Xn = self.dataset[self.indices_[i]]
            return self.transform(Xn)
        if isinstance(i, slice):
            return SliceDataset(self.dataset, indices=self.indices_[i])
        if isinstance(i, np.ndarray):
            if i.ndim != 1:
                raise IndexError(
                    "SliceDataset only supports slicing with 1 "
                    "dimensional arrays, got {} dimensions "
                    "instead".format(i.ndim)
                )
            if i.dtype == np.bool:
                i = np.flatnonzero(i)
        return SliceDataset(self.dataset, indices=self.indices_[i])
