"""
The :mod:`blue_meth.model_selection` module includes classes and
functions to split the data based on a preset strategy.

"""

from ._split import train_test_split, WalkForwardCV

__all__ = [
    'train_test_split',
    'WalkForwardCV'
]
