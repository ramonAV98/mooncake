from pandas.api.types import is_datetime64_any_dtype


def check_group_id_presence(X, group_ids, group_id):
    """Checks passed `group_id` exists.
    """
    all_groups = list(X.groupby(group_ids).groups)
    if group_id not in all_groups:
        raise ValueError(f'Group id {group_id} does not exist.')


def check_is_datetime(series):
    """Checks if passed pandas `series` is datetime64 compatible.
    """
    if not is_datetime64_any_dtype(series):
        raise ValueError('Series is not datetime64 compatible.')


def check_group_ids(X, group_ids):
    """Checks `group_ids` columns are present in `X`.
    """
    msg = 'group_id column {} not found in X.'
    for col in group_ids:
        if col not in X:
            raise ValueError(msg.format(col))
