"""Деление на обучение/тест ТОЛЬКО по пациентам: куски одной записи не должны попасть в обе части."""
import numpy as np
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold


def assert_no_leak(train_groups, test_groups) -> None:
    common = set(train_groups) & set(test_groups)
    if common:
        raise AssertionError(f"Утечка: {len(common)} пациентов есть и в обучении, и в тесте: {sorted(common)[:5]}")


def patient_holdout(groups, test_size: float = 0.2, seed: int = 42):
    """Один раз делит выборки (куски/записи) по пациентам. Возвращает индексы (train_idx, test_idx).

    groups — id пациента для каждого куска. Для нескольких записей одного пациента id должен быть один.
    """
    groups = np.asarray(groups)
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, test_idx = next(gss.split(np.zeros(len(groups)), groups=groups))
    assert_no_leak(groups[train_idx], groups[test_idx])
    return train_idx, test_idx


def patient_kfold(labels, groups, n_splits: int = 5, seed: int = 42):
    """Кросс-валидация по пациентам с сохранением доли классов. Генератор (train_idx, test_idx)."""
    labels, groups = np.asarray(labels), np.asarray(groups)
    skf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for train_idx, test_idx in skf.split(np.zeros(len(labels)), labels, groups):
        assert_no_leak(groups[train_idx], groups[test_idx])
        yield train_idx, test_idx
