"""Метрики классификации. Считаем на уровне ПАЦИЕНТА, а не куска: кусков много, пациентов мало."""
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score


def aggregate_by_patient(patient_ids, y_true, y_score) -> pd.DataFrame:
    """Среднее по кускам -> одна оценка на пациента. Метка пациента должна быть одинаковой у всех его кусков."""
    df = pd.DataFrame({"patient": patient_ids, "y": y_true, "score": y_score})
    bad = df.groupby("patient")["y"].nunique()
    if (bad > 1).any():
        raise ValueError(f"У пациентов разные метки: {list(bad[bad > 1].index[:5])}")
    return df.groupby("patient").agg(y=("y", "first"), score=("score", "mean")).reset_index()


def binary_report(y_true, y_score, threshold: float = 0.5) -> dict:
    """ROC AUC, PR AUC, чувствительность, специфичность. Порог задавайте по валидации, не по тесту."""
    y_true, y_score = np.asarray(y_true), np.asarray(y_score)
    y_pred = (y_score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "n": int(len(y_true)),
        "n_pos": int(y_true.sum()),
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "pr_auc": float(average_precision_score(y_true, y_score)),
        "sensitivity": float(tp / (tp + fn)) if (tp + fn) else float("nan"),
        "specificity": float(tn / (tn + fp)) if (tn + fp) else float("nan"),
        "threshold": threshold,
    }


def bootstrap_auc_ci(y_true, y_score, n_boot: int = 1000, seed: int = 42, alpha: float = 0.05):
    """Доверительный интервал ROC AUC бутстрепом. Подавайте данные уровня пациента (aggregate_by_patient)."""
    y_true, y_score = np.asarray(y_true), np.asarray(y_score)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y_true), len(y_true))
        if len(set(y_true[idx])) < 2:
            continue
        vals.append(roc_auc_score(y_true[idx], y_score[idx]))
    lo, hi = np.percentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)
