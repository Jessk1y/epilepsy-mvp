"""Порог клиники — «настройка под клинику».

Правила те же, что в экспериментах (h1_eeg/tuep/tuep_protocol.py: thr_youden, thr_recall80; h1_eeg/notebooks/h1_2026-11-10_zs_unlabeled-threshold.py):
- по размеченным записям клиники: индекс Юдена; дополнительно — порог, дающий чувствительность не ниже 80 %;
- без разметки: медиана оценок записей клиники.
Запись считается «выше порога», если оценка >= порога. Здесь нет sklearn; совпадение с tuep_protocol проверяется тестом.
"""
import numpy as np

MIN_LABELED, MIN_PER_CLASS, MIN_UNLABELED = 20, 5, 20


def _roc_points(y, s):
    """Пороги (по убыванию), чувствительность и доля ложных тревог при «оценка >= порог»."""
    y, s = np.asarray(y, int), np.asarray(s, float)
    th = np.unique(s)[::-1]
    tpr = np.array([((s >= t) & (y == 1)).sum() for t in th]) / max((y == 1).sum(), 1)
    fpr = np.array([((s >= t) & (y == 0)).sum() for t in th]) / max((y == 0).sum(), 1)
    return th, tpr, fpr


def thr_youden(y, s):
    """Порог с наибольшим «чувствительность + специфичность − 1»; при равенстве — самый высокий порог."""
    th, tpr, fpr = _roc_points(y, s)
    return float(th[np.argmax(tpr - fpr)])


def thr_sens80(y, s, target=0.8):
    """Самый высокий порог, при котором чувствительность не ниже 80 %."""
    th, tpr, _ = _roc_points(y, s)
    return float(th[np.argmax(tpr >= target)])


def thr_median(s):
    return float(np.median(np.asarray(s, float)))


def confusion(y, s, t):
    """Счётчики и метрики при пороге t: всё, что показывается врачу и идёт в отчёты."""
    y, s = np.asarray(y, int), np.asarray(s, float)
    pr = s >= t
    tp, tn = int((pr & (y == 1)).sum()), int((~pr & (y == 0)).sum())
    fp, fn = int((pr & (y == 0)).sum()), int((~pr & (y == 1)).sum())
    sens, spec = tp / max(tp + fn, 1), tn / max(tn + fp, 1)
    return dict(tp=tp, tn=tn, fp=fp, fn=fn, n_pos=tp + fn, n_neg=tn + fp, sensitivity=sens, specificity=spec,
                accuracy=(tp + tn) / max(len(y), 1), balanced_accuracy=(sens + spec) / 2,
                precision=tp / (tp + fp) if tp + fp else None)


def boot_ci(y, s, t, n=1000, seed=0):
    """95 % ДИ чувствительности и специфичности (бутстреп по записям) при уже выбранном пороге.
    Порог выбран по этим же записям, поэтому оценка оптимистична — это пишется рядом с цифрами."""
    y, s = np.asarray(y, int), np.asarray(s, float)
    r, se, sp = np.random.default_rng(seed), [], []
    for _ in range(n):
        i = r.integers(0, len(y), len(y))
        if (y[i] == 1).any() and (y[i] == 0).any():
            c = confusion(y[i], s[i], t); se.append(c["sensitivity"]); sp.append(c["specificity"])
    q = lambda v: [float(x) for x in np.percentile(v, [2.5, 97.5])] if v else None
    return dict(sensitivity=q(se), specificity=q(sp))


def auc(y, s):
    """ROC AUC (ранговая формула) — справочно, на записях клиники."""
    y, s = np.asarray(y, int), np.asarray(s, float)
    n1, n0 = int((y == 1).sum()), int((y == 0).sum())
    if not n1 or not n0:
        return None
    order = np.argsort(s, kind="mergesort"); ranks = np.empty(len(s)); ranks[order] = np.arange(1, len(s) + 1)
    for v in np.unique(s):                                  # средние ранги при совпадениях
        m = s == v
        if m.sum() > 1:
            ranks[m] = ranks[m].mean()
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def propose(scores, labels):
    """Варианты порога по записям клиники. scores — оценки; labels — 1 (эпилепсия), 0 (нет), None (без разметки).
    Возвращает {"labeled": {...} | None, "unlabeled": {...} | None, "warnings": [...]}."""
    s_all = np.array([float(x) for x in scores], float)
    lab = [(float(x), int(l)) for x, l in zip(scores, labels) if l in (0, 1)]
    out, warn = dict(labeled=None, unlabeled=None, warnings=[]), []
    if lab:
        s, y = np.array([a for a, _ in lab]), np.array([b for _, b in lab])
        n1, n0 = int(y.sum()), int((y == 0).sum())
        if n1 >= 1 and n0 >= 1:
            ty, t80 = thr_youden(y, s), thr_sens80(y, s)
            out["labeled"] = dict(n=len(y), n_pos=n1, n_neg=n0, auc=auc(y, s),
                                  youden=dict(threshold=ty, **confusion(y, s, ty), ci=boot_ci(y, s, ty)),
                                  sens80=dict(threshold=t80, **confusion(y, s, t80), ci=boot_ci(y, s, t80)))
            if len(y) < MIN_LABELED:
                warn.append(f"Размеченных записей {len(y)} — меньше {MIN_LABELED}: порог неустойчив.")
            if min(n1, n0) < MIN_PER_CLASS:
                warn.append(f"В одном из классов меньше {MIN_PER_CLASS} записей (эпилепсия {n1}, без эпилепсии {n0}).")
        else:
            warn.append("Нужны размеченные записи обоих классов: и с эпилепсией, и без.")
    if len(s_all):
        t = thr_median(s_all)
        out["unlabeled"] = dict(n=len(s_all), threshold=t, share_above=float((s_all >= t).mean()))
        if len(s_all) < MIN_UNLABELED:
            warn.append(f"Записей клиники {len(s_all)} — для порога по медиане лучше не меньше {MIN_UNLABELED}.")
    out["warnings"] = warn
    return out
