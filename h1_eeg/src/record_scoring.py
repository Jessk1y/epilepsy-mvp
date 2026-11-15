"""Оценка ОДНОЙ записи в памяти — для приложения (epilepsy-mvp/app). Только numpy и scipy.

Здесь то же, что делают скрипты TUEP над подготовленным набором на диске, но для массива (19, n) в памяти:
- окна для оценки: как tuep_finetune.eval_windows / read_window (фильтр по куску «окно ± 5 с») и как tuep_infer.py
  (ФВЧ по записи целиком, окна подряд);
- сборка оценки записи: среднее верхних 20 % вероятностей окон (top20);
- признаки эталона: мощности ритмов, амплитуда (tuep_baseline.file_features) и wPLI (tuep_baseline.wpli_features);
- логистическая регрессия из сохранённых массивов (без sklearn).

Совпадение с tuep_finetune.py и tuep_baseline.py проверяется тестом app/tests/test_record_scoring.py на синтетике.
Вход везде: X (19, n), мкВ, 200 Гц, каналы в порядке shared_data.EL, после общей подготовки (tuep_common.load_record).
"""
import json
import numpy as np, scipy.signal as ss

FS, L10, HP, CAP_WIN, TOP, MARGIN = 200, 2000, 0.3, 60, 0.2, 1000
BANDS = [(1, 4), (4, 8), (8, 13), (13, 30), (30, 45)]
IU = np.triu_indices(19, 1)


def spread(n, k):
    """До k номеров из n, равномерно (как в блокнотах и tuep_infer.py)."""
    return np.unique(np.linspace(0, n - 1, k).astype(int)) if n > k else np.arange(n)


def top20(p):
    """Оценка записи: среднее верхних 20 % вероятностей окон."""
    p = np.asarray(p, float)
    return float(np.sort(p)[::-1][:int(np.ceil(TOP * len(p)))].mean())


def make_sos(lowpass=0.0, hp=HP):
    """ФВЧ 0,3 Гц (+ ФНЧ, если задан) — как tuep_finetune.make_sos."""
    sos = ss.butter(4, hp, btype="highpass", fs=FS, output="sos")
    if lowpass:
        sos = np.vstack([sos, ss.butter(6, lowpass, btype="lowpass", fs=FS, output="sos")])
    return sos


def window_from_array(X, start, sos, L=L10):
    """Окно L отсчётов: кусок с полями ±5 с (сколько есть) фильтруется, вырезается середина — как tuep_finetune.read_window."""
    n = X.shape[-1]
    a, b = max(0, start - MARGIN), min(n, start + L + MARGIN)
    return ss.sosfiltfilt(sos, np.asarray(X[:, a:b], np.float64), axis=-1)[:, start - a:start - a + L].astype(np.float32)


def eval_windows_array(X, sos, n_win=CAP_WIN, L=L10):
    """Окна записи для оценки (как tuep_finetune.eval_windows): до n_win равномерно из окон длины L подряд с начала.
    Возвращает (W (k, 19, L) float32, начала окон в отсчётах). Запись короче окна — k = 0."""
    n = X.shape[-1]
    if n < L:
        return np.zeros((0, 19, L), np.float32), np.zeros(0, int)
    starts = spread(n // L, n_win) * L
    return np.stack([window_from_array(X, int(s), sos, L) for s in starts]), starts


def whole_record_windows(X, n_win=CAP_WIN, L=L10, hp=HP):
    """Окна как в tuep_infer.py и блокнотах REVE v4 / EEG-DINO v2: ФВЧ 0,3 Гц по записи целиком -> окна подряд -> до n_win равномерно."""
    n = X.shape[-1]
    if n < L:
        return np.zeros((0, 19, L), np.float32), np.zeros(0, int)
    Y = ss.sosfiltfilt(ss.butter(4, hp, btype="highpass", fs=FS, output="sos"), np.asarray(X, np.float64), axis=-1).astype(np.float32)
    idx = spread(n // L, n_win)
    return np.stack([Y[:, i * L:(i + 1) * L] for i in idx]), idx * L


# ---------------------------------------------------------------- эталон: мощности, амплитуда, wPLI
def band_amp_matrix(W):
    """По окнам: lg относительной мощности 5 ритмов в 19 каналах (95) + lg СКО канала (19) -> (k, 114). Как tuep_baseline.file_features."""
    f, P = ss.welch(np.asarray(W, np.float64), FS, nperseg=400, axis=-1)
    tot = P[..., (f >= 1) & (f < 45)].sum(-1) + 1e-12
    rel = np.stack([np.log10(P[..., (f >= lo) & (f < hi)].sum(-1) / tot + 1e-9) for lo, hi in BANDS], -1).reshape(len(W), -1)
    amp = np.log10(np.asarray(W).std(-1) + 1e-3)
    return np.concatenate([rel, amp], 1)


def band_sos():
    return [ss.butter(4, b, btype="bandpass", fs=FS, output="sos") for b in BANDS]


def wpli_matrix(W, bsos=None):
    """По окнам: средний wPLI по 171 паре в 5 диапазонах + сила узла каждого канала в θ 4–8 Гц -> (k, 24). Как tuep_baseline.wpli_features."""
    bsos = bsos or band_sos()
    out = []
    for w in np.asarray(W, np.float64):
        v = []
        for b, sb in enumerate(bsos):
            z = ss.hilbert(ss.sosfiltfilt(sb, w, axis=-1), axis=-1)
            im = (z[IU[0]] * np.conj(z[IU[1]])).imag
            wp = np.abs(im.mean(-1)) / (np.abs(im).mean(-1) + 1e-12)
            v.append([wp.mean()])
            if b == 1:
                M = np.zeros((19, 19)); M[IU] = wp; M = M + M.T
                v.append(M.sum(1) / 18)
        out.append(np.concatenate(v))
    return np.array(out)


def agg(M):
    """Окна -> запись: медиана и 90-й процентиль по окнам."""
    return np.concatenate([np.median(M, 0), np.percentile(M, 90, 0)])


def baseline_vector(MB, MW):
    """Вектор признаков набора bpampw: [мощности и амплитуда: медиана, p90 (228)] + [wPLI: медиана, p90 (48)]."""
    return np.concatenate([agg(MB), agg(MW)])


class LogReg:
    """Логистическая регрессия из сохранённых массивов (tuep_baseline.py --save-model): стандартизация + линейный слой."""

    def __init__(self, path):
        d = np.load(path, allow_pickle=False)
        self.mean, self.scale, self.coef, self.intercept = d["mean"], d["scale"], d["coef"], float(d["intercept"])
        self.info = json.loads(str(d["info"])) if "info" in d else {}

    def predict_proba(self, x):
        z = np.nan_to_num((np.asarray(x, np.float64) - self.mean) / self.scale) @ self.coef + self.intercept
        return float(1.0 / (1.0 + np.exp(-z)))


def save_logreg(path, scaler, model, info):
    """Сохранить StandardScaler + LogisticRegression как набор массивов (читается LogReg без sklearn)."""
    np.savez(path, mean=scaler.mean_, scale=scaler.scale_, coef=model.coef_[0], intercept=np.float64(model.intercept_[0]),
             info=np.array(json.dumps(info, ensure_ascii=False)))
