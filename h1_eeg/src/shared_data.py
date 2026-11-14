"""Общий набор Г1 (NMT + турецкий) — выход блокнота Kaggle zakharsann/h1-prep-shared.

Подключение в kernel-metadata.json: "kernel_sources": ["zakharsann/h1-prep-shared"].
Блокноты моделей копируют этот файл в начало (одна ячейка) и пользуются функциями ниже.

Формат (подробно — в описании блокнота h1_2026-10-26_zs_prep-shared.ipynb):
- h1_shared_meta.csv — по строке на запись: dataset, part, split, id, y, ..., shard200/off200/n200 (и shard256/... если есть);
- h1_shared_info.json — фильтр, множитель масштаба NMT, сколько минут NMT train сохранено, порядок каналов;
- fs200/<имя>.npy — плоские массивы float16: записи (19, n) в C-порядке подряд, запись = arr[off : off + 19*n].
Сигнал в мкВ, 19 каналов в порядке EL, средний референт, 0,1-75 Гц + режекция 50 Гц. Окна НЕ нормированы.
"""
import glob, json, os
from fractions import Fraction
from concurrent.futures import ThreadPoolExecutor
import numpy as np, pandas as pd, scipy.signal as ss

EL = ["Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2", "F7", "F8", "T3", "T4", "T5", "T6", "Fz", "Cz", "Pz"]


def find_root(base="/kaggle/input"):
    """Папка общего набора среди подключённых источников."""
    hits = glob.glob(f"{base}/**/h1_shared_meta.csv", recursive=True)
    assert hits, "общий набор не подключён: добавьте zakharsann/h1-prep-shared в kernel_sources"
    return os.path.dirname(hits[0])


def load_meta(root):
    """Таблица записей и описание набора."""
    meta = pd.read_csv(f"{root}/h1_shared_meta.csv")
    info = json.load(open(f"{root}/h1_shared_info.json", encoding="utf-8"))
    return meta, info


_SHARDS = {}
def _shard(root, fs, name):
    k = (root, fs, name)
    if k not in _SHARDS:
        _SHARDS[k] = np.load(f"{root}/fs{fs}/{name}", mmap_mode="r")
    return _SHARDS[k]


def read_record(root, row, fs=200):
    """Одна запись: (19, n) float16, мкВ. row — строка meta (Series или dict).
    Через mmap — для единичных записей; много записей читать через read_all (на /kaggle/input mmap очень медленный)."""
    n, off = int(row[f"n{fs}"]), int(row[f"off{fs}"])
    return np.array(_shard(root, fs, row[f"shard{fs}"])[off:off + 19 * n]).reshape(19, n)


def read_all(root, meta, fs=200, threads=4):
    """Все записи meta в словарь id -> (19, n) float16: каждый файл-кусок читается целиком последовательно, записи режутся в памяти."""
    def load(item):
        name, g = item
        arr = np.load(f"{root}/fs{fs}/{name}")
        return [(i, arr[o:o + 19 * n].reshape(19, n).copy()) for i, o, n in zip(g["id"], g[f"off{fs}"], g[f"n{fs}"])]
    out = {}
    with ThreadPoolExecutor(threads) as ex:
        for part in ex.map(load, list(meta.groupby(f"shard{fs}"))):
            out.update(part)
    return {i: out[i] for i in meta["id"]}


def windows(X, L, S=None):
    """Окна длиной L с шагом S (по умолчанию S = L, подряд): (k, 19, L)."""
    S = S or L
    k = (X.shape[-1] - L) // S + 1 if X.shape[-1] >= L else 0
    if k <= 0:
        return np.zeros((0,) + X.shape[:-1] + (L,), X.dtype)
    return np.lib.stride_tricks.sliding_window_view(X, L, axis=-1)[..., ::S, :][..., :k, :].swapaxes(0, -2).copy()


def random_windows(X, L, K, rng):
    """K окон длиной L со случайным началом (любой отсчёт) — свежие окна на каждой эпохе: (K, 19, L)."""
    n = X.shape[-1] - L + 1
    if n <= 0:
        return np.zeros((0,) + X.shape[:-1] + (L,), X.dtype)
    st = np.sort(rng.integers(0, n, K))
    return np.stack([X[..., s:s + L] for s in st])


def resample(X, fs_from, fs_to):
    """Пересчёт частоты (полифазный, как scipy.signal.resample_poly) по последней оси, float32 на выходе."""
    if fs_from == fs_to:
        return X
    fr = Fraction(int(fs_to), int(fs_from))
    return ss.resample_poly(np.asarray(X, np.float32), fr.numerator, fr.denominator, axis=-1).astype(np.float32)


def highpass(X, f, fs=200, order=4):
    """Дофильтровка сверху общей нижней границы 0,1 Гц (например, 0,3 Гц как у CBraMod): по всей записи, не по окну."""
    sos = ss.butter(order, f, btype="highpass", fs=fs, output="sos")
    return ss.sosfiltfilt(sos, np.asarray(X, np.float64), axis=-1).astype(np.float32)
