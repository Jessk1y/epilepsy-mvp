"""Общее для работы с TUEP: разбор путей, имена каналов TUH, чтение EDF и подготовка записи.

Подготовка — как у общего набора Г1 (h1_eeg/src/shared_data.py, блокнот h1_2026-10-26_zs_prep-shared.ipynb):
19 каналов в порядке EL, средний референт по 19 каналам, 200 Гц, Баттерворт 4-го порядка 0,1–75 Гц
(sosfiltfilt) + режекция сети (iirnotch, Q = 30, filtfilt) по записи целиком, мкВ, обрезка ±60 000, float16.
Отличие: режекция 60 Гц (США), а не 50 Гц.
"""
import os, re, sys
from fractions import Fraction
import numpy as np, scipy.signal as ss

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from shared_data import EL  # noqa: E402  (тот же порядок каналов, что у общего набора)

SFREQ, BAND, NOTCH_HZ, NOTCH_Q, F16_MAX = 200, (0.1, 75.0), 60.0, 30, 60000.0
ALIAS = {"T7": "T3", "T8": "T4", "P7": "T5", "P8": "T6"}
TARGET = [e.upper() for e in EL]
MONTAGES = ("01_tcp_ar", "02_tcp_le", "03_tcp_ar_a")

# patient: 8 букв (aaaaaaaa), сеанс: s001_2012 или s001, файл: aaaaaaaa_s001_t000.edf
RE_FILE = re.compile(r"^(?P<patient>[a-z]{8})_(?P<session>s\d{3})_(?P<tok>t\d{3})\.edf$", re.I)


def norm_channel(label):
    """«EEG FP1-REF» / «EEG FP1-LE» / «Fp1» -> «FP1». Суффикс референта отбрасывается, T7/T8/P7/P8 -> T3/T4/T5/T6."""
    c = label.upper().strip()
    c = re.sub(r"^EEG[\s_-]*", "", c)
    c = re.sub(r"[-_ ](REF|LE|AR|AVG|A1|A2)$", "", c)
    c = c.strip(" -_")
    return ALIAS.get(c, c)


def ref_kind(labels):
    """Что стоит в суффиксах каналов: REF / LE / другое — для таблицы описи."""
    ends = {re.sub(r"^.*[-_ ]", "", l.upper().strip()) for l in labels if re.search(r"EEG", l.upper())}
    return "|".join(sorted(ends & {"REF", "LE", "AR", "AVG"})) or "?"


def channel_index(labels):
    """Индексы 19 каналов в порядке EL внутри списка имён файла; None, если какого-то нет. Лишние каналы (A1, EKG, PHOTIC…) игнорируются."""
    pos = {}
    for i, l in enumerate(labels):
        pos.setdefault(norm_channel(l), i)
    try:
        return [pos[t] for t in TARGET]
    except KeyError:
        return None


def parse_path(path):
    """Пациент / сеанс / файл / монтаж / группа по пути. Группа: часть пути «epilepsy» или «no_epilepsy»; нет — None (возьмём из таблицы)."""
    p = os.path.normpath(path)
    parts = [x.lower() for x in p.split(os.sep)]
    m = RE_FILE.match(os.path.basename(p))
    group = None
    for x in parts[:-1]:
        if x in ("epilepsy", "epi", "tuep_epilepsy"):
            group = 1
        elif x in ("no_epilepsy", "non_epilepsy", "no_epi", "nonepilepsy", "control", "tuep_no_epilepsy"):
            group = 0
    montage = next((x for x in parts[:-1] if x in MONTAGES), None)
    if m is None:
        return dict(patient=None, session=None, file=os.path.basename(p), montage=montage, group=group)
    # папка сеанса: s001_2012 (с годом) — берём её целиком, чтобы сеансы разных лет не слипались
    sess_dir = next((x for x in parts[:-1] if re.match(r"^s\d{3}(_\d{4})?$", x)), m["session"].lower())
    return dict(patient=m["patient"].lower(), session=sess_dir, file=os.path.basename(p), montage=montage, group=group)


def edf_header(f):
    """Заголовок EDF (стандарт EDF, без mne — чтобы не тянуть зависимости): словарь с метками, частотами, масштабами."""
    h = f.read(256)
    ns = int(h[252:256]); n_rec = int(h[236:244]); rec_dur = float(h[244:252])
    g = lambda w: [f.read(w).decode("latin-1").strip() for _ in range(ns)]
    labels, _tr, dim = g(16), g(80), g(8)
    pmin, pmax, dmin, dmax = ([float(x) for x in g(8)] for _ in range(4))
    _pf = g(80)
    spr = [int(x) for x in g(8)]
    f.read(32 * ns)
    return dict(ns=ns, n_rec=n_rec, rec_dur=rec_dur, labels=labels, dim=dim, pmin=pmin, pmax=pmax, dmin=dmin, dmax=dmax, spr=spr, hdr_bytes=256 * (ns + 1))


def read_header(path):
    """(частота, число отсчётов, список имён каналов) без чтения сигнала. Частота — по первому каналу ЭЭГ."""
    with open(path, "rb") as f:
        H = edf_header(f)
    n_rec = H["n_rec"]
    if n_rec < 0:                                           # у части файлов число записей не заполнено
        n_rec = (os.path.getsize(path) - H["hdr_bytes"]) // (2 * sum(H["spr"]))
    i = next((k for k, l in enumerate(H["labels"]) if "EDF ANNOTATIONS" not in l.upper()), 0)
    fs = H["spr"][i] / H["rec_dur"]
    return float(fs), int(n_rec * H["spr"][i]), [l for l in H["labels"]]


def read_edf_channels(path, idx):
    """Сигнал выбранных каналов (список индексов) в мкВ: (len(idx), n) float64 и частота. Все выбранные каналы должны иметь одну частоту."""
    with open(path, "rb") as f:
        H = edf_header(f)
        n_rec = H["n_rec"] if H["n_rec"] >= 0 else (os.path.getsize(path) - H["hdr_bytes"]) // (2 * sum(H["spr"]))
        rates = {H["spr"][i] for i in idx}
        if len(rates) != 1:
            raise ValueError(f"у выбранных каналов разные частоты: {sorted(rates)}")
        raw = np.fromfile(f, dtype="<i2", count=n_rec * sum(H["spr"])).reshape(n_rec, sum(H["spr"]))
    off = np.concatenate([[0], np.cumsum(H["spr"])])
    out = []
    for i in idx:
        d = raw[:, off[i]:off[i + 1]].reshape(-1).astype(np.float64)
        gain = (H["pmax"][i] - H["pmin"][i]) / (H["dmax"][i] - H["dmin"][i])
        p = (d - H["dmin"][i]) * gain + H["pmin"][i]
        dim = H["dim"][i].lower().replace("µ", "u").replace("μ", "u")
        out.append(p * {"uv": 1.0, "mv": 1e3, "v": 1e6, "": 1.0}.get(dim, 1.0))
    return np.stack(out), H["spr"][idx[0]] / H["rec_dur"]


def _filters(fs=SFREQ, notch=NOTCH_HZ):
    sos = ss.butter(4, list(BAND), btype="bandpass", fs=fs, output="sos")
    b, a = ss.iirnotch(notch, NOTCH_Q, fs=fs)
    return sos, b, a


def filt(X, notch=NOTCH_HZ, extra_notch=()):
    """0,1–75 Гц + режекция сети по записи целиком (как filt в блокноте общего набора, но 60 Гц)."""
    sos, b, a = _filters(notch=notch)
    Y = ss.sosfiltfilt(sos, X.astype(np.float64), axis=-1)
    Y = ss.filtfilt(b, a, Y, axis=-1)
    for f in extra_notch:                       # напр. 50 Гц — чтобы «дырка» была и там, как при обучении (по желанию)
        b2, a2 = ss.iirnotch(f, NOTCH_Q, fs=SFREQ)
        Y = ss.filtfilt(b2, a2, Y, axis=-1)
    return Y


def to_fs200(X, fs):
    """Пересчёт в 200 Гц полифазным фильтром (scipy.signal.resample_poly), как в shared_data.resample."""
    if int(round(fs)) == SFREQ:
        return np.asarray(X, np.float64)
    fr = Fraction(SFREQ, int(round(fs)))
    return ss.resample_poly(np.asarray(X, np.float64), fr.numerator, fr.denominator, axis=-1)


def load_record(path, extra_notch=()):
    """EDF -> (X float16 (19, n) мкВ, словарь сведений) или (None, {'error': …}).

    Порядок: 19 каналов EL -> мкВ -> 200 Гц -> средний референт по 19 -> фильтры по записи целиком -> обрезка ±60 000 -> float16.
    """
    try:
        _, _, labels = read_header(path)
        idx = channel_index(labels)
        if idx is None:
            return None, dict(error="нет всех 19 каналов 10–20: " + "|".join(labels)[:200])
        X, fs = read_edf_channels(path, idx)
    except Exception as e:  # битые заголовки у TUH встречаются
        return None, dict(error=f"чтение: {type(e).__name__}: {str(e)[:150]}")
    if X.shape[1] < 10 * int(fs):
        return None, dict(error=f"короче 10 с ({X.shape[1]} отсчётов)")
    X = to_fs200(X, fs)
    X = X - X.mean(0, keepdims=True)           # средний референт по 19 каналам (как у турецкого набора)
    std_raw = float(np.std(X))
    Xf = filt(X, extra_notch=extra_notch)
    clipped = int((np.abs(Xf) > F16_MAX).sum())
    return np.clip(Xf, -F16_MAX, F16_MAX).astype(np.float16), dict(
        sfreq_orig=fs, dur_full_s=X.shape[1] / SFREQ, dur_s=X.shape[1] / SFREQ, std_uv=float(np.std(Xf)), std_raw_uv=std_raw, clipped=clipped)
