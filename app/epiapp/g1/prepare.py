"""Чтение EDF, подготовка записи и проверка входа.

Подготовка — как у общего набора Г1 (h1_eeg/tuep/tuep_common.py, h1_eeg/src/shared_data.py): 19 каналов 10–20 в порядке EL,
200 Гц, средний референт, 0,1–75 Гц + режекция сети по записи целиком, мкВ, обрезка ±60 000.
Отличия от tuep_common.load_record (он рассчитан на TUH):
- имена каналов клинических приборов («EEG Fp1-A1», «Fp1-M1», «FP1») разбираются терпимее;
- каналы без сигнала не участвуют в среднем референте и обнуляются (при обучении моделей 1–2 канала обнулялись как аугментация);
- режекция сети — 50 Гц по умолчанию (настройка EPI_NOTCH_HZ), у TUH было 60 Гц;
- считается проверка входа: длительность, каналы, частота, доля артефактов, масштаб, сеть.
"""
import os, re
from dataclasses import dataclass, field
import numpy as np, scipy.signal as ss

import tuep_common as C
from shared_data import EL

TARGET = [e.upper() for e in EL]
REF_SUFFIX = {"REF", "LE", "AR", "AVG", "A1", "A2", "M1", "M2", "AA", "CAR", "G2", "N"}
ALIAS = {"T7": "T3", "T8": "T4", "P7": "T5", "P8": "T6"}
ELECTRODES = set(TARGET) | set(ALIAS) | {"A1", "A2", "M1", "M2", "OZ", "FPZ"}

# пороги проверки входа (см. README, «Проверка входа»)
MIN_RUN_S, WARN_SHORT_S, LONG_S = 60, 20 * 60, 60 * 60
FLAT_UV, MAX_FLAT_RUN = 0.5, 4
ART_PTP_UV, ART_HF_K, ART_WARN, ART_BAD = 300.0, 4.0, 0.20, 0.50
SCALE_LO_UV, SCALE_HI_UV = 2.0, 150.0


def norm_name(label):
    """«EEG Fp1-A1» / «FP1-REF» / «Fp1» -> («FP1», «A1»). Биполярные («Fp1-F3») и служебные каналы -> (None, None)."""
    c = label.upper().strip()
    c = re.sub(r"^(EEG|ЭЭГ)[\s_:\-]*", "", c).strip(" -_")
    parts = [p for p in re.split(r"[\s\-_]+", c) if p]
    if not parts:
        return None, None
    name, suf = ALIAS.get(parts[0], parts[0]), (parts[1] if len(parts) > 1 else "")
    if name not in TARGET and name not in ("A1", "A2", "M1", "M2"):
        return None, None
    if suf and suf not in REF_SUFFIX and ALIAS.get(suf, suf) in ELECTRODES:
        return None, None                                   # второй электрод — значит биполярное отведение
    return name, suf


def find_channels(labels):
    """Индексы 19 каналов в порядке EL (None, если канала нет), вид референта по суффиксам, список отсутствующих."""
    pos, sufs = {}, set()
    for i, l in enumerate(labels):
        name, suf = norm_name(l)
        if name in TARGET and name not in pos:
            pos[name] = i; sufs.add(suf or "—")
    idx = [pos.get(t) for t in TARGET]
    missing = [EL[k] for k, i in enumerate(idx) if i is None]
    return idx, sorted(sufs), missing


@dataclass
class PreparedRecord:
    """Подготовленная запись — вход любой модели Г1."""
    x: np.ndarray                      # (19, n) float32, мкВ, 200 Гц, порядок EL, средний референт, 0,1–75 Гц + режекция
    sfreq: int = 200
    channels: list = field(default_factory=lambda: list(EL))
    flat: list = field(default_factory=list)      # каналы без сигнала (обнулены)
    sfreq_orig: float = 0.0
    dur_s: float = 0.0
    artifacts: list = field(default_factory=list)  # [[t0, t1], …] с — показатель качества, не «разряды»


def _is_edf(path):
    with open(path, "rb") as f:
        b = f.read(8)
    if b[:1] == b"\xff":
        return "это BDF (24 бита): сохраните запись в формате EDF"
    if b[:1] != b"0":
        return "файл не похож на EDF (неверный заголовок)"
    return ""


def detect_artifacts(X, good):
    """Секунды с артефактами (приближённо): размах > 300 мкВ хотя бы в одном канале, либо высокочастотная составляющая (мышцы,
    движение) в ≥ 3 каналах в 4 раза выше обычного для этой записи, либо пропадание сигнала во всех каналах.
    Возвращает (маска по секундам, участки [[t0, t1]])."""
    n_s = X.shape[1] // 200
    if n_s == 0 or not len(good):
        return np.zeros(0, bool), []
    E = X[good, :n_s * 200].reshape(len(good), n_s, 200)
    ptp = E.max(-1) - E.min(-1)
    hf = np.abs(np.diff(E, axis=-1)).mean(-1)                       # грубая мера высоких частот
    hf_ref = np.median(hf, axis=1, keepdims=True) + 1e-6
    flag = (ptp > ART_PTP_UV).any(0) | ((hf > ART_HF_K * hf_ref).sum(0) >= 3) | (E.std(-1) < 0.05).all(0)
    iv, s = [], None
    for i, f in enumerate(flag):
        if f and s is None:
            s = i
        if not f and s is not None:
            iv.append([s, i]); s = None
    if s is not None:
        iv.append([s, n_s])
    return flag, iv


def _mains(Xraw, fs):
    """Какая сетевая помеха в сырой записи: 50 или 60 Гц (или None) — по спектру первых минут."""
    if fs < 130:
        return None
    seg = Xraw[:, :int(min(Xraw.shape[1], 120 * fs))]
    f, P = ss.welch(seg, fs, nperseg=int(4 * fs), axis=-1)
    P = np.median(P, 0)
    def peak(f0):
        m, b = (f > f0 - 1) & (f < f0 + 1), ((f > f0 - 6) & (f < f0 - 2)) | ((f > f0 + 2) & (f < f0 + 6))
        return P[m].max() / (np.median(P[b]) + 1e-12)
    p50, p60 = peak(50), peak(60)
    if max(p50, p60) < 5:
        return None
    return 50 if p50 >= p60 else 60


def _fmt_dur(s):
    return f"{int(s // 60)} мин {int(s % 60):02d} с"


def prepare_edf(path, notch_hz=50.0):
    """EDF -> (PreparedRecord или None, проверка входа).

    Проверка: {"items": [{"key", "title", "value", "note", "status": ok|warn|bad}], "can_run": bool, "low_quality": bool,
               "notes": [...], "flat": [...], "missing": [...], "artifacts": [[t0, t1]], "summary": {...}}.
    can_run = False — оценку получить нельзя (нет каналов, слишком короткая запись, файл не читается)."""
    items, notes = [], []
    def fail(title, value, note):
        items.append(dict(key="file", title=title, value=value, note=note, status="bad"))
        return None, dict(items=items, can_run=False, low_quality=True, notes=notes, flat=[], missing=[], artifacts=[], summary={})

    err = _is_edf(path)
    if err:
        return fail("Файл", "не читается", err)
    try:
        fs_hdr, n_hdr, labels = C.read_header(path)
    except Exception as e:
        return fail("Файл", "не читается", f"заголовок EDF: {type(e).__name__}: {str(e)[:120]}")
    idx, sufs, missing = find_channels(labels)
    if missing:
        items.append(dict(key="channels", title="Каналы 10–20", value=f"{19 - len(missing)} из 19", status="bad",
                          note="В файле нет каналов: " + ", ".join(missing) + ". Нужны все 19 каналов системы 10–20 (монополярная запись)."))
        return None, dict(items=items, can_run=False, low_quality=True, notes=["Каналы в файле: " + ", ".join(labels)[:400]],
                          flat=[], missing=missing, artifacts=[], summary={})
    try:
        Xraw, fs = C.read_edf_channels(path, idx)
    except Exception as e:
        return fail("Файл", "не читается", f"сигнал: {type(e).__name__}: {str(e)[:120]}")
    dur = Xraw.shape[1] / fs
    if dur < MIN_RUN_S:
        items.append(dict(key="duration", title="Длительность", value=_fmt_dur(dur), status="bad", note="Короче 1 минуты — оценка невозможна."))
        return None, dict(items=items, can_run=False, low_quality=True, notes=notes, flat=[], missing=[], artifacts=[], summary={})

    mains = _mains(Xraw, fs)
    X = C.to_fs200(Xraw, fs)
    del Xraw
    Xc = X - np.median(X, axis=-1, keepdims=True)
    flat = [EL[i] for i in range(19) if np.std(Xc[i]) < FLAT_UV or np.ptp(Xc[i]) < 4 * FLAT_UV]
    good = [i for i in range(19) if EL[i] not in flat]
    if len(flat) > MAX_FLAT_RUN:
        items.append(dict(key="channels", title="Каналы 10–20", value=f"{len(good)} из 19", status="bad",
                          note="Без сигнала: " + ", ".join(flat) + ". Слишком много каналов без сигнала — оценка невозможна."))
        return None, dict(items=items, can_run=False, low_quality=True, notes=notes, flat=flat, missing=[], artifacts=[], summary={})
    X = X - X[good].mean(0, keepdims=True)                            # средний референт по каналам с сигналом
    X = C.filt(X, notch=notch_hz)
    for i in range(19):
        if EL[i] in flat:
            X[i] = 0.0
    X = np.clip(X, -C.F16_MAX, C.F16_MAX).astype(np.float32)
    flag, arts = detect_artifacts(X, good)
    art_frac = float(flag.mean()) if len(flag) else 0.0
    scale = float(np.median(X[good].std(-1)))

    # ---- карточки проверки (как в макете)
    st = "ok" if dur >= WARN_SHORT_S else "warn"
    items.append(dict(key="duration", title="Длительность", value=_fmt_dur(dur), status=st,
                      note=("Короче 20 мин — оценка менее надёжна" if dur < WARN_SHORT_S else
                            "Длинная запись: оценка по 60 окнам, взятым равномерно" if dur > LONG_S else "20 минут и больше")))
    items.append(dict(key="channels", title="Каналы 10–20", value=f"{len(good)} из 19", status="bad" if flat else "ok",
                      note=(", ".join(flat) + " без сигнала") if flat else "Все на месте"))
    fs_r = int(round(fs))
    items.append(dict(key="sfreq", title="Частота оцифровки", value=f"{fs_r} Гц", status="ok" if fs_r >= 200 else "warn",
                      note=("Будет приведена к 200 Гц" if fs_r > 200 else "Совпадает с рабочей частотой модели" if fs_r == 200 else
                            "Ниже 200 Гц: верхние частоты в записи отсутствуют")))
    st = "ok" if art_frac < ART_WARN else "warn" if art_frac < ART_BAD else "bad"
    items.append(dict(key="artifacts", title="Артефакты", value=f"≈ {round(100 * art_frac)} % записи", status=st,
                      note={"ok": "Допустимый уровень", "warn": "Много движений и мышечных помех", "bad": "Большая часть записи с помехами"}[st]))
    if not SCALE_LO_UV <= scale <= SCALE_HI_UV:
        items.append(dict(key="scale", title="Масштаб сигнала", value=f"СКО {scale:.1f} мкВ".replace(".", ","), status="warn",
                          note="Необычная амплитуда: проверьте единицы измерения в файле"))
    if mains and abs(mains - notch_hz) > 1:
        notes.append(f"В записи видна сетевая помеха {mains} Гц, а режекция настроена на {notch_hz:g} Гц (настройка EPI_NOTCH_HZ).")
    if {"A1", "A2", "M1", "M2"} & set(sufs) and len(sufs) > 1:
        notes.append("Каналы записаны относительно разных ушных электродов — пересчёт на средний референт приближённый.")
    low = bool(flat) or dur < WARN_SHORT_S or art_frac >= ART_WARN
    rec = PreparedRecord(x=X, flat=flat, sfreq_orig=float(fs), dur_s=float(X.shape[1] / 200), artifacts=arts)
    why = []
    if dur < WARN_SHORT_S: why.append("короткая запись")
    if flat: why.append(f"{len(flat)} {'канал' if len(flat) == 1 else 'канала'} без сигнала")
    if art_frac >= ART_WARN: why.append("много артефактов")
    return rec, dict(items=items, can_run=True, low_quality=low, low_reasons=why, notes=notes, flat=flat, missing=[], artifacts=arts,
                     summary=dict(dur_s=float(dur), dur_text=_fmt_dur(dur), n_good=len(good), sfreq=fs_r, art_pct=round(100 * art_frac),
                                  reference=", ".join(sufs), scale_uv=round(scale, 1), mains=mains))


# ---------------------------------------------------------------- хранение подготовленной записи и куски для просмотра
def save_record(rec, path):
    np.save(path, rec.x.astype(np.float16))


def load_record(path, check):
    x = np.load(path).astype(np.float32)
    return PreparedRecord(x=x, flat=check.get("flat", []), sfreq_orig=check.get("summary", {}).get("sfreq", 0),
                          dur_s=x.shape[1] / 200, artifacts=check.get("artifacts", []))


BIPOLAR = [("Fp1", "F7"), ("F7", "T3"), ("T3", "T5"), ("T5", "O1"), ("Fp1", "F3"), ("F3", "C3"), ("C3", "P3"), ("P3", "O1"),
           ("Fz", "Cz"), ("Cz", "Pz"),
           ("Fp2", "F4"), ("F4", "C4"), ("C4", "P4"), ("P4", "O2"), ("Fp2", "F8"), ("F8", "T4"), ("T4", "T6"), ("T6", "O2")]
VIEW_ORDER = ["Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "T3", "C3", "Cz", "C4", "T4", "T5", "P3", "Pz", "P4", "T6", "O1", "O2"]   # как в макете


def view_chunk(x, t0, sec, montage="avg", hp=0.5, lp=70.0, flat=()):
    """Кусок сигнала для просмотрщика: (имена, (каналы, отсчёты) int16 в 0,1 мкВ, частота). Фильтр — по куску с полями ±3 с.
    montage: avg — средний референт (как видит модель), bip — продольный биполярный."""
    n = x.shape[1]
    a, b = int(max(0, t0) * 200), int(min(n, (t0 + sec) * 200))
    pa, pb = max(0, a - 600), min(n, b + 600)
    seg = x[:, pa:pb].astype(np.float64)
    if seg.shape[1] > 60:
        if hp:
            seg = ss.sosfiltfilt(ss.butter(2, hp, btype="highpass", fs=200, output="sos"), seg, axis=-1)
        if lp and lp < 95:
            seg = ss.sosfiltfilt(ss.butter(4, lp, btype="lowpass", fs=200, output="sos"), seg, axis=-1)
    seg = seg[:, a - pa:a - pa + (b - a)]
    if montage == "bip":
        names = [f"{p}-{q}" for p, q in BIPOLAR]
        data = np.stack([seg[EL.index(p)] - seg[EL.index(q)] for p, q in BIPOLAR])
        bad = [f"{p}-{q}" for p, q in BIPOLAR if p in flat or q in flat]
    else:
        names = VIEW_ORDER
        data = np.stack([seg[EL.index(c)] for c in VIEW_ORDER])
        bad = [c for c in VIEW_ORDER if c in flat]
    return names, np.clip(np.round(data * 10), -32767, 32767).astype(np.int16), 200, bad
