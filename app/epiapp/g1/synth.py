"""Синтетическая ЭЭГ для демо-режима и тестов. Это НЕ запись пациента: шум со спектром 1/f, альфа-ритм в затылочных
отведениях, моргания, мышечные помехи, сеть 50 Гц. Пишется в настоящий EDF, чтобы демо шло тем же путём, что и запись врача."""
import numpy as np


from shared_data import EL

EXTRA = ["A1", "A2", "EKG1"]


def make_eeg(seconds, fs=250, seed=0, theta=0.0, flat=(), art_frac=0.05, mains=50.0, mains_uv=6.0):
    """(22, n) мкВ относительно условного референта: 19 каналов EL + A1, A2, EKG1; список участков артефактов [(t0, t1)].
    theta — добавка медленной активности 4–7 Гц (0..1): чем больше, тем выше оценка демо-модели.
    flat — каналы без сигнала; art_frac — доля записи с артефактами движения/мышц."""
    rng = np.random.default_rng(seed)
    n = int(seconds * fs)
    t = np.arange(n) / fs
    names = EL + EXTRA
    f = np.fft.rfftfreq(n, 1 / fs)
    shape = np.zeros_like(f); shape[1:] = 1 / np.maximum(f[1:], 0.5) ** 0.9
    shape[f > 70] *= 0.2
    spec = (rng.normal(size=(len(names), len(f))) + 1j * rng.normal(size=(len(names), len(f)))) * shape
    X = np.fft.irfft(spec, n, axis=-1)
    X = X / X.std(-1, keepdims=True) * 9.0                                   # фон ~9 мкВ
    common = np.fft.irfft((rng.normal(size=len(f)) + 1j * rng.normal(size=len(f))) * shape, n)
    X += common / common.std() * 4.0                                         # общая составляющая (референт)
    env = lambda period: 0.55 + 0.45 * np.sin(2 * np.pi * t / period + rng.uniform(0, 6.28))
    for i, name in enumerate(EL):
        post = {"O": 1.0, "P": 0.75, "T": 0.4, "C": 0.3, "F": 0.15}.get(name[0], 0.2)
        fa = 9.6 + rng.normal(0, 0.15)
        X[i] += 22.0 * post * env(rng.uniform(6, 14)) * np.sin(2 * np.pi * fa * t + rng.uniform(0, 6.28))
        if theta:
            ft = 5.5 + rng.normal(0, 0.3)
            X[i] += 26.0 * theta * (0.5 + 0.5 * env(rng.uniform(15, 40))) * np.sin(2 * np.pi * ft * t + rng.uniform(0, 6.28))
    # моргания: лобные отведения, раз в 4–9 с (обычная запись, артефактом качества не считаем)
    tb = 2.0
    while tb < seconds - 1:
        w = np.exp(-0.5 * ((t - tb) / 0.12) ** 2) * rng.uniform(60, 110)
        for name, k in (("Fp1", 1.0), ("Fp2", 1.0), ("F7", 0.35), ("F8", 0.35), ("F3", 0.3), ("F4", 0.3), ("Fz", 0.3)):
            X[EL.index(name)] += k * w
        tb += rng.uniform(4, 9)
    # участки артефактов: движение (медленный выброс большой амплитуды) + мышцы (высокочастотный шум)
    arts, total, guard = [], 0.0, 0
    while total < art_frac * seconds and guard < 10000:
        guard += 1
        d = rng.uniform(3, 14); a = rng.uniform(5, max(6, seconds - d - 2))
        if any(a < y + 3 and x < a + d + 3 for x, y in arts):
            continue
        arts.append((float(a), float(a + d))); total += d
        m = (t >= a) & (t <= a + d); k = int(m.sum())
        ch = rng.choice(19, rng.integers(5, 12), replace=False)
        X[np.ix_(ch, np.where(m)[0])] += rng.normal(0, 55, (len(ch), k)) + 260 * np.sin(np.pi * np.linspace(0, 1, k)) * rng.choice([-1, 1], (len(ch), 1))
    X += mains_uv * np.sin(2 * np.pi * mains * t)
    X[names.index("EKG1")] = 300 * (np.mod(t, 0.85) < 0.04)
    for name in flat:
        X[names.index(name)] = rng.normal(0, 0.05, n)
    return X, names, sorted(arts)


def write_edf(path, labels, fs, data_uv, patient="X X X DEMO", phys=3276.7):
    """Запись EDF: int16, блоки по 1 с, шаг ~0,1 мкВ. data_uv: (каналы, отсчёты)."""
    ns, n_rec, fs = len(labels), data_uv.shape[1] // int(fs), int(fs)
    f = lambda s, w: str(s).ljust(w)[:w].encode("ascii")
    hdr = (f("0", 8) + f(patient, 80) + f("Startdate 01-JAN-2026 X X X", 80) + f("01.01.26", 8) + f("10.00.00", 8) + f(256 * (ns + 1), 8)
           + f("", 44) + f(n_rec, 8) + f("1", 8) + f(ns, 4))
    for field, w in ((labels, 16), ([""] * ns, 80), (["uV"] * ns, 8), ([-phys] * ns, 8), ([phys] * ns, 8),
                     ([-32768] * ns, 8), ([32767] * ns, 8), ([""] * ns, 80), ([fs] * ns, 8), ([""] * ns, 32)):
        hdr += b"".join(f(x, w) for x in field)
    assert len(hdr) == 256 * (ns + 1)
    d = np.clip(np.round(data_uv / phys * 32767), -32768, 32767).astype("<i2")
    d = d[:, :n_rec * fs].reshape(ns, n_rec, fs).transpose(1, 0, 2).reshape(-1)
    with open(path, "wb") as fh:
        fh.write(hdr); fh.write(d.tobytes())


DEMO = {  # два демо-случая как в макете: обычная запись и запись с замечаниями
    "ok": dict(file="демо_ЭЭГ_004.edf", seconds=24 * 60 + 10, seed=4, theta=0.62, flat=(), art_frac=0.06),
    "bad": dict(file="демо_ЭЭГ_017.edf", seconds=12 * 60 + 40, seed=17, theta=0.42, flat=("T4", "O2"), art_frac=0.27),
}


def make_demo_edf(path, kind="ok", fs=250, seconds=None):
    """Синтетическая демо-запись в EDF (имена каналов «EEG Fp1-A1» — как у клинических приборов)."""
    p = DEMO[kind]
    X, names, arts = make_eeg(seconds or p["seconds"], fs, p["seed"], p["theta"], p["flat"], p["art_frac"])
    labels = [("EEG " + n + "-Ref") if n in EL or n.startswith("A") else n for n in names]
    write_edf(path, labels, fs, X)
    return arts
