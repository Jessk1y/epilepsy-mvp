"""Лёгкий локальный тест на синтетических EDF (настоящих данных нет): py -I test_synthetic.py

Создаёт во временной папке структуру в духе TUEP — 4 пациента, 2 группы, монтажи 01_tcp_ar (…-REF) и 02_tcp_le (…-LE),
частоты 250 и 256 Гц, лишние каналы (A1, EKG), имена T3/T4/T5/T6 — и прогоняет опись -> подготовку -> фолды -> оценку.
Модели (torch, веса) тут не нужны: оценки подставляются случайные. Нагрузка — секунды, ~20 МБ.
"""
import os, subprocess, sys, tempfile
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")   # -I игнорирует PYTHONIOENCODING (Windows)
sys.path.insert(0, HERE)
import tuep_common as C

LABELS_REF = ["EEG FP1-REF", "EEG FP2-REF", "EEG F3-REF", "EEG F4-REF", "EEG C3-REF", "EEG C4-REF", "EEG P3-REF", "EEG P4-REF", "EEG O1-REF",
              "EEG O2-REF", "EEG F7-REF", "EEG F8-REF", "EEG T3-REF", "EEG T4-REF", "EEG T5-REF", "EEG T6-REF", "EEG A1-REF", "EEG A2-REF",
              "EEG FZ-REF", "EEG CZ-REF", "EEG PZ-REF", "EEG PHOTIC-REF", "EEG EKG1-REF"]


def write_edf(path, labels, fs, seconds, data_uv):
    """Минимальная запись EDF: int16, записи по 1 с, физ. диапазон ±3200 мкВ (шаг ~0,1 мкВ)."""
    ns, n_rec = len(labels), int(seconds)
    f = lambda s, w: s.ljust(w)[:w].encode("ascii")
    hdr = (f("0", 8) + f("X X X X", 80) + f("Startdate 01-JAN-2020 X X X", 80) + f("01.01.20", 8) + f("00.00.00", 8) + f(str(256 * (ns + 1)), 8)
           + f("", 44) + f(str(n_rec), 8) + f("1", 8) + f(str(ns), 4))
    hdr += b"".join(f(l, 16) for l in labels) + b"".join(f("", 80) for _ in labels) + b"".join(f("uV", 8) for _ in labels)
    hdr += b"".join(f("-3200", 8) for _ in labels) + b"".join(f("3200", 8) for _ in labels)
    hdr += b"".join(f("-32768", 8) for _ in labels) + b"".join(f("32767", 8) for _ in labels)
    hdr += b"".join(f("", 80) for _ in labels) + b"".join(f(str(fs), 8) for _ in labels) + b"".join(f("", 32) for _ in labels)
    assert len(hdr) == 256 * (ns + 1)
    d = np.clip(np.round(data_uv / 3200 * 32767), -32768, 32767).astype("<i2")          # (ns, n_rec*fs)
    d = d[:, :n_rec * fs].reshape(ns, n_rec, fs).transpose(1, 0, 2).reshape(-1)         # запись за записью, канал за каналом
    with open(path, "wb") as fh:
        fh.write(hdr); fh.write(d.tobytes())


def make(root):
    rng = np.random.default_rng(0)
    plan = [("aaaaaaaa", "epilepsy", "s001_2015", "01_tcp_ar", 250, "REF", 40), ("aaaaaaaa", "epilepsy", "s002_2016", "02_tcp_le", 256, "LE", 40),
            ("bbbbbbbb", "epilepsy", "s001_2014", "01_tcp_ar", 250, "REF", 35), ("cccccccc", "no_epilepsy", "s001_2013", "01_tcp_ar", 256, "REF", 40),
            ("dddddddd", "no_epilepsy", "s001_2012", "02_tcp_le", 250, "LE", 40), ("dddddddd", "no_epilepsy", "s001_2012", "02_tcp_le", 250, "LE", 8)]
    for k, (pat, grp, sess, mont, fs, ref, sec) in enumerate(plan):
        labels = [l.replace("-REF", "-" + ref) for l in LABELS_REF]
        n = sec * fs
        t = np.arange(n) / fs
        common = 20 * np.sin(2 * np.pi * 10 * t)                                          # альфа 20 мкВ
        data = common + rng.normal(0, 8, (len(labels), n)) + 15 * np.sin(2 * np.pi * 60 * t)   # + сеть 60 Гц
        tok = f"t{(k % 2):03d}" if (pat, sess) != ("dddddddd", "s001_2012") else f"t{k - 4:03d}"
        d = os.path.join(root, "edf", grp, pat, sess, mont); os.makedirs(d, exist_ok=True)
        write_edf(os.path.join(d, f"{pat}_{sess[:4]}_{tok}.edf"), labels, fs, sec, data)
        lab = "seiz" if (pat, sess) == ("aaaaaaaa", "s002_2016") else "bckg"           # приступ — во втором сеансе aaaaaaaa
        with open(os.path.join(d, f"{pat}_{sess[:4]}_{tok}.csv_bi"), "w") as fh:
            fh.write(f"# version = csv_v1.0.0\n#\nchannel,start_time,stop_time,label,confidence\nTERM,0.0000,{sec}.0000,{lab},1.0000\n")


def run(*args):
    print("$", " ".join(os.path.basename(str(x)) if os.sep in str(x) else str(x) for x in args))
    r = subprocess.run([sys.executable, "-X", "utf8", "-I", *map(str, args)], capture_output=True, text=True, encoding="utf-8", errors="replace", env={**os.environ, "PYTHONIOENCODING": "utf-8"})
    print(r.stdout[-1500:]); assert r.returncode == 0, r.stderr[-2000:]
    return r.stdout


def main():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        root, work = os.path.join(tmp, "v3.1.0"), os.path.join(tmp, "work")
        make(root)
        # 1. разбор имён и каналов
        assert C.norm_channel("EEG FP1-REF") == "FP1" and C.norm_channel("EEG T3-LE") == "T3" and C.norm_channel("EEG CZ-REF") == "CZ"
        # 2. опись
        run(os.path.join(HERE, "tuep_inventory.py"), "--root", root, "--out", os.path.join(work, "inv"), "--threads", 2)
        inv = pd.read_csv(os.path.join(work, "inv", "inventory_files.csv"))
        assert len(inv) == 6 and inv.has19.all() and inv.patient.nunique() == 4 and set(inv.group) == {0, 1}, inv
        assert set(inv.montage) == {"01_tcp_ar", "02_tcp_le"} and set(inv.sfreq) == {250.0, 256.0}
        assert inv.n_seiz.sum() == 1 and set(inv.year) == {2012, 2013, 2014, 2015, 2016}, inv[["patient", "session", "n_seiz", "year"]]
        # 3. подготовка
        run(os.path.join(HERE, "tuep_prepare.py"), "--root", root, "--out", os.path.join(work, "prep"), "--threads", 2)
        meta = pd.read_csv(os.path.join(work, "prep", "h1_shared_meta.csv"))
        failed = pd.read_csv(os.path.join(work, "prep", "h1_shared_failed.csv"))
        assert len(meta) == 5 and len(failed) == 1 and "короче 10 с" in failed.error.iloc[0], (meta.shape, failed)
        print("не подготовлены:", failed.to_dict("records"))
        # 4. читается теми же функциями общего набора
        from shared_data import load_meta, read_record, read_all
        m2, info = load_meta(os.path.join(work, "prep"))
        X = read_record(os.path.join(work, "prep"), m2.iloc[0].to_dict())
        assert X.shape == (19, 40 * 200) and X.dtype == np.float16, X.shape
        allr = read_all(os.path.join(work, "prep"), m2, threads=1)
        assert len(allr) == len(m2)
        # сеть 60 Гц подавлена (было 15 мкВ), фон сохранён, референт средний
        f, P = __import__("scipy.signal").signal.welch(X.astype(np.float64), 200, nperseg=400)
        p60, p55 = P[:, np.argmin(abs(f - 60))].mean(), P[:, np.argmin(abs(f - 55))].mean()
        print(f"мощность на 60 Гц / на 55 Гц = {p60 / p55:.2f} (было бы ~{1 + (15 ** 2 / 2) / (8 ** 2 / 100):.0f} без режекции); СКО {X.astype(np.float64).std():.1f} мкВ; "
              f"среднее по каналам {abs(X.astype(float).mean(0)).max():.3f}")
        assert p60 / p55 < 0.5 and abs(X.astype(float).mean(0)).max() < 0.5
        # 5. фолды: пациенты не делятся
        run(os.path.join(HERE, "tuep_protocol.py"), "folds", "--meta", os.path.join(work, "prep", "h1_shared_meta.csv"), "--out", os.path.join(work, "prot", "folds.csv"), "--n-folds", 2)
        run(os.path.join(HERE, "tuep_protocol.py"), "leakage", "--out", os.path.join(work, "prot", "leak.csv"))
        # 6. оценка на случайных «оценках записей» (проверка механики; пороги берутся из ../results)
        rng = np.random.default_rng(1)
        sc = meta[["id", "patient", "session", "y", "dur_s"]].copy()
        for c in ["reve4", "dino2"]:
            sc[c] = np.clip(0.3 + 0.4 * sc.y + rng.normal(0, 0.15, len(sc)), 0, 1)
        sc["ens"] = (sc.reve4 + sc.dino2) / 2
        sp = os.path.join(work, "scores.csv"); sc.to_csv(sp, index=False)
        res = os.path.join(HERE, "..", "results")
        if os.path.exists(os.path.join(res, "eegdino_v2_per_person.csv")):
            run(os.path.join(HERE, "tuep_protocol.py"), "eval", "--scores", sp, "--meta", os.path.join(work, "prep", "h1_shared_meta.csv"),
                "--folds", os.path.join(work, "prot", "folds.csv"), "--results", res, "--out", os.path.join(work, "ev"), "--calib-k", 2)
            pt = pd.read_csv(os.path.join(work, "ev", "tuep_patient_scores.csv"))
            # один сеанс на пациента; у aaaaaaaa второй сеанс с приступом исключён -> остаётся первый
            assert len(pt) == 4 and (pt.n_sessions == 1).all(), pt
            assert pt.set_index("patient").session["aaaaaaaa"] == "s001_2015", pt
            assert os.path.exists(os.path.join(work, "ev", "tuep_confounds.csv"))
        else:
            print("results/ нет — шаг eval пропущен")
        print("ТЕСТ ПРОЙДЕН")


if __name__ == "__main__":
    main()
