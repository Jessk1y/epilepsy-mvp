"""Лёгкий тест схемы дообучения на синтетических данных: py -I test_finetune_synthetic.py  (процессор, секунды, без весов и интернета)

Создаёт во временной папке «подготовленный TUEP» (формат tuep_prepare.py): 40 пациентов, 2 класса, 1–4 сеанса, монтажи и частоты
с перекосом по классу (слой 03_tcp_ar_a|512 только у больных), один сеанс с приступом. У больных в сигнал добавлен ритм 6 Гц.
Проверяет: веса сеансов и исключение «утекающего» слоя; обучение игрушечной сети по 5 фолдам; продолжение после обрыва;
эталон (мощности ритмов) и контроль «без ЭЭГ»; сводный отчёт — один пациент = одна оценка, тест не пересекается с обучением.
"""
import json, os, subprocess, sys, tempfile
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, HERE)
import tuep_finetune as FT
from tuep_prepare import Shards


def make(prep):
    rng = np.random.default_rng(0)
    W, rows = Shards(prep), []
    for i in range(40):
        y = i % 2
        pat = "p" + "abcdefghij"[i // 10] + "abcdefghij"[i % 10] + "aaaaa"
        for s in range(int(rng.integers(1, 5)) if y else int(rng.integers(1, 3))):
            if y and i < 12 and s == 0:
                mont, fs = "03_tcp_ar_a", 512.0                          # слой только у больных — в обучение идти не должен
            else:
                mont, fs = ("01_tcp_ar", 256.0 if rng.random() < (0.7 if y else 0.4) else 250.0)
            sess = f"s{s+1:03d}_{2010 + s}"
            seiz = int(i == 0 or (i == 3 and s == 1))                    # у пациента 0 приступы во всех сеансах — он должен выпасть
            for t in range(int(rng.integers(1, 3))):
                n = int(rng.integers(30, 60)) * 200
                tt = np.arange(n) / 200
                X = rng.normal(0, 10, (19, n)) + 8 * np.sin(2 * np.pi * 10 * tt)
                if y:
                    X += 14 * np.sin(2 * np.pi * 6 * tt + rng.uniform(0, 6, (19, 1)))
                X = X - X.mean(0, keepdims=True)
                sh, off, nn = W.add(X.astype(np.float16))
                rows.append(dict(dataset="tuep", part="tuep", split="tuep", id=f"{pat}_{sess[:4]}_t{t:03d}", y=y, label="epilepsy" if y else "no_epilepsy",
                                 sfreq_orig=fs, dur_full_s=n / 200, dur_s=n / 200, scale_k=1.0, std_uv=float(X.std()), std_raw_uv=float(X.std()), clipped=0,
                                 patient=pat, session=sess, montage=mont, year=2010 + s, n_seiz=seiz if t == 0 else 0, shard200=sh, off200=off, n200=nn))
    W.flush()
    pd.DataFrame(rows).to_csv(os.path.join(prep, "h1_shared_meta.csv"), index=False)
    json.dump(dict(fs=[200]), open(os.path.join(prep, "h1_shared_info.json"), "w"))


def run(*args, ok=(0,)):
    print("$", " ".join(os.path.basename(str(x)) if os.sep in str(x) else str(x) for x in args))
    r = subprocess.run([sys.executable, "-X", "utf8", "-I", *map(str, args)], capture_output=True, text=True, encoding="utf-8", errors="replace")
    print(r.stdout[-1200:]); assert r.returncode in ok, r.stderr[-3000:]
    return r


def main():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        prep, out = os.path.join(tmp, "prep"), os.path.join(tmp, "ft")
        os.makedirs(prep); make(prep)
        meta = FT.load_prep(prep); S = FT.session_frame(meta); F = FT.get_folds(meta)
        chosen, dropped = FT.chosen_sessions(S)
        assert len(chosen) + len(dropped) == 40 and all(not S.set_index(["patient", "session"]).has_seiz[(p, s)] for p, s in chosen.items())
        # веса сеансов: сумма 1, классы поровну, слой только-у-больных и сеансы с приступами — 0, слой не зависит от класса
        fit_p, val_p, test_p = FT.split_patients(F, 0)
        assert len(fit_p) == 24 and len(val_p) == 8 and len(test_p) == 8
        Sf = S[S.patient.isin(fit_p)].reset_index(drop=True)
        w, tab = FT.session_weights(Sf, "strata", 3)
        print(tab.to_string())
        assert abs(w.sum() - 1) < 1e-9 and abs(w[Sf.y == 1].sum() - 0.5) < 1e-9
        assert w[Sf.stratum == "03_tcp_ar_a|512"].sum() == 0 and w[Sf.has_seiz].sum() == 0
        for st in Sf.stratum.unique():
            assert abs(w[(Sf.stratum == st) & (Sf.y == 1)].sum() - w[(Sf.stratum == st) & (Sf.y == 0)].sum()) < 1e-9, st
        wc, _ = FT.session_weights(Sf, "class")
        assert wc[Sf.stratum == "03_tcp_ar_a|512"].sum() > 0 and abs(wc[Sf.y == 1].sum() - 0.5) < 1e-9
        # окно с полями совпадает по форме; фильтр убирает постоянную составляющую
        r0 = next(meta.itertuples())
        x = FT.read_window(prep, r0.shard200, int(r0.off200), int(r0.n200), 0, FT.make_sos(45))
        assert x.shape == (19, FT.L) and abs(x.mean()) < 1.0
        # обучение игрушечной сети: обрыв после 1-й эпохи первого запуска -> продолжение -> остальные фолды
        base = [os.path.join(HERE, "tuep_finetune.py"), "train", "--model", "toy", "--prep", prep, "--out", out, "--seeds", 0, 1,
                "--max-epochs", 3, "--epoch-windows", 192, "--bs", 16, "--workers", 2, "--device", "cpu"]
        run(*base, "--fold-list", 0, "--debug-stop-epoch", 1, ok=(3,))
        assert os.path.exists(os.path.join(out, "ckpt", "toy_f0_s0.pt")) and not os.path.exists(os.path.join(out, "runs", "toy_f0_s0.csv"))
        r = run(*base)
        assert "продолжение после эпохи 1" in r.stdout
        assert len([f for f in os.listdir(os.path.join(out, "runs")) if f.endswith(".csv")]) == 10 and not os.listdir(os.path.join(out, "ckpt"))
        h = json.load(open(os.path.join(out, "runs", "toy_f0_s0.json"), encoding="utf-8"))
        assert [e["epoch"] for e in h["hist"]] == list(range(1, len(h["hist"]) + 1)), h["hist"]
        r2 = run(*base)                                                       # повторный запуск ничего не пересчитывает
        assert r2.stdout.count("готово, пропуск") == 10
        run(*base[:8], "--smoke", "--bs", 16, "--workers", 2, "--device", "cpu")   # пробный режим — в отдельную папку
        assert os.path.exists(os.path.join(out, "smoke", "runs", "toy_f0_s0.csv"))
        # эталон и контроль без ЭЭГ
        run(os.path.join(HERE, "tuep_baseline.py"), "--prep", prep, "--out", out, "--threads", 2)
        # отчёт
        zs = pd.DataFrame({"patient": sorted(chosen), "ens": 0.5}); zs["y"] = zs.patient.map(F.set_index("patient").y)
        zs["ens"] = np.random.default_rng(1).random(len(zs)); zp = os.path.join(tmp, "zs.csv"); zs.to_csv(zp, index=False)
        run(os.path.join(HERE, "tuep_finetune.py"), "report", "--prep", prep, "--out", out, "--zero-shot", zp, "--ensemble", "toy", "bp")
        P = pd.read_csv(os.path.join(out, "tuep_ft_patient_scores.csv")); T = pd.read_csv(os.path.join(out, "tuep_ft_metrics.csv"))
        assert len(P) == len(chosen) and P.patient.is_unique and P.toy_score.notna().all() and P.bp_score.notna().all(), P.shape
        assert set(T["код"]) == {"toy", "bp", "bpamp", "tech", "toy+bp"}, T["код"].tolist()
        assert (T.n == len(chosen)).all() and (T["фолдов"] == 5).all()
        auc = {c: float(v.split()[0]) for c, v in zip(T["код"], T["ROC AUC (95 % ДИ)"])}
        print("ROC AUC на синтетике:", auc)
        assert auc["bp"] > 0.9, auc                                           # ритм 6 Гц эталон обязан найти
        assert auc["toy"] > 0.6, auc
        # тест фолда не участвовал в обучении: каждый пациент встречается как test ровно в одном фолде, как val — в одном другом
        d = pd.concat(pd.read_csv(os.path.join(out, "runs", f)) for f in os.listdir(os.path.join(out, "runs")) if f.startswith("toy_") and f.endswith(".csv"))
        g = d.groupby(["patient", "split"]).fold.nunique().unstack()
        assert (g.test == 1).all() and (g.val.dropna() == 1).all()
        tv = d.groupby("patient").apply(lambda q: set(q.fold[q.split == "test"]) & set(q.fold[q.split == "val"]))
        assert all(len(v) == 0 for v in tv)
        md = open(os.path.join(out, "tuep_ft_summary.md"), encoding="utf-8").read()
        assert not any(p in md for p in chosen), "в сводке не должно быть кодов пациентов"
        # проверка данных (tuep_audit.py) на подставных оценках
        os.makedirs(os.path.join(tmp, "out"))
        rs = meta[["id", "patient", "session", "y", "dur_s"]].copy(); rg = np.random.default_rng(2)
        rs["reve4"], rs["dino2"] = rg.random(len(rs)), rg.random(len(rs)); rs["ens"] = (rs.reve4 + rs.dino2) / 2
        rs.to_csv(os.path.join(tmp, "out", "tuep_record_scores.csv"), index=False)
        np.savez(os.path.join(tmp, "out", "tuep_window_scores.npz"), **{f"{f}_seed{i}__{rid}": rg.random(5).astype(np.float32) for rid in rs.id for f in ("reve4", "dino2") for i in range(3)})
        ra = run(os.path.join(HERE, "tuep_audit.py"), "--work", tmp, "--sample", 12)
        assert "верхние 20 % по всем окнам сеанса" in ra.stdout and os.path.exists(os.path.join(tmp, "out", "tuep_audit.csv"))
        print("ТЕСТ ДООБУЧЕНИЯ ПРОЙДЕН")


if __name__ == "__main__":
    main()
