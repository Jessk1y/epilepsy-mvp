"""Лёгкий тест tuep_variants.py на синтетике: py -I test_variants_synthetic.py  (процессор, без весов и интернета)

Данные — как в test_finetune_synthetic.py (40 пациентов, у больных ритм 6 Гц, перекос слоёв по классу). Проверяет:
выброс окон на заданных заранее порогах; биполярный монтаж; признаки и их кэш; эталон заново (b0) совпадает с tuep_baseline.py bpampw;
все варианты по 5 фолдам; вложенный выбор берёт в каждом фолде лучшего по отложенной части и не трогает тест; сводка без кодов пациентов.
"""
import os, subprocess, sys, tempfile
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, HERE)
import tuep_finetune as FT
import tuep_variants as TV
from test_finetune_synthetic import make, run


def unit():
    rng = np.random.default_rng(0)
    t = np.arange(2000) / 200
    clean = 20 * np.sin(2 * np.pi * 10 * t) + rng.normal(0, 2, (19, 2000))
    big = clean.copy(); big[3, 500:520] += 900
    musc = clean.copy(); musc[:5] += np.stack([ss_hp(rng.normal(0, 40, 2000)) for _ in range(5)])
    flat = clean.copy(); flat[7] = 0.1 * rng.normal(size=2000)
    ok = TV.window_ok(np.stack([clean, big, musc, flat]))
    assert ok.tolist() == [True, False, False, False], ok
    # биполярный монтаж: 18 отведений, разности каналов
    assert len(TV.BIP_A) == 18 and TV.EL[TV.BIP_A[0]] == "Fp1" and TV.EL[TV.BIP_B[0]] == "F7" and TV.EL[TV.BIP_B[-1]] == "Pz"
    W = np.stack([clean, clean])
    fa, _, _ = TV.window_features(W, "avg", TV._band_sos())
    fb, _, _ = TV.window_features(W[:, TV.BIP_A] - W[:, TV.BIP_B], "bip", TV._band_sos())
    assert fa["rel"].shape == (2, 95) and fb["rel"].shape == (2, 90) and fb["wpli_node_all"].shape == (2, 90) and fa["asym"].shape == (2, 48)
    # 1/f: у белого шума наклон ~0, у броуновского ~ −2; пик альфа у чистого сигнала 10 Гц
    wn = rng.normal(0, 10, (1, 19, 2000)); br = np.cumsum(rng.normal(0, 1, (1, 19, 2000)), -1)
    sw = TV.window_features(wn, "avg", TV._band_sos())[0]["aper"][0, :19].mean()
    sb = TV.window_features(br - br.mean(-1, keepdims=True), "avg", TV._band_sos())[0]["aper"][0, :19].mean()
    assert abs(sw) < 0.3 and -2.5 < sb < -1.5, (sw, sb)
    _, fr, lp = TV.window_features(np.stack([clean + rng.normal(0, 5, (19, 2000)) for _ in range(4)]), "avg", TV._band_sos())
    assert np.allclose(TV.paf_from_spectrum(fr, lp), 10.0), TV.paf_from_spectrum(fr, lp)
    print("модульные проверки пройдены")


def ss_hp(x):
    import scipy.signal as ss
    return ss.sosfiltfilt(ss.butter(4, 35, btype="highpass", fs=200, output="sos"), x)


def main():
    unit()
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        prep, out = os.path.join(tmp, "prep"), os.path.join(tmp, "ft")
        os.makedirs(prep); make(prep)
        meta = FT.load_prep(prep); S = FT.session_frame(meta); chosen, _ = FT.chosen_sessions(S)
        # прежние запуски: эталон bpampw и подставная «REVE» (эталон + шум) в <out>/runs
        run(os.path.join(HERE, "tuep_baseline.py"), "--prep", prep, "--out", out, "--threads", 2, "--sets", "bpampw", "tech")
        rng = np.random.default_rng(3)
        for k in range(5):
            d = pd.read_csv(os.path.join(out, "runs", f"bpampw_f{k}_s0.csv"))
            d.assign(score=(0.3 * d.score + 0.7 * rng.random(len(d)))).to_csv(os.path.join(out, "runs", f"reve_f{k}_s0.csv"), index=False)
        V = os.path.join(HERE, "tuep_variants.py")
        r = run(V, "features", "--prep", prep, "--out", out, "--procs", 2, "--limit", 6, "--configs", "avg_r1_all")
        assert "оценка на все" in r.stdout and not os.path.exists(os.path.join(out, "variants", "feats_avg_r1_all.npz"))
        CF = ["avg_r0_w60", "avg_r1_w60", "bip_r1_all"]                            # часть настроек: остальные варианты должны пропуститься
        run(V, "features", "--prep", prep, "--out", out, "--procs", 3, "--configs", *CF)
        r = run(V, "features", "--prep", prep, "--out", out, "--procs", 3, "--configs", *CF)
        assert r.stdout.count("уже есть") == len(CF)                               # кэш: повторно не считается
        z = np.load(os.path.join(out, "variants", "feats_bip_r1_all.npz"))
        z60 = np.load(os.path.join(out, "variants", "feats_avg_r0_w60.npz"))
        assert (z["n_win"] >= z60["n_win"]).all() and z["g_rel"].shape[1] == 2 * 90 and z60["g_rel"].shape[1] == 2 * 95
        run(V, "embed", "--prep", prep, "--out", out, "--model", "toy", "--device", "cpu", "--seeds", 0, 1)
        vd = os.path.join(out, "variants")
        r = run(V, "run", "--prep", prep, "--out", out, "--emb-file", os.path.join(vd, "toy_emb.npz"))
        assert "v3_all: пропущен" in r.stdout and "v7_comb: пропущен" in r.stdout
        vd = os.path.join(out, "variants")
        T = pd.read_csv(os.path.join(vd, "tuep_variants_metrics.csv")).set_index("код")
        CH = pd.read_csv(os.path.join(vd, "tuep_variants_nested_choice.csv"))
        names = [v["name"] for v in TV.variants() if v["cfg"] in CF]
        assert set(names) <= set(T.index) and {"nested", "ref_reve+bpampw", "ref_bpampw", "ref_tech"} <= set(T.index), T.index.tolist()
        assert (T.n == len(chosen)).all(), T.n
        auc = {c: float(v.split()[0]) for c, v in T["ROC AUC (95 % ДИ)"].items()}
        print("ROC AUC на синтетике:", auc)
        assert auc["b0"] > 0.9 and auc["nested"] > 0.8, auc
        # b0 = прежний bpampw: те же признаки -> те же оценки
        for k in range(5):
            a_ = pd.read_csv(os.path.join(vd, "runs", f"b0_f{k}_s0.csv")); b_ = pd.read_csv(os.path.join(out, "runs", f"bpampw_f{k}_s0.csv"))
            m = a_.merge(b_, on=["split", "id"])
            assert len(m) == len(a_) and np.abs(m.score_x - m.score_y).max() < 1e-3, np.abs(m.score_x - m.score_y).max()
        # вложенный выбор: в каждом фолде выбран кандидат с максимальным ROC AUC отложенной части; итог по тесту = его тест
        R = {**FT.load_runs(os.path.join(vd, "runs"))}
        P = pd.read_csv(os.path.join(vd, "tuep_variants_patient_scores.csv"))
        assert len(CH) == 5 and P.patient.is_unique and P.nested_score.notna().all()
        for _, c in CH.iterrows():
            m = c["выбран"]
            if m in R:
                d = R[m]; te = FT.patient_scores(d[(d.split == "test") & (d.fold == c["фолд"])], chosen)
                q = P.set_index("patient").loc[te.patient]
                assert np.allclose(q.nested_score.to_numpy(), te.score.to_numpy()), m
                assert abs(TV.val_auc(d, c["фолд"], chosen) - c["отложенные ROC AUC"]) < 1e-3
        for nm in ("b0", "v5_hgb", "v2_rej"):                                        # отложенные AUC выбранного не меньше, чем у этих
            for _, c in CH.iterrows():
                assert c["отложенные ROC AUC"] >= TV.val_auc(R[nm], c["фолд"], chosen) - 1e-3
        md = open(os.path.join(vd, "tuep_variants_summary.md"), encoding="utf-8").read()
        assert "вложенный выбор" in md and "поисково" in md and "Юден" in md and not any(p in md for p in chosen)
        r2 = run(V, "run", "--prep", prep, "--out", out, "--emb-file", os.path.join(vd, "toy_emb.npz"))                           # повторно модели не обучаются
        assert "отложенные ROC AUC" not in r2.stdout.split("# TUEP")[0]
        print(md[:1800])
        print("ТЕСТ ВАРИАНТОВ ПРОЙДЕН")


if __name__ == "__main__":
    main()
