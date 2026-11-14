"""Лёгкий тест ветки DeepEpilepsy (--model de) на синтетических данных: py -I test_de_synthetic.py  (процессор, без кода и весов авторов)

Вместо ViT1d авторов — крошечная сеть с тем же устройством (основа + линейная голова на 2 класса, флаг --de-toy); весь остальной путь
настоящий: окна 30 с, порядок каналов авторов, standard_scale, RandAugment авторов, медиана по окнам, пропуск файлов короче 30 с,
заморозка основы (defz), пара de + bpampw в отчёте. Данные — как в test_finetune_synthetic.py плюс файлы по 15 с.
"""
import argparse, os, sys, tempfile
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import tuep_finetune as FT
from test_finetune_synthetic import make, run


def unit():
    import torch
    g = np.random.default_rng(0)
    x = g.normal(0, 10, (19, 6000)) + 30 * np.sin(np.arange(6000) / 5.0)
    # RandAugment авторов: форма сохраняется, значения конечны, примерно в половине случаев окно не меняется, все приёмы встречаются
    same, kinds = 0, set()
    for _ in range(300):
        y = FT.de_randaugment(x, g)
        assert y.shape == x.shape and np.isfinite(y).all()
        same += y is x
        zero_ch = int((np.abs(y).max(-1) == 0).sum())
        kinds.add("каналы" if zero_ch else "прямоугольник" if (y == 0).any() else "без изменений" if y is x else "фильтр или шум")
    assert 150 < same < 240, same                                # 1/2 «не сработало» + 1/12 пустой приём
    assert kinds == {"каналы", "прямоугольник", "без изменений", "фильтр или шум"}, kinds
    # вход: порядок каналов авторов и нормировка по каналу
    W = np.stack([np.arange(19)[:, None] * 100.0 + g.normal(0, 1 + np.arange(19)[:, None], (19, 6000)) for _ in range(2)])
    t = FT.de_input(W, torch)
    assert t.shape == (2, 19, 6000) and t.dtype == torch.float32
    assert float(t.mean(-1).abs().max()) < 1e-3 and float((t.std(-1) - 1).abs().max()) < 1e-2
    assert [FT.EL[i] for i in FT.DE_IDX] == FT.DE_ORDER and FT.DE_ORDER[8:10] == ["F7", "F8"] and FT.EL[8:10] == ["O1", "O2"]
    k = 3; ref = W[0, FT.DE_IDX[k]]
    assert np.allclose(t[0, k].numpy(), (ref - ref.mean()) / ref.std(), atol=1e-3)
    # обёртка: логит класса 1 = z1 − z0, сигмоида от него = softmax[:, 1]; заморозка трогает только основу
    a = argparse.Namespace(de_toy=True, freeze_backbone=True, lr_scale=1.0, init="nmt")
    FT.set_window("de")
    m, inp, groups, warm, back, train_in = FT.build_model("de", a, torch, "cpu", 0)
    xb = inp(W)
    z = m.head(m.backbone(xb))
    assert torch.allclose(torch.sigmoid(m(xb)), torch.softmax(z, 1)[:, 1], atol=1e-6)
    assert [n for n, p in m.named_parameters() if p.requires_grad] == ["head.weight", "head.bias"] and len(groups(m)) == 1
    assert train_in(W, g).shape == (2, 19, 6000)
    assert FT.L == 6000 and FT.n_windows(60) == 20 and FT.n_windows(12) == 4
    FT.set_window("reve"); assert FT.L == 2000 and FT.n_windows(60) == 60


def unit_edf(tmp):
    """Проверка референта (tuep_deepepilepsy.load_edf) на синтетических EDF: связанные уши = канал − (A1 + A2) / 2."""
    import tuep_common as C, tuep_deepepilepsy as DE
    from test_synthetic import write_edf, LABELS_REF
    fs, sec = 200, 40
    src = np.random.default_rng(3).normal(0, 20, (len(LABELS_REF), fs * sec))
    names = [C.norm_channel(l) for l in LABELS_REF]
    i19, ears = [names.index(e.upper()) for e in FT.EL], (src[names.index("A1")] + src[names.index("A2")]) / 2
    p = os.path.join(tmp, "ref.edf"); write_edf(p, LABELS_REF, fs, sec, src)
    X, why = DE.load_edf(p, "ears", "none")
    assert why == "" and X.shape == (19, fs * sec) and np.abs(X - (src[i19] - ears)).max() < 0.2          # шаг квантования EDF ~0,1 мкВ
    X, _ = DE.load_edf(p, "avg", "none")
    assert np.abs(X.mean(0)).max() < 1e-3 and np.abs(X - (src[i19] - src[i19].mean(0))).max() < 0.2
    X, _ = DE.load_edf(p, "avg", "prep")
    assert X.shape == (19, fs * sec) and np.isfinite(X).all()
    keep = [i for i, n in enumerate(names) if n not in ("A1", "A2")]
    write_edf(p, [LABELS_REF[i] for i in keep], fs, sec, src[keep])                                       # монтаж без ушных электродов
    assert DE.load_edf(p, "ears", "none") == (None, "нет A1/A2")
    write_edf(p, [LABELS_REF[i].replace("-REF", "-LE") for i in keep], fs, sec, src[keep])                # -LE: уже связанные уши
    X, why = DE.load_edf(p, "ears", "none")
    assert why == "" and np.abs(X - src[i19]).max() < 0.2
    d = pd.DataFrame(dict(patient=["a", "a", "b"], y=[1, 1, 0], dur_s=[10.0, 30.0, 5.0], de=[0.0, 1.0, 0.5]))
    assert DE.patient_auc(d).s.round(3).to_dict() == {"a": 0.75, "b": 0.5}
    # подготовка «как у авторов»: уши + только ФВЧ 0,1 Гц; постоянная составляющая EDF уходит, сеть 60 Гц и верх спектра остаются
    import scipy.signal as ss
    tt = np.arange(fs * sec) / fs
    src2 = src + 500.0 + 15 * np.sin(2 * np.pi * 60 * tt)
    src2[names.index("A1")] -= 15 * np.sin(2 * np.pi * 60 * tt)                                           # сеть не должна сократиться целиком
    write_edf(p, LABELS_REF, fs, sec, src2)
    X, info = C.load_record(p, ref="ears", flt="hp")
    want = src2[i19] - (src2[names.index("A1")] + src2[names.index("A2")]) / 2
    want = ss.sosfiltfilt(ss.butter(4, 0.1, btype="highpass", fs=fs, output="sos"), want, axis=-1)
    assert X.dtype == np.float16 and np.abs(X.astype(float) - want)[:, 2000:-2000].max() < 0.3 and abs(float(X.astype(float).mean())) < 1.0
    f, P = ss.welch(X.astype(float), fs, nperseg=400)
    assert P[:, np.argmin(abs(f - 60))].mean() > 5 * P[:, np.argmin(abs(f - 55))].mean()                  # режекции нет
    Xa, _ = C.load_record(p)                                                                              # основная подготовка не изменилась
    f, P = ss.welch(Xa.astype(float), fs, nperseg=400)
    assert abs(Xa.astype(float).mean(0)).max() < 0.5 and P[:, np.argmin(abs(f - 90))].mean() < 0.05 * P[:, np.argmin(abs(f - 30))].mean()
    write_edf(p, [LABELS_REF[i] for i in keep], fs, sec, src2[keep])
    assert C.load_record(p, ref="ears", flt="hp")[0] is None and "A1/A2" in C.load_record(p, ref="ears", flt="hp")[1]["error"]


def unit_prepare(tmp):
    """tuep_prepare.py --ref ears --filt hp: те же id, что в основной подготовке, минус файлы без ушных электродов."""
    import test_synthetic as TS
    root = os.path.join(tmp, "v3.1.0"); TS.make(root)
    d = os.path.join(root, "edf", "epilepsy", "bbbbbbbb", "s002_2015", "03_tcp_ar_a"); os.makedirs(d)
    lab = [l for l in TS.LABELS_REF if "A1" not in l and "A2" not in l]
    TS.write_edf(os.path.join(d, "bbbbbbbb_s002_t000.edf"), lab, 250, 40, np.random.default_rng(5).normal(0, 10, (len(lab), 40 * 250)))
    pa, pe = os.path.join(tmp, "prep_avg"), os.path.join(tmp, "prep_ears")
    run(os.path.join(HERE, "tuep_prepare.py"), "--root", root, "--out", pa, "--threads", 2)
    r = run(os.path.join(HERE, "tuep_prepare.py"), "--root", root, "--out", pe, "--threads", 2, "--ref", "ears", "--filt", "hp")
    A, E = pd.read_csv(os.path.join(pa, "h1_shared_meta.csv")), pd.read_csv(os.path.join(pe, "h1_shared_meta.csv"))
    assert set(A.id) - set(E.id) == {"bbbbbbbb_s002_t000"} and set(E.id) < set(A.id) and "нет A1/A2" in r.stdout
    m = A.merge(E, on="id", suffixes=("", "_e"))
    assert (m.n200 == m.n200_e).all() and (m.patient == m.patient_e).all() and (m.session == m.session_e).all() and (m.montage == m.montage_e).all()
    import json
    assert "A1 + A2" in json.load(open(os.path.join(pe, "h1_shared_info.json"), encoding="utf-8"))["reference"]


def auth_conditions(tmp, prep, out, chosen):
    """Условия авторов: сигнал из неполной подготовки (нет слоя 03_tcp_ar_a и одного пациента), протокол — по основной; отчёт с доп. папкой."""
    prep2, out2 = os.path.join(tmp, "prep_ears_syn"), os.path.join(tmp, "ft_auth")
    os.makedirs(prep2); make(prep2, short=True)                                                           # те же id и те же сигналы
    mp = os.path.join(prep2, "h1_shared_meta.csv"); M = pd.read_csv(mp)
    gone = sorted(M.patient[M.montage == "01_tcp_ar"].unique())[5]
    M2 = M[(M.montage != "03_tcp_ar_a") & (M.patient != gone)]; M2.to_csv(mp, index=False)
    have = set(zip(M2.patient[M2.n200 >= 6000], M2.session[M2.n200 >= 6000]))
    scored = {p for p, s in chosen.items() if (p, s) in have}
    assert 0 < len(scored) < len(chosen) - 1, (len(scored), len(chosen))
    base = [os.path.join(HERE, "tuep_finetune.py"), "train", "--model", "de", "--de-toy", "--name", "dea", "--prep", prep2, "--proto-prep", prep, "--out", out2,
            "--lowpass", 0, "--seeds", 0, "--max-epochs", 2, "--epoch-windows", 192, "--bs", 16, "--workers", 2, "--device", "cpu", "--lr-scale", 200]
    r = run(*base)
    assert "протокол (сеансы, слои, фолды, выбранный сеанс)" in r.stdout and os.path.isdir(os.path.join(out2, "evalcache_lp0_w60_L6000"))
    run(*base, "--freeze-backbone", "--fold-list", 0)
    runs = os.path.join(out2, "runs")
    assert sorted(f for f in os.listdir(runs) if f.endswith(".csv")) == [f"dea_f{k}_s0.csv" for k in range(5)] + ["deafz_f0_s0.csv"]
    d = pd.concat(pd.read_csv(os.path.join(runs, f"dea_f{k}_s0.csv")) for k in range(5))
    F = FT.get_folds(FT.load_prep(prep)).set_index("patient").fold                                        # фолды — по основной подготовке
    t = d[d.split == "test"]
    assert (t.fold == t.patient.map(F)).all() and gone not in set(d.patient) and set(t.id) == set(M2.id[M2.n200 >= 6000])
    rep = [os.path.join(HERE, "tuep_finetune.py"), "report", "--prep", prep, "--out", out2, "--extra-runs", os.path.join(out, "runs"),
           "--ensemble", "dea", "bpampw", "--ensemble", "de", "bpampw"]
    run(*rep)
    T = pd.read_csv(os.path.join(out2, "tuep_ft_metrics.csv")).set_index("код")
    assert set(T.index) == {"dea", "deafz", "de", "defz", "bpampw", "dea+bpampw", "de+bpampw"}, T.index.tolist()
    assert T.n["dea"] == len(scored) and T.n["dea+bpampw"] == len(scored) and T.n["bpampw"] == len(chosen) and T.n["de+bpampw"] > len(scored), T.n.to_dict()
    run(*rep, "--same-patients", "dea")
    T2 = pd.read_csv(os.path.join(out2, "tuep_ft_metrics_same_dea.csv")).set_index("код")
    assert (T2.n.drop(["deafz", "defz"]) == len(scored)).all() and T2.n["deafz"] == T2.n["defz"] <= len(scored), T2.n.to_dict()   # эти два — один фолд
    TH = pd.read_csv(os.path.join(out2, "tuep_ft_threshold_metrics_same_dea.csv"))
    assert {"accuracy", "precision", "recall (чувств.)", "специфичность"} <= set(TH.columns) and (TH.n <= len(scored)).all()
    md = open(os.path.join(out2, "tuep_ft_summary_same_dea.md"), encoding="utf-8").read()
    assert "на одних и тех же пациентах" in md and "условиях авторов" in md and not any(p in md for p in chosen)
    assert os.path.exists(os.path.join(out2, "tuep_ft_summary.md"))                                       # полная сводка не затёрта


def main():
    unit()
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        unit_edf(tmp)
        unit_prepare(tmp)
        prep, out = os.path.join(tmp, "prep"), os.path.join(tmp, "ft")
        os.makedirs(prep); make(prep, short=True)
        meta = FT.load_prep(prep); S = FT.session_frame(meta); F = FT.get_folds(meta)
        chosen, dropped = FT.chosen_sessions(S)
        short_ids = set(meta.id[meta.n200 < 6000])
        assert len(short_ids) == 10
        has_long = set(zip(meta.patient[meta.n200 >= 6000], meta.session[meta.n200 >= 6000]))
        scored = {p for p, s in chosen.items() if (p, s) in has_long}       # пациенты, у которых в выбранном сеансе есть файл от 30 с
        print(f"пациентов с выбранным сеансом: {len(chosen)}, из них с файлом от 30 с: {len(scored)}")
        FT.set_window("de")
        r0 = next(r for r in meta.itertuples() if r.id in short_ids)
        assert FT.eval_windows(prep, r0, FT.make_sos(45), 20).shape == (0, 19, 6000)
        FT.set_window("reve")
        base = [os.path.join(HERE, "tuep_finetune.py"), "train", "--model", "de", "--de-toy", "--prep", prep, "--out", out, "--seeds", 0,
                "--max-epochs", 3, "--epoch-windows", 192, "--bs", 16, "--workers", 2, "--device", "cpu", "--lr-scale", 200]
        r = run(*base)
        assert "файлов короче 30 с (без оценки)" in r.stdout
        run(*base, "--freeze-backbone", "--fold-list", 0, "--save-weights")
        runs = os.path.join(out, "runs")
        assert sorted(f for f in os.listdir(runs) if f.endswith(".csv")) == [f"de_f{k}_s0.csv" for k in range(5)] + ["defz_f0_s0.csv"]
        assert os.path.isdir(os.path.join(out, "evalcache_lp45_w60_L6000"))
        d = pd.concat(pd.read_csv(os.path.join(runs, f"de_f{k}_s0.csv")) for k in range(5))
        assert not set(d.id) & short_ids and set(d.id[d.split == "test"]) == set(meta.id) - short_ids
        import json, torch
        h = json.load(open(os.path.join(runs, "defz_f0_s0.json"), encoding="utf-8"))
        assert h["win_s"] == 30 and h["n_win"] == 20 and h["agg"] == "median" and h["init"] == "authors" and h["model"] == "defz"
        # заморозка: основа после обучения та же, что при создании с тем же seed; голова изменилась
        FT.set_window("de")
        m0 = FT.build_model("de", argparse.Namespace(de_toy=True, freeze_backbone=True, lr_scale=1.0, init="nmt"), torch, "cpu", 0)[0].state_dict()
        FT.set_window("reve")
        w = torch.load(os.path.join(out, "weights", "defz_f0_s0.pt"), map_location="cpu", weights_only=False)["state_dict"]
        assert torch.equal(w["backbone.stem.weight"], m0["backbone.stem.weight"]) and not torch.equal(w["head.weight"], m0["head.weight"])
        # эталон с wPLI и отчёт с двумя заданными парами (второй пары нет — она пропускается с сообщением)
        run(os.path.join(HERE, "tuep_baseline.py"), "--prep", prep, "--out", out, "--threads", 2, "--sets", "bpampw")
        rr = run(os.path.join(HERE, "tuep_finetune.py"), "report", "--prep", prep, "--out", out, "--ensemble", "de", "bpampw", "--ensemble", "reve", "bpampw")
        assert "пара reve+bpampw пропущена" in rr.stdout
        T = pd.read_csv(os.path.join(out, "tuep_ft_metrics.csv")).set_index("код")
        assert set(T.index) == {"de", "defz", "bpampw", "de+bpampw"}, T.index.tolist()
        assert T.n["de"] == len(scored) and T.n["de+bpampw"] == len(scored) and T.n["bpampw"] == len(chosen), T.n.to_dict()
        assert T["фолдов"]["de"] == 5 and T["фолдов"]["defz"] == 1
        auc = {c: float(v.split()[0]) for c, v in T["ROC AUC (95 % ДИ)"].items()}
        print("ROC AUC на синтетике:", auc)
        assert auc["bpampw"] > 0.9 and auc["de"] > 0.7, auc                 # ритм 6 Гц должны найти и эталон, и сеть на окнах 30 с
        P = pd.read_csv(os.path.join(out, "tuep_ft_patient_scores.csv"))
        assert set(P.patient[P.de_score.notna()]) == scored
        md = open(os.path.join(out, "tuep_ft_summary.md"), encoding="utf-8").read()
        assert "DeepEpilepsy авторов, дообучена на TUEP" in md and not any(p in md for p in chosen), "в сводке не должно быть кодов пациентов"
        auth_conditions(tmp, prep, out, chosen)
        print("ТЕСТ ВЕТКИ DEEPEPILEPSY ПРОЙДЕН")


if __name__ == "__main__":
    main()
