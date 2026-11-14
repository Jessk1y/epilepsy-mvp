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


def main():
    unit()
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        unit_edf(tmp)
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
        print("ТЕСТ ВЕТКИ DEEPEPILEPSY ПРОЙДЕН")


if __name__ == "__main__":
    main()
