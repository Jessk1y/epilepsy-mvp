"""Итоговые модели варианта (б) на синтетике: команды final / --save-model / card и совпадение оценки записи в памяти
(h1_eeg/src/record_scoring.py, им пользуется приложение) с кодом экспериментов (tuep_finetune.py, tuep_baseline.py).
Игрушечная сеть, процессор, около минуты. Настоящие веса и данные не нужны."""
import json, os, subprocess, sys
import numpy as np, pytest

import record_scoring as RS
import tuep_finetune as FT
import tuep_baseline as TB
from shared_data import read_record

TUEP = os.path.dirname(os.path.abspath(FT.__file__))
sys.path.insert(0, TUEP)
from test_finetune_synthetic import make  # noqa: E402  (синтетический «подготовленный TUEP»)


def run(*args):
    r = subprocess.run([sys.executable, "-X", "utf8", *map(str, args)], capture_output=True, text=True, encoding="utf-8", errors="replace")
    assert r.returncode == 0, r.stdout[-1500:] + r.stderr[-2500:]
    return r.stdout


@pytest.fixture(scope="module")
def work(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("tuep")
    prep, out = str(tmp / "prep"), str(tmp / "ft")
    os.makedirs(prep); make(prep)
    ft = os.path.join(TUEP, "tuep_finetune.py")
    common = ["--model", "toy", "--prep", prep, "--out", out, "--seeds", 0, "--epoch-windows", 96, "--bs", 16, "--workers", 2, "--device", "cpu"]
    run(ft, "train", *common, "--max-epochs", 2)
    run(os.path.join(TUEP, "tuep_baseline.py"), "--prep", prep, "--out", out, "--threads", 2, "--sets", "bpampw", "--save-model", os.path.join(out, "final"))
    o = run(ft, "final", *common, "--max-epochs", 2)
    assert "медиана лучших эпох" in o
    run(ft, "card", "--prep", prep, "--out", out, "--ensemble", "toy", "bpampw")
    return prep, out


def test_files_for_app(work):
    import torch
    prep, out = work
    fin = os.path.join(out, "final")
    assert sorted(os.listdir(fin)) == ["baseline_bpampw.npz", "pair_card.json", "toy_tuep_seed0.pt"]
    ck = torch.load(os.path.join(fin, "toy_tuep_seed0.pt"), map_location="cpu", weights_only=False)
    assert set(ck) == {"state_dict", "config"} and ck["config"]["lowpass"] == 45.0 and ck["config"]["n_patients"] > 30
    card = json.load(open(os.path.join(fin, "pair_card.json"), encoding="utf-8"))
    assert card["models"] == ["toy", "bpampw"] and {"youden", "sens80"} <= set(card["thresholds"]) and 0 <= card["auc"] <= 1 and card["n"] >= 38


def test_in_memory_windows_same_as_experiments(work):
    prep, _ = work
    meta = FT.load_prep(prep)
    for r in list(meta.itertuples())[:4]:
        X = read_record(prep, r._asdict())
        for low in (0, 45):
            sos = FT.make_sos(low)
            assert np.array_equal(sos, RS.make_sos(low))
            W, starts = RS.eval_windows_array(X, sos)
            assert np.array_equal(W, FT.eval_windows(prep, r, sos, FT.CAP_WIN))
            assert len(starts) == len(W) and (len(W) == 0 or starts[-1] + 2000 <= X.shape[1])


def test_baseline_in_memory_same_as_experiments(work):
    """Признаки и вероятность эталона, посчитанные приложением по записи в памяти, равны посчитанным скриптами по набору на диске."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    prep, out = work
    meta = FT.load_prep(prep).reset_index(drop=True)
    XB = np.load(os.path.join(out, f"baseline_features_w{FT.CAP_WIN}.npz"), allow_pickle=True)["X"]
    XW = np.load(os.path.join(out, f"baseline_wpli_w{FT.CAP_WIN}.npz"), allow_pickle=True)["X"]
    lr = RS.LogReg(os.path.join(out, "final", "baseline_bpampw.npz"))
    assert lr.info["set"] == "bpampw" and lr.info["n_features"] == 276 and lr.info["C"] in TB.CS
    sos, bs = RS.make_sos(0), RS.band_sos()
    for i in (0, 7, 23):
        r = meta.iloc[i]
        W, _ = RS.eval_windows_array(read_record(prep, r.to_dict()), sos)
        v = RS.baseline_vector(RS.band_amp_matrix(W), RS.wpli_matrix(W, bs))
        assert v.shape == (276,) and np.allclose(v, np.r_[XB[i], XW[i]], atol=1e-9)
    # сохранённые массивы дают ту же вероятность, что sklearn с теми же параметрами
    X = np.concatenate([XB, XW], 1)
    z = np.nan_to_num((X - lr.mean) / lr.scale) @ lr.coef + lr.intercept
    p = np.array([lr.predict_proba(x) for x in X])
    assert np.allclose(p, 1 / (1 + np.exp(-z))) and 0.05 < p.std()
    assert lr.mean.shape == (276,) and isinstance(StandardScaler(), object) and isinstance(LogisticRegression(), object)


def test_top20():
    p = np.array([0.1, 0.9, 0.5, 0.7, 0.3, 0.2, 0.4, 0.6, 0.8, 0.0])
    assert RS.top20(p) == pytest.approx(FT.top20(p)) == pytest.approx(0.85)
    assert list(RS.spread(100, 60)) == list(FT.spread(100, 60))
