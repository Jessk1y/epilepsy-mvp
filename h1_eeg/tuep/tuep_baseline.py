"""Эталон для задачи 1.6: мощности ритмов + логистическая регрессия и контроль утечки «без ЭЭГ» — по тем же фолдам и правилам,
что tuep_finetune.py (тот же тест, та же отложенная часть, те же веса сеансов, тот же сеанс на пациента). Только процессор, минуты.

    python tuep_baseline.py --prep ~/data/tuep_work/prep --out ~/data/tuep_work/ft [--threads 6]
    python tuep_finetune.py report --prep ~/data/tuep_work/prep --out ~/data/tuep_work/ft        # эталон попадёт в общую таблицу

Модели (код в отчёте):
  bp     — по каждому из 19 каналов относительная мощность δ 1–4, θ 4–8, α 8–13, β 13–30, γ 30–45 Гц (lg доли от 1–45 Гц);
           по окнам файла — медиана и 90-й процентиль (190 признаков);
  bpamp  — то же + lg СКО канала (ещё 38 признаков). Амплитуда зависит от аппаратуры, поэтому отдельно;
  tech   — БЕЗ ЭЭГ: монтаж, частота дискретизации, длительность файла и сеанса, число файлов, год. Показывает, сколько ROC AUC
           даёт утечка при той же схеме проверки; дообученная модель должна быть выше неё и держаться внутри слоёв.
Окна — те же 60 равномерных окон 10 с на файл, ФВЧ 0,3 Гц. Сила регуляризации C выбирается по отложенной части фолда.
"""
import argparse, os, sys
from concurrent.futures import ThreadPoolExecutor
import numpy as np, pandas as pd, scipy.signal as ss
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import tuep_finetune as FT                                 # noqa: E402

BANDS = [(1, 4), (4, 8), (8, 13), (13, 30), (30, 45)]
N_REL = 19 * len(BANDS)
CS = [0.003, 0.03, 0.3]


def file_features(prep, r, sos, n_win):
    """Признаки файла: (2 × (95 + 19),) — медиана и 90-й процентиль по окнам."""
    W = FT.eval_windows(prep, r, sos, n_win)
    f, P = ss.welch(W.astype(np.float64), FT.FS, nperseg=400, axis=-1)
    tot = P[..., (f >= 1) & (f < 45)].sum(-1) + 1e-12
    rel = np.stack([np.log10(P[..., (f >= lo) & (f < hi)].sum(-1) / tot + 1e-9) for lo, hi in BANDS], -1).reshape(len(W), -1)
    amp = np.log10(W.std(-1) + 1e-3)
    X = np.concatenate([rel, amp], 1)
    return np.concatenate([np.median(X, 0), np.percentile(X, 90, 0)])


def tech_features(meta, S):
    """Признаки без ЭЭГ по файлу: монтаж, частота, длительности, число файлов, год."""
    s = S.set_index(["patient", "session"])
    D = pd.DataFrame({"lg_dur_file": np.log10(meta.dur_s.to_numpy()),
                      "lg_dur_sess": np.log10([s.dur_s[(p, q)] for p, q in zip(meta.patient, meta.session)]),
                      "n_files": [s.n_files[(p, q)] for p, q in zip(meta.patient, meta.session)],
                      "year": meta.year.fillna(meta.year.median()).fillna(0).to_numpy() if "year" in meta else 0.0})
    D = pd.concat([D, pd.get_dummies(meta.montage.astype(str).to_numpy(), prefix="m", dtype=float),
                   pd.get_dummies(meta.sfreq_orig.round().astype(int).astype(str).to_numpy(), prefix="fs", dtype=float)], axis=1)
    return D.to_numpy(float)


def fit_predict(X, meta, S, F, chosen, fold, balance, min_pat):
    """Одна модель на фолд: обучение на файлах обучающей части (вес файла = вес сеанса × доля длительности), C — по отложенной части."""
    fit_p, val_p, test_p = FT.split_patients(F, fold)
    S_fit = S[S.patient.isin(fit_p)].reset_index(drop=True)
    w, _ = FT.session_weights(S_fit, balance, min_pat)
    ws = {(p, s): wi for p, s, wi in zip(S_fit.patient, S_fit.session, w)}
    dur = meta.groupby(["patient", "session"]).dur_s.transform("sum").to_numpy()
    wf = np.array([ws.get(k, 0.0) for k in zip(meta.patient, meta.session)]) * meta.dur_s.to_numpy() / dur
    tr = wf > 0
    va = (meta.patient.isin(val_p) & np.array([chosen.get(p) == s for p, s in zip(meta.patient, meta.session)])).to_numpy()
    te = meta.patient.isin(test_p).to_numpy()
    assert not (tr & te).any() and not (tr & va).any() and not (va & te).any()
    sc = StandardScaler().fit(X[tr])
    Xs = np.nan_to_num(sc.transform(X))
    best = None
    for c in CS:
        m = LogisticRegression(C=c, max_iter=3000).fit(Xs[tr], meta.y.to_numpy()[tr], sample_weight=wf[tr] / wf[tr].mean())
        pv = m.predict_proba(Xs[va])[:, 1]
        V = FT.patient_scores(meta[va].assign(score=pv), chosen)
        auc = roc_auc_score(V.y, V.score)
        if best is None or auc > best[0]:
            best = (auc, c, pv, m.predict_proba(Xs[te])[:, 1])
    auc, c, pv, pt = best
    cols = ["id", "patient", "session", "y", "dur_s"]
    return pd.concat([meta.loc[va, cols].assign(split="val", score=pv), meta.loc[te, cols].assign(split="test", score=pt)]).assign(fold=fold, seed=0), c, auc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prep", required=True); ap.add_argument("--out", required=True); ap.add_argument("--folds", default="")
    ap.add_argument("--balance", default="strata", choices=["strata", "class"]); ap.add_argument("--min-pat", type=int, default=3)
    ap.add_argument("--exclude-sessions", default=""); ap.add_argument("--n-win", type=int, default=FT.CAP_WIN); ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0, help="только первые N файлов — проверка скорости, без обучения")
    a = ap.parse_args()
    for k in ("prep", "out", "folds", "exclude_sessions"):
        setattr(a, k, FT.expand(getattr(a, k)))
    os.makedirs(os.path.join(a.out, "runs"), exist_ok=True)
    meta = FT.load_prep(a.prep).reset_index(drop=True)
    S = FT.session_frame(meta, a.exclude_sessions); F = FT.get_folds(meta, a.folds); chosen, _ = FT.chosen_sessions(S)
    fpath = os.path.join(a.out, f"baseline_features_w{a.n_win}.npz")
    if os.path.exists(fpath) and list(np.load(fpath, allow_pickle=True)["ids"]) == list(meta.id):
        X = np.load(fpath, allow_pickle=True)["X"]; print("признаки взяты из", fpath)
    else:
        import time
        rows, sos, t0 = list(meta.itertuples()), FT.make_sos(0), time.time()
        if a.limit:
            rows = rows[:a.limit]
        with ThreadPoolExecutor(a.threads) as ex:
            X = np.stack(list(ex.map(lambda r: file_features(a.prep, r, sos, a.n_win), rows)))
        print(f"признаки: {X.shape} за {time.time()-t0:.0f} с")
        if a.limit:
            print(f"оценка на все {len(meta)} файлов: ~{(time.time()-t0) / a.limit * len(meta) / 60:.0f} мин"); return
        np.savez_compressed(fpath, X=X, ids=meta.id.to_numpy())
    half = X.shape[1] // 2
    rel = np.r_[0:N_REL, half:half + N_REL]
    sets = {"bp": X[:, rel], "bpamp": X, "tech": tech_features(meta, S)}
    for name, Xm in sets.items():
        for k in sorted(F.fold.unique()):
            d, c, auc = fit_predict(Xm, meta, S, F, chosen, int(k), a.balance, a.min_pat)
            d.to_csv(os.path.join(a.out, "runs", f"{name}_f{k}_s0.csv"), index=False)
            print(f"{name}: фолд {k}, C = {c}, отложенные ROC AUC {auc:.3f}")
    print("готово; сводка: python tuep_finetune.py report --prep … --out", a.out)


if __name__ == "__main__":
    main()
