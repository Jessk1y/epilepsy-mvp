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
  bpampw — bpamp + связность wPLI (15.11): средний wPLI по 171 паре в каждом из 5 диапазонов и «сила узла» каждого канала в θ 4–8 Гц
           (по окнам — медиана и 90-й процентиль, 48 признаков). wPLI 4–8 Гц был выше у больных во всех трёх открытых больницах (08.11);
  bpampwd — bpampw + возраст и пол из заголовка EDF (нужен --root);
  demo   — только возраст и пол (контроль: сколько даёт демография сама по себе).
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


IU = np.triu_indices(19, 1)


def wpli_features(prep, r, sos, n_win, band_sos):
    """wPLI по окнам файла: средний по парам в 5 диапазонах + сила узла в θ (19). Медиана и 90-й процентиль по окнам: (48,)."""
    W = FT.eval_windows(prep, r, sos, n_win).astype(np.float64)
    out = []
    for w in W:
        v = []
        for b, sb in enumerate(band_sos):
            z = ss.hilbert(ss.sosfiltfilt(sb, w, axis=-1), axis=-1)
            im = (z[IU[0]] * np.conj(z[IU[1]])).imag
            wp = np.abs(im.mean(-1)) / (np.abs(im).mean(-1) + 1e-12)
            v.append([wp.mean()])
            if b == 1:                                         # θ 4–8 Гц: сила узла
                M = np.zeros((19, 19)); M[IU] = wp; M = M + M.T
                v.append(M.sum(1) / 18)
        out.append(np.concatenate(v))
    X = np.array(out)
    return np.concatenate([np.median(X, 0), np.percentile(X, 90, 0)])


def edf_demo(root, inventory):
    """id файла -> (возраст, пол 1 = женский) из заголовка EDF TUH («… F 01-JAN-0000 … Age:68»)."""
    import re
    inv = pd.read_csv(inventory)
    res = {}
    for f, pth in zip(inv.file, inv.path):
        with open(os.path.join(root, pth), "rb") as fh:
            h = fh.read(88).decode("latin-1")
        m = re.search(r"Age:(\d+)", h); t = h[8:].split()
        res[os.path.splitext(f)[0]] = (float(m.group(1)) if m else np.nan, 1.0 if len(t) > 1 and t[1].upper() == "F" else 0.0)
    return res


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


def save_final(X, meta, S, cs, a, out_dir):
    """Итоговая модель эталона для приложения: обучение на всех пациентах (веса файлов — как в фолдах), C — чаще всего выбиравшееся
    в фолдах (при равенстве — меньшее, т. е. более сильная регуляризация). Сохраняются только массивы — приложению не нужен sklearn."""
    import time
    from collections import Counter
    from record_scoring import save_logreg
    cnt = Counter(cs); c = min(k for k, v in cnt.items() if v == max(cnt.values()))
    w, _ = FT.session_weights(S, a.balance, a.min_pat)
    ws = {(p, s): wi for p, s, wi in zip(S.patient, S.session, w)}
    dur = meta.groupby(["patient", "session"]).dur_s.transform("sum").to_numpy()
    wf = np.array([ws.get(k, 0.0) for k in zip(meta.patient, meta.session)]) * meta.dur_s.to_numpy() / dur
    tr = wf > 0
    sc = StandardScaler().fit(X[tr])
    m = LogisticRegression(C=c, max_iter=3000).fit(np.nan_to_num(sc.transform(X[tr])), meta.y.to_numpy()[tr], sample_weight=wf[tr] / wf[tr].mean())
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"baseline_{a.save_set}.npz")
    save_logreg(path, sc, m, dict(set=a.save_set, C=c, C_by_fold=list(cs), n_features=int(X.shape[1]), n_files=int(tr.sum()), n_patients=int(meta.patient[tr].nunique()),
                                  n_win=a.n_win, hp=FT.HP, bands=BANDS, trained="все пациенты TUEP (итоговая модель для приложения)", date=time.strftime("%Y-%m-%d")))
    print(f"итоговая модель {a.save_set}: C = {c} (по фолдам {list(cs)}), файлов {int(tr.sum())}, пациентов {meta.patient[tr].nunique()} → {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prep", required=True); ap.add_argument("--out", required=True); ap.add_argument("--folds", default="")
    ap.add_argument("--balance", default="strata", choices=["strata", "class"]); ap.add_argument("--min-pat", type=int, default=3)
    ap.add_argument("--exclude-sessions", default=""); ap.add_argument("--n-win", type=int, default=FT.CAP_WIN); ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0, help="только первые N файлов — проверка скорости, без обучения")
    ap.add_argument("--root", default="", help="папка TUEP с EDF — для возраста и пола (наборы bpampwd, demo)")
    ap.add_argument("--inventory", default="", help="inventory_files.csv (по умолчанию <out>/../inventory/inventory_files.csv)")
    ap.add_argument("--sets", nargs="*", default=["bp", "bpamp", "tech", "bpampw", "bpampwd", "demo"])
    ap.add_argument("--save-model", default="", help="папка: обучить набор --save-set на ВСЕХ пациентах и сохранить baseline_<набор>.npz для приложения (обычно <out>/final)")
    ap.add_argument("--save-set", default="bpampw")
    a = ap.parse_args()
    for k in ("prep", "out", "folds", "exclude_sessions", "root", "inventory"):
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
    if {"bpampw", "bpampwd"} & set(a.sets):
        wpath = os.path.join(a.out, f"baseline_wpli_w{a.n_win}.npz")
        if os.path.exists(wpath) and list(np.load(wpath, allow_pickle=True)["ids"]) == list(meta.id):
            XW = np.load(wpath, allow_pickle=True)["X"]; print("wPLI взяты из", wpath)
        else:
            import time
            t0, sos = time.time(), FT.make_sos(0)
            band_sos = [ss.butter(4, b, btype="bandpass", fs=FT.FS, output="sos") for b in BANDS]
            with ThreadPoolExecutor(a.threads) as ex:
                XW = np.stack(list(ex.map(lambda r: wpli_features(a.prep, r, sos, a.n_win, band_sos), meta.itertuples())))
            np.savez_compressed(wpath, X=XW, ids=meta.id.to_numpy()); print(f"wPLI: {XW.shape} за {time.time()-t0:.0f} с")
        sets["bpampw"] = np.concatenate([X, XW], 1)
    if {"bpampwd", "demo"} & set(a.sets) and not a.root:
        print("без --root (папка TUEP с EDF) наборы bpampwd и demo пропущены"); a.sets = [k for k in a.sets if k not in ("bpampwd", "demo")]
    if {"bpampwd", "demo"} & set(a.sets):
        inv = a.inventory or os.path.join(os.path.dirname(a.out.rstrip("/")), "inventory", "inventory_files.csv")
        dm = edf_demo(a.root, inv)
        D = np.array([dm.get(i, (np.nan, 0.0)) for i in meta.id], float)
        D[:, 0] = np.where(np.isnan(D[:, 0]), np.nanmedian(D[:, 0]), D[:, 0])
        print(f"возраст и пол: из заголовков {int(np.isfinite(D[:, 0]).sum())} файлов")
        sets["demo"] = D
        if "bpampw" in sets:
            sets["bpampwd"] = np.concatenate([sets["bpampw"], D], 1)
    sets = {k: v for k, v in sets.items() if k in a.sets}
    chosen_c = {}
    for name, Xm in sets.items():
        for k in sorted(F.fold.unique()):
            d, c, auc = fit_predict(Xm, meta, S, F, chosen, int(k), a.balance, a.min_pat)
            chosen_c.setdefault(name, []).append(c)
            d.to_csv(os.path.join(a.out, "runs", f"{name}_f{k}_s0.csv"), index=False)
            print(f"{name}: фолд {k}, C = {c}, отложенные ROC AUC {auc:.3f}")
    if a.save_model:
        assert a.save_set in sets and a.save_set in ("bp", "bpamp", "bpampw"), "сохраняются только наборы по ЭЭГ: bp, bpamp, bpampw (и он должен быть в --sets)"
        save_final(sets[a.save_set], meta, S, chosen_c[a.save_set], a, FT.expand(a.save_model))
    print("готово; сводка: python tuep_finetune.py report --prep … --out", a.out)


if __name__ == "__main__":
    main()
