# Г1: синхронность каналов в самых подозрительных окнах REVE v4 (окна по 10 с, без пересечений, от начала записи)
import os, glob, json, time
for k in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]: os.environ[k] = "1"
import numpy as np, pandas as pd, scipy.signal as ss
from multiprocessing import Pool
from sklearn.metrics import roc_auc_score, roc_curve
from scipy.stats import wilcoxon, spearmanr
T0 = time.time(); OUT = "/kaggle/working"
root = os.path.dirname(glob.glob("/kaggle/input/**/h1_shared_meta.csv", recursive=True)[0])
meta = pd.read_csv(f"{root}/h1_shared_meta.csv")
WS = pd.read_csv(glob.glob("/kaggle/input/**/reve_v4_window_scores.csv", recursive=True)[0])
print(WS.columns.tolist(), WS.dataset.unique())
W = WS.groupby(["dataset", "id", "window"], as_index=False).p.mean()     # среднее трёх запусков
nm = {"nmt_eval": "eval", "turkey": "turkey"}
meta = meta[meta.split.isin(nm.values())]
YM = {(d, i): int(v) for d in nm for i, v in meta[meta.split == nm[d]].set_index("id").y.items()}
PREP = os.path.dirname(glob.glob("/kaggle/input/**/epieeg60_records.csv", recursive=True)[0])
ER = pd.read_csv(f"{PREP}/epieeg60_records.csv"); ER = ER[ER.n200.notna()]
print(ER.head(3).to_string()); print(W[W.dataset == "epieeg60"].id.unique()[:5])
EY = dict(zip(ER.id.astype(str), ER.y.astype(int))); EP = dict(zip(ER.id.astype(str), ER.person.astype(str)))
YM.update({("epieeg60", i): EY[i] for i in W[W.dataset == "epieeg60"].id.unique()})
W["y"] = [YM[(d, i)] for d, i in zip(W.dataset, W.id)]
L = 2000; BANDS = [(1, 4), (4, 8), (8, 13), (13, 20), (20, 30), (1, 30)]; BL = ["1-4", "4-8", "8-13", "13-20", "20-30", "1-30"]
SOS = [ss.butter(4, b, btype="bandpass", fs=200, output="sos") for b in BANDS]
IU = np.triu_indices(19, 1)
JOBS = []
for d, g in W.groupby("dataset"):
    if d == "epieeg60":
        for rid, gg in g.groupby("id"): JOBS.append((d, rid, None, 0, 0, gg.window.values))
        continue
    mm = meta[meta.split == nm[d]].set_index("id")
    for rid, gg in g.groupby("id"):
        r = mm.loc[rid]; JOBS.append((d, rid, r.shard200, int(r.off200), int(r.n200), gg.window.values))
def job(a):
    d, rid, sh, off, n, wins = a
    if d == "epieeg60":
        X = np.asarray(np.load(f"{PREP}/fs200/{rid}.npy"), np.float32); n = X.shape[1]
    else:
        X = np.asarray(np.load(f"{root}/fs200/{sh}", mmap_mode="r")[off:off + 19 * n], np.float32).reshape(19, n)
    assert (wins.max() + 1) * L <= n and wins.max() + 1 == n // L, (rid, wins.max(), n)
    F = [ss.sosfiltfilt(s, X, axis=-1) for s in SOS]
    rows = []
    for w in wins:
        r = {"dataset": d, "id": rid, "window": int(w), "std": float(X[:, w * L:(w + 1) * L].std())}
        for b, xb_all in enumerate(F):
            xb = xb_all[:, w * L:(w + 1) * L]
            c = np.nan_to_num(np.corrcoef(xb)[IU]); r[f"acorr_{BL[b]}"] = float(np.abs(c).mean())
            z = ss.hilbert(xb, axis=-1); im = (z[IU[0]] * np.conj(z[IU[1]])).imag
            r[f"wpli_{BL[b]}"] = float((np.abs(im.mean(-1)) / (np.abs(im).mean(-1) + 1e-12)).mean())
        rows.append(r)
    return rows
with Pool(4) as p: R = [x for rr in p.map(job, JOBS, chunksize=4) for x in rr]
F = pd.DataFrame(R).merge(W, on=["dataset", "id", "window"])
F["person"] = [EP[i] if d == "epieeg60" else i for d, i in zip(F.dataset, F.id)]
F.to_csv(f"{OUT}/wincorr_windows.csv.gz", index=False)
print(f"окон {len(F)}, людей {F.groupby(['dataset','id']).ngroups} | {time.time()-T0:.0f} с")
FEATS = [c for c in F.columns if c.startswith(("acorr_", "wpli_"))] + ["std"]

# по человеку: среднее признака в верхних 20 % окон по REVE, в остальных и во всех
def agg(g):
    g = g.sort_values("p", ascending=False); k = max(1, int(np.ceil(0.2 * len(g))))
    o = {"y": g.y.iloc[0], "reve": g.p.iloc[:k].mean()}
    for f in FEATS:
        o[f"top_{f}"] = g[f].iloc[:k].mean(); o[f"rest_{f}"] = g[f].iloc[k:].mean() if len(g) > k else np.nan; o[f"all_{f}"] = g[f].mean()
    return pd.Series(o)
P = F.groupby(["dataset", "person"]).apply(agg).reset_index()
P.to_csv(f"{OUT}/wincorr_per_person.csv", index=False)
DS = ["nmt_eval", "turkey", "epieeg60"]

print("\n1) ROC AUC признака: больные против здоровых (>0,5 — у больных выше)")
T1 = []
for f in FEATS:
    r = {"признак": f}
    for d in DS:
        q = P[P.dataset == d]
        for part in ["top", "rest", "all"]:
            r[f"{d}:{part}"] = round(roc_auc_score(q.y, q[f"{part}_{f}"]), 3)
    T1.append(r)
T1 = pd.DataFrame(T1); T1.to_csv(f"{OUT}/wincorr_auc.csv", index=False)
pd.set_option("display.width", 300); print(T1.to_string(index=False))

print("\n2) Внутри человека: верхние окна минус остальные (медиана разницы, доля людей с ростом, p Уилкоксона)")
T2 = []
for f in FEATS:
    for d in DS:
        for cl in [1, 0]:
            q = P[(P.dataset == d) & (P.y == cl)].dropna(subset=[f"rest_{f}"]); dif = q[f"top_{f}"] - q[f"rest_{f}"]
            T2.append(dict(признак=f, набор=d, класс="больные" if cl else "здоровые", n=len(q), медиана=round(dif.median(), 4), доля_рост=round((dif > 0).mean(), 2), p=f"{wilcoxon(dif).pvalue:.1e}"))
T2 = pd.DataFrame(T2); T2.to_csv(f"{OUT}/wincorr_top_vs_rest.csv", index=False); print(T2.to_string(index=False))

print("\n3) Связь с амплитудой окна (Спирмен по окнам внутри класса)")
for f in FEATS[:-1]:
    print(f, {f"{d}:{cl}": round(spearmanr(F[(F.dataset == d) & (F.y == cl)][f], F[(F.dataset == d) & (F.y == cl)]["std"])[0], 2) for d in DS for cl in [0, 1]})

print("\n4) Второй голос: среди людей, которых REVE отметил «болен», отличает ли признак верные тревоги от ложных")
THR = {"nmt_eval": 0.832, "turkey": 0.348, "epieeg60": 0.810}   # порог NMT (Юден по val) и местный порог по 20 людям Турции
T4 = []
for f in FEATS:
    r = {"признак": f}
    for d in DS:
        q = P[(P.dataset == d)]; q = q[q.reve >= THR[d]]
        r[f"{d} (n={len(q)}, больных {int(q.y.sum())})"] = round(roc_auc_score(q.y, q[f"top_{f}"]), 3) if q.y.nunique() == 2 else np.nan
    T4.append(r)
T4 = pd.DataFrame(T4); T4.to_csv(f"{OUT}/wincorr_second_voice.csv", index=False); print(T4.to_string(index=False))
print("примечание: оценка REVE здесь — верхние 20 % по всем окнам, а не по 60; порог Турции 0,348 подобран на 20 людях этой же больницы")
print(f"всего {time.time()-T0:.0f} с")
