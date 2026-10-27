"""REVE v5 против v4 по csv (лёгкий расчёт, локально): парные разности ROC AUC, Спирмен оценка–амплитуда внутри классов,
калибровка порога на k турецких людях (как в ensemble_recall80.py), ансамбль с EEG-DINO.
Запуск из h1_eeg/results:  py ../src/reve_v5_compare.py"""
import sys, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve
from scipy.stats import spearmanr, rankdata
sys.stdout.reconfigure(encoding="utf-8")
pd.set_option("display.width", 250)
LAB = {"nmt_val": "отложенные 15 %", "nmt_eval": "NMT eval", "turkey": "Турция"}
B = 2000

v4 = pd.read_csv("reve_v4_per_person.csv"); v5 = pd.read_csv("reve_v5_per_person.csv")
dn = pd.read_csv("eegdino_finetune_per_person.csv")
dn["dino_prob"] = dn[["eegdino_seed0", "eegdino_seed1", "eegdino_seed2"]].mean(1)
key = ["dataset", "id", "y", "std_uv", "окна", "сборка"]
m = v4[key + ["reve4_ens_rank", "reve4_ens_prob"]].merge(v5[key + ["reve5_ens_rank", "reve5_ens_prob"]], on=key)
assert len(m) == len(v4) == len(v5), (len(m), len(v4), len(v5))
m = m.merge(dn[["dataset", "id", "dino_prob"]], on=["dataset", "id"], how="left")
view = lambda d, w, a: d[(d["окна"] == w) & (d["сборка"] == a)].reset_index(drop=True)

# 1. Парные разности ROC AUC v5 − v4 (ансамбль — средний ранг 3 запусков), бутстреп по людям
rows = []
for w in ["60 окон", "все окна"]:
    for a in ["верхние 20 %", "медиана"]:
        d0 = view(m, w, a)
        for ds in LAB:
            d = d0[d0.dataset == ds]; y = d.y.to_numpy(); s4, s5 = d.reve4_ens_rank.to_numpy(), d.reve5_ens_rank.to_numpy()
            r = np.random.default_rng(0); dv = []
            for _ in range(B):
                i = r.integers(0, len(y), len(y))
                if len(set(y[i])) == 2: dv.append(roc_auc_score(y[i], s5[i]) - roc_auc_score(y[i], s4[i]))
            a4, a5 = roc_auc_score(y, s4), roc_auc_score(y, s5); lo, hi = np.percentile(dv, [2.5, 97.5])
            rows.append({"окна": w, "сборка": a, "набор": LAB[ds], "n": len(y), "REVE v4": round(a4, 3), "REVE v5": round(a5, 3),
                         "v5 − v4": round(a5 - a4, 3), "95% ДИ": f"{lo:+.3f}…{hi:+.3f}"})
pdiff = pd.DataFrame(rows); pdiff.to_csv("reve_v5_paired_differences.csv", index=False)
print("1. ROC AUC, v5 − v4 (парный бутстреп по людям)\n", pdiff.to_string(index=False), "\n")

# 2. Спирмен оценка записи – амплитуда (std_uv) внутри классов: v4, v5, разность (парный бутстреп)
rows = []
for w in ["все окна", "60 окон"]:
    d0 = view(m, w, "верхние 20 %")
    for ds in LAB:
        for yv in [0, 1]:
            d = d0[(d0.dataset == ds) & (d0.y == yv)]; amp = d.std_uv.to_numpy(); s4, s5 = d.reve4_ens_prob.to_numpy(), d.reve5_ens_prob.to_numpy()
            r4, r5 = spearmanr(s4, amp).correlation, spearmanr(s5, amp).correlation
            r = np.random.default_rng(0); bb = []
            for _ in range(B):
                i = r.integers(0, len(amp), len(amp)); bb.append((spearmanr(s4[i], amp[i]).correlation, spearmanr(s5[i], amp[i]).correlation))
            bb = np.array(bb); q = lambda v: np.nanpercentile(v, [2.5, 97.5])
            (l4, h4), (l5, h5), (ld, hd) = q(bb[:, 0]), q(bb[:, 1]), q(bb[:, 1] - bb[:, 0])
            rows.append({"окна": w, "набор": LAB[ds], "класс": ("эпилепсия" if ds == "turkey" else "патология") if yv else "норма", "n": len(d),
                         "v4": round(r4, 2), "v4 ДИ": f"{l4:+.2f}…{h4:+.2f}", "v5": round(r5, 2), "v5 ДИ": f"{l5:+.2f}…{h5:+.2f}",
                         "v5 − v4": round(r5 - r4, 2), "разность ДИ": f"{ld:+.2f}…{hd:+.2f}", "_l4": l4, "_h4": h4, "_l5": l5, "_h5": h5})
spd = pd.DataFrame(rows)
spd.drop(columns=[c for c in spd if c.startswith("_")]).to_csv("reve_v5_amp_spearman_v4_vs_v5.csv", index=False)
print("2. Спирмен оценка (ансамбль, верхние 20 %) – амплитуда записи внутри классов\n",
      spd.drop(columns=[c for c in spd if c.startswith("_")]).to_string(index=False), "\n")

# 3. Порог по отложенным 15 % (Юден, recall 80 %) — перенос на NMT eval и Турцию: v5 и v5 + EEG-DINO (верхние 20 %, 60 окон)
def mets(y, p):
    tp = ((p == 1) & (y == 1)).sum(); fp = ((p == 1) & (y == 0)).sum(); fn = ((p == 0) & (y == 1)).sum(); tn = ((p == 0) & (y == 0)).sum()
    rec = tp / max(tp + fn, 1); spec = tn / max(tn + fp, 1)
    return dict(accuracy=(tp + tn) / len(y), bal=(rec + spec) / 2, precision=tp / (tp + fp) if tp + fp else np.nan, recall=rec, spec=spec)
def thr_youden(y, s): f, t, th = roc_curve(y, s); return th[np.argmax(t - f)]
def thr_sens(y, s, target=0.8): f, t, th = roc_curve(y, s); return th[np.where(t >= target)[0][0]]
def auc_ci(y, s):
    r = np.random.default_rng(0); v = []
    for _ in range(B):
        i = r.integers(0, len(y), len(y))
        if len(set(y[i])) == 2: v.append(roc_auc_score(y[i], s[i]))
    lo, hi = np.percentile(v, [2.5, 97.5]); return f"{roc_auc_score(y, s):.3f} ({lo:.2f}–{hi:.2f})"
d0 = view(m, "60 окон", "верхние 20 %").copy()
d0["ens5"] = (d0.reve5_ens_prob + d0.dino_prob) / 2; d0["ens4"] = (d0.reve4_ens_prob + d0.dino_prob) / 2
MODELS = [("reve5_ens_prob", "REVE v5 (верхние 20 %)"), ("ens5", "REVE v5 (верхние 20 %) + EEG-DINO"), ("ens4", "REVE v4 (верхние 20 %) + EEG-DINO")]
rows = []
for col, name in MODELS:
    v = d0[d0.dataset == "nmt_val"]
    for rule, f in [("Юден", thr_youden), ("recall 80 %", thr_sens)]:
        t = f(v.y.values, v[col].values)
        for ds in LAB:
            e = d0[d0.dataset == ds]; ye, se = e.y.values, e[col].values; pr = (se >= t).astype(int); mm = mets(ye, pr)
            r = np.random.default_rng(0); bb = pd.DataFrame([mets(ye[i], pr[i]) for i in (r.integers(0, len(ye), len(ye)) for _ in range(B))])
            ci = {k: np.nanpercentile(bb[k], [2.5, 97.5]) for k in mm}
            rows.append({"модель": name, "порог": rule, "значение порога": round(float(t), 3), "набор": LAB[ds], "n": len(ye), "ROC AUC": auc_ci(ye, se),
                         **{nm: f"{mm[k]:.2f} ({ci[k][0]:.2f}–{ci[k][1]:.2f})" for k, nm in
                            [("accuracy", "accuracy"), ("bal", "сбаланс. accuracy"), ("precision", "precision"), ("recall", "recall"), ("spec", "специфичность")]}})
thr = pd.DataFrame(rows); thr.to_csv("reve_v5_ensemble_threshold_metrics.csv", index=False)
print("3. Порог по отложенным 15 %, перенос без изменений (60 окон, верхние 20 %)\n", thr.to_string(index=False), "\n")

# 4. Калибровка порога на k турецких людях (Юден по k людям, метрики на остальных), 500 повторов — для v4 и v5 одни и те же выборки
t = d0[d0.dataset == "turkey"].reset_index(drop=True); y = t.y.values
CAL = [("reve4_ens_prob", "REVE v4 (верхние 20 %)"), ("reve5_ens_prob", "REVE v5 (верхние 20 %)"), ("dino_prob", "EEG-DINO"),
       ("ens4", "REVE v4 (верхние 20 %) + EEG-DINO"), ("ens5", "REVE v5 (верхние 20 %) + EEG-DINO")]
res = []
for k in [10, 20, 30]:
    rng = np.random.default_rng(0)
    for rep in range(500):
        idx = rng.permutation(len(t)); cal, test = idx[:k], idx[k:]
        if len(set(y[cal])) < 2: continue
        for col, name in CAL:
            s = t[col].values
            for rule, f in [("Юден", thr_youden), ("recall 80 %", thr_sens)]:
                th = f(y[cal], s[cal]); res.append(dict(калибровка=f"{k} чел.", модель=name, порог=rule, rep=rep, **mets(y[test], (s[test] >= th).astype(int))))
R = pd.DataFrame(res)
g = R.drop(columns="rep").groupby(["калибровка", "порог", "модель"], sort=False)
cal_tab = (g.mean().round(2).astype(str) + " (" + g.quantile(.025).round(2).astype(str) + "–" + g.quantile(.975).round(2).astype(str) + ")")
cal_tab = cal_tab.rename(columns={"bal": "сбаланс. accuracy", "spec": "специфичность"}).reset_index()
cal_tab.to_csv("reve_v5_turkey_local_threshold_calibration.csv", index=False)
print("4. Калибровка порога на k людях Турции (среднее и 2,5–97,5 % по 500 повторам)\n",
      cal_tab[cal_tab["порог"] == "Юден"].to_string(index=False), "\n")
# парная разность сбаланс. accuracy v5 − v4 по тем же повторам (k = 20, Юден)
for a_, b_ in [("REVE v5 (верхние 20 %)", "REVE v4 (верхние 20 %)"), ("REVE v5 (верхние 20 %) + EEG-DINO", "REVE v4 (верхние 20 %) + EEG-DINO")]:
    q = R[(R["калибровка"] == "20 чел.") & (R["порог"] == "Юден")].pivot(index="rep", columns="модель", values="bal")
    dd = q[a_] - q[b_]; print(f"   k=20, Юден, сбаланс. accuracy: {a_} − {b_.split(' (')[0]}: {dd.mean():+.3f} (2,5–97,5 %: {dd.quantile(.025):+.2f}…{dd.quantile(.975):+.2f})")

# 5. Картинки: Спирмен v4 / v5 с ДИ; оценка против амплитуды (Турция, NMT eval)
C4, C5 = "#2a78d6", "#eb6834"
d = spd[spd["окна"] == "все окна"].reset_index(drop=True)
lab = [f"{r['набор']}\n{r['класс']}" for _, r in d.iterrows()]; x = np.arange(len(d))
fig, ax = plt.subplots(figsize=(10, 4.2))
for off, c, col, nm in [(-0.17, C4, "v4", "REVE v4"), (0.17, C5, "v5", "REVE v5 (аугментации спектра)")]:
    lo, hi = d[f"_l{col[-1]}"], d[f"_h{col[-1]}"]
    ax.errorbar(x + off, d[col], yerr=[d[col] - lo, hi - d[col]], fmt="o", ms=8, color=c, capsize=4, lw=2, label=nm)
ax.axhline(0, color="#888", lw=1); ax.set_xticks(x, lab, fontsize=9); ax.set_ylabel("Спирмен: оценка – амплитуда записи")
ax.set_title("Связь оценки с амплитудой внутри классов (все окна, верхние 20 %, 95% ДИ)", fontsize=11)
ax.spines[["top", "right"]].set_visible(False); ax.grid(axis="y", color="#e5e5e5"); ax.legend(frameon=False, loc="upper left")
fig.tight_layout(); fig.savefig("figures/reve_v5_amp_spearman.png", dpi=130); plt.close(fig)

d1 = view(m, "все окна", "верхние 20 %")
fig, axs = plt.subplots(2, 2, figsize=(10, 7.5), sharey=True)
for j, ds in enumerate(["turkey", "nmt_eval"]):
    for i_, (col, nm) in enumerate([("reve4_ens_prob", "REVE v4"), ("reve5_ens_prob", "REVE v5")]):
        ax = axs[j, i_]; d = d1[d1.dataset == ds]
        for yv, c, lb in [(0, C4, "норма"), (1, C5, "эпилепсия" if ds == "turkey" else "патология")]:
            e = d[d.y == yv]; ax.scatter(e.std_uv, e[col], s=22, color=c, alpha=0.75, edgecolor="white", lw=0.5,
                                         label=f"{lb}: ρ = {spearmanr(e[col], e.std_uv).correlation:+.2f}")
        ax.set_xscale("log"); ax.set_title(f"{nm}, {LAB[ds]}", fontsize=10); ax.legend(frameon=False, fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)
        if j == 1: ax.set_xlabel("амплитуда записи, СКО (мкВ)")
        if i_ == 0: ax.set_ylabel("оценка записи (вероятность)")
fig.tight_layout(); fig.savefig("figures/reve_v5_amp_scatter.png", dpi=130); plt.close(fig)
print("картинки: figures/reve_v5_amp_spearman.png, figures/reve_v5_amp_scatter.png")
