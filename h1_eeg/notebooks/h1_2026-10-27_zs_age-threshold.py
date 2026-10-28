"""Г1, REVE v4: отдельный порог для детей (<18) и взрослых (≥18) против одного общего.

Захар Санников, 27.10.2026. Лёгкий расчёт по готовым таблицам (без сырых записей):
- results/reve_v4_per_person.csv — REVE v4, 60 окон, верхние 20 %, среднее вероятностей 3 запусков (reve4_ens_prob);
- results/eegdino_finetune_per_person.csv — EEG-DINO, среднее вероятностей 3 запусков; ансамбль = (REVE + EEG-DINO) / 2
  (как в src/reve_v5_compare.py);
- Labels.csv набора Kaggle mhmdrdwn/nmt-eeg (age) — в git не кладётся, путь передаётся аргументом.
Пороги (Юден, recall 80 %) выбираются только на NMT val (отложенные 15 % NMT train) и без изменений переносятся на NMT eval:
(1) один общий; (2) свой для детей и для взрослых. Три группы (<12, 12–17, ≥18) не считаются: в NMT val у 12–17 лет 9 больных.

Запуск: py notebooks/h1_2026-10-27_zs_age-threshold.py <путь к Labels.csv>
Выход: results/age_threshold_counts.csv, age_threshold_values.csv, age_threshold_metrics.csv, age_threshold_paired.csv,
       results/figures/age_threshold.png
"""
import sys, os
import numpy as np, pandas as pd
from sklearn.metrics import roc_curve
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.stdout.reconfigure(encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
FIG = os.path.join(RES, "figures")
B, SEED = 2000, 0
GRP = {1: "дети <18", 0: "взрослые ≥18"}

lab = pd.read_csv(sys.argv[1])
v4 = pd.read_csv(os.path.join(RES, "reve_v4_per_person.csv"))
v4 = v4[v4.dataset.isin(["nmt_val", "nmt_eval"]) & (v4["окна"] == "60 окон") & (v4["сборка"] == "верхние 20 %")]
dn = pd.read_csv(os.path.join(RES, "eegdino_finetune_per_person.csv"))
dn["dino_prob"] = dn[["eegdino_seed0", "eegdino_seed1", "eegdino_seed2"]].mean(1)
d = (v4[["dataset", "id", "y", "reve4_ens_prob"]].merge(dn[["dataset", "id", "dino_prob"]], on=["dataset", "id"], how="left")
     .merge(lab[["recordname", "label", "age"]], left_on="id", right_on="recordname", how="left"))
assert d.age.notna().all() and d.dino_prob.notna().all() and ((d.label == "abnormal").astype(int) == d.y).all()
d["ens4"] = (d.reve4_ens_prob + d.dino_prob) / 2
d["child"] = (d.age < 18).astype(int)
d["age3"] = pd.cut(d.age, [-np.inf, 12, 18, np.inf], right=False, labels=["<12", "12–17", "≥18"])
MODELS = [("reve4_ens_prob", "REVE v4 (верхние 20 %)"), ("ens4", "REVE v4 (верхние 20 %) + EEG-DINO")]
val = d[d.dataset == "nmt_val"].reset_index(drop=True)
ev = d[d.dataset == "nmt_eval"].reset_index(drop=True)

# 0) Сколько людей в группах
cnt = (d.assign(набор=d.dataset.map({"nmt_val": "NMT val", "nmt_eval": "NMT eval"}), класс=d.y.map({0: "норма", 1: "патология"}))
       .pivot_table(index=["набор", "класс"], columns="age3", values="id", aggfunc="size", observed=False).reset_index())
cnt["дети <18"] = cnt["<12"] + cnt["12–17"]
cnt = cnt.rename(columns={"≥18": "взрослые ≥18"})
cnt.to_csv(os.path.join(RES, "age_threshold_counts.csv"), index=False)


def thr_youden(y, s):
    f, t, th = roc_curve(y, s); return th[np.argmax(t - f)]


def thr_sens(y, s, target=0.8):
    f, t, th = roc_curve(y, s); return th[np.where(t >= target)[0][0]]


RULES = [("Юден", thr_youden), ("recall 80 %", thr_sens)]


def thresholds(y, s, ch, rule):
    """(общий, порог для детей, порог для взрослых)."""
    return rule(y, s), rule(y[ch == 1], s[ch == 1]), rule(y[ch == 0], s[ch == 0])


def predict(s, ch, t, scheme):
    return (s >= (t[0] if scheme == "общий" else np.where(ch == 1, t[1], t[2]))).astype(int)


def mets(y, p):
    tp = ((p == 1) & (y == 1)).sum(); fp = ((p == 1) & (y == 0)).sum(); fn = ((p == 0) & (y == 1)).sum(); tn = ((p == 0) & (y == 0)).sum()
    rec = tp / (tp + fn) if tp + fn else np.nan; spec = tn / (tn + fp) if tn + fp else np.nan
    return dict(accuracy=(tp + tn) / len(y), bal=(rec + spec) / 2, precision=tp / (tp + fp) if tp + fp else np.nan, recall=rec, spec=spec)


def strata_idx(y, ch):
    return [np.flatnonzero((y == a) & (ch == b)) for a in (0, 1) for b in (0, 1)]


def boot(strata, rng):
    """Бутстреп по людям внутри (класс × дети/взрослые): размеры групп и доли классов сохраняются."""
    return np.concatenate([rng.choice(g, len(g)) for g in strata])


SCHEMES = ["общий", "дети/взрослые"]
SUBS = ["все", GRP[1], GRP[0]]
KEYS = ["accuracy", "bal", "precision", "recall", "spec"]
NAMES = dict(accuracy="accuracy", bal="сбаланс. accuracy", precision="precision", recall="recall", spec="специфичность")
fmt = lambda v, lo, hi: f"{v:.2f} ({lo:.2f}–{hi:.2f})"

rows_thr, rows_m, rows_d, plot = [], [], [], []
yv, cv, ye, ce = val.y.values, val.child.values, ev.y.values, ev.child.values
st_v, st_e = strata_idx(yv, cv), strata_idx(ye, ce)
for col, mname in MODELS:
    sv, se = val[col].values, ev[col].values
    for rname, rule in RULES:
        t = thresholds(yv, sv, cv, rule)
        # устойчивость порогов: бутстреп NMT val
        rng = np.random.default_rng(SEED)
        tb = np.array([thresholds(yv[i], sv[i], cv[i], rule) for i in (boot(st_v, rng) for _ in range(B))])
        for k, nm in enumerate(["общий", "дети <18", "взрослые ≥18"]):
            m = np.ones(len(yv), bool) if k == 0 else (cv == (1 if k == 1 else 0))
            lo, hi = np.percentile(tb[:, k], [2.5, 97.5])
            rows_thr.append({"модель": mname, "правило": rname, "порог для": nm, "n_пат_val": int(yv[m].sum()), "n_норм_val": int((1 - yv[m]).sum()),
                             "порог": round(float(t[k]), 3), "ДИ порога (бутстреп val)": f"{lo:.3f}–{hi:.3f}",
                             "recall на val": round(float((sv[m & (yv == 1)] >= t[k]).mean()), 2),
                             "специфичность на val": round(float((sv[m & (yv == 0)] < t[k]).mean()), 2)})
        # метрики на NMT eval: пороги фиксированы, бутстреп по людям eval
        pred = {sc: predict(se, ce, t, sc) for sc in SCHEMES}
        mask = {"все": np.ones(len(ye), bool), GRP[1]: ce == 1, GRP[0]: ce == 0}
        est = {(sc, g): mets(ye[mask[g]], pred[sc][mask[g]]) for sc in SCHEMES for g in SUBS}
        rng = np.random.default_rng(SEED)
        bs = {k: [] for k in est}; dbs = {g: [] for g in SUBS}
        for _ in range(B):
            i = boot(st_e, rng)
            for sc in SCHEMES:
                for g in SUBS:
                    j = i[mask[g][i]]
                    bs[(sc, g)].append(mets(ye[j], pred[sc][j]))
            for g in SUBS:
                dbs[g].append(bs[("дети/взрослые", g)][-1]["bal"] - bs[("общий", g)][-1]["bal"])
        # та же разность, но пороги заново выбираются на бутстреп-выборке val (учёт неустойчивости порогов)
        rng = np.random.default_rng(SEED + 1)
        dfull = {g: [] for g in SUBS}
        for _ in range(B):
            iv, ie = boot(st_v, rng), boot(st_e, rng)
            tt = thresholds(yv[iv], sv[iv], cv[iv], rule)
            for g in SUBS:
                j = ie[mask[g][ie]]
                dfull[g].append(mets(ye[j], predict(se[j], ce[j], tt, "дети/взрослые"))["bal"] - mets(ye[j], predict(se[j], ce[j], tt, "общий"))["bal"])
        for sc in SCHEMES:
            for g in SUBS:
                bb = pd.DataFrame(bs[(sc, g)]); e = est[(sc, g)]
                row = {"модель": mname, "правило": rname, "схема порога": sc, "группа": g,
                       "n_пат": int(ye[mask[g]].sum()), "n_норм": int((1 - ye[mask[g]]).sum()),
                       "пороги": f"{t[0]:.3f}" if sc == "общий" else f"дети {t[1]:.3f} / взрослые {t[2]:.3f}"}
                for k in KEYS:
                    lo, hi = np.nanpercentile(bb[k], [2.5, 97.5]); row[NAMES[k]] = fmt(e[k], lo, hi)
                rows_m.append(row)
                plot.append(dict(модель=mname, правило=rname, схема=sc, группа=g, bal=e["bal"], lo=np.nanpercentile(bb.bal, 2.5),
                                 hi=np.nanpercentile(bb.bal, 97.5)))
        for g in SUBS:
            dd = est[("дети/взрослые", g)]["bal"] - est[("общий", g)]["bal"]
            lo, hi = np.nanpercentile(dbs[g], [2.5, 97.5]); lo2, hi2 = np.nanpercentile(dfull[g], [2.5, 97.5])
            rows_d.append({"модель": mname, "правило": rname, "группа": g, "сбаланс. accuracy общий": round(est[("общий", g)]["bal"], 3),
                           "сбаланс. accuracy дети/взрослые": round(est[("дети/взрослые", g)]["bal"], 3), "разность (2) − (1)": round(dd, 3),
                           "ДИ (пороги фиксированы)": f"{lo:+.3f}…{hi:+.3f}", "ДИ (пороги заново на бутстрепе val)": f"{lo2:+.3f}…{hi2:+.3f}"})

thr_tab, met_tab, dif_tab = pd.DataFrame(rows_thr), pd.DataFrame(rows_m), pd.DataFrame(rows_d)
thr_tab.to_csv(os.path.join(RES, "age_threshold_values.csv"), index=False)
met_tab.to_csv(os.path.join(RES, "age_threshold_metrics.csv"), index=False)
dif_tab.to_csv(os.path.join(RES, "age_threshold_paired.csv"), index=False)

# Рисунок: сбалансированная accuracy на NMT eval, общий порог против отдельных (95% ДИ)
P = pd.DataFrame(plot)
fig, axs = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
C = {"общий": "#2a78d6", "дети/взрослые": "#eb6834"}
for ax, (col, mname) in zip(axs, MODELS):
    x = 0
    ticks, labs = [], []
    for rname, _ in RULES:
        for g in SUBS:
            for off, sc in [(-0.15, "общий"), (0.15, "дети/взрослые")]:
                r = P[(P.модель == mname) & (P.правило == rname) & (P.группа == g) & (P.схема == sc)].iloc[0]
                ax.errorbar(x + off, r.bal, yerr=[[r.bal - r.lo], [r.hi - r.bal]], fmt="o", ms=7, color=C[sc], capsize=3, lw=1.8,
                            label=("общий порог" if sc == "общий" else "отдельные пороги дети/взрослые") if x == 0 else None)
            ticks.append(x); labs.append(f"{rname}\n{g}"); x += 1
        x += 0.5
    ax.set_xticks(ticks, labs, fontsize=8); ax.set_title(mname, fontsize=10); ax.grid(axis="y", color="#e5e5e5")
    ax.spines[["top", "right"]].set_visible(False)
axs[0].set_ylabel("сбаланс. accuracy, NMT eval (95% ДИ)"); axs[0].legend(frameon=False, fontsize=8, loc="lower left")
fig.suptitle("Пороги выбраны на NMT val и перенесены на NMT eval без изменений", fontsize=10)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "age_threshold.png"), dpi=130); plt.close(fig)

pd.set_option("display.width", 250, "display.max_columns", 30, "display.max_colwidth", 40)
print(cnt.to_string(index=False), "\n"); print(thr_tab.to_string(index=False), "\n")
print(met_tab.to_string(index=False), "\n"); print(dif_tab.to_string(index=False))
