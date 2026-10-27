"""Г1, REVE v4: объясняет ли возраст (и пол) связь оценки с амплитудой и ошибки модели.

Захар Санников, 27.10.2026. Лёгкий расчёт по готовым таблицам (без сырых записей):
- results/reve_v4_errors_factors.csv — оценка по человеку, порог, группа TP/FP/TN/FN, амплитуда (СКО, мкВ) из разбора ошибок;
- Labels.csv набора Kaggle mhmdrdwn/nmt-eeg (age, gender) — в git не кладётся, путь передаётся аргументом.
Турецкий набор: в .mat только data и label, в описании на Kaggle возраста по людям нет -> только NMT.

Запуск: py notebooks/h1_2026-10-27_zs_age-check.py <путь к Labels.csv>
Выход: results/age_check_*.csv, results/figures/age_check.png
"""
import sys, os
import numpy as np, pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
FIG = os.path.join(RES, "figures")
os.makedirs(FIG, exist_ok=True)
B, SEED = 2000, 0
NAMES = {"nmt_val": "NMT val (15 % train)", "nmt_eval": "NMT eval", "nmt_all": "NMT val+eval"}
AGE_BINS = [-1, 2, 5, 12, 17, 39, 59, 200]
AGE_LAB = ["0–2", "3–5", "6–12", "13–17", "18–39", "40–59", "60+"]

lab = pd.read_csv(sys.argv[1])
f = pd.read_csv(os.path.join(RES, "reve_v4_errors_factors.csv"))
f = f[f.dataset.str.startswith("nmt")].merge(lab[["recordname", "label", "age", "gender"]], left_on="id", right_on="recordname", how="left")
assert f.age.notna().all() and ((f.label == "abnormal").astype(int) == f.y).all()
f["child"] = (f.age < 18).astype(int)
f["age_bin"] = pd.cut(f.age, AGE_BINS, labels=AGE_LAB)
f["log_amp"] = np.log(f.amp_uv)
f = pd.concat([f, f.assign(dataset="nmt_all")], ignore_index=True)
rng = np.random.default_rng(SEED)


def boot_idx(d, strat="y"):
    """Бутстреп по людям внутри класса (сохраняет доли классов)."""
    groups = [np.flatnonzero(d[strat].values == v) for v in np.unique(d[strat])]
    for _ in range(B):
        yield np.concatenate([rng.choice(g, len(g)) for g in groups])


def ci(vals):
    vals = np.asarray(vals, float)
    vals = vals[np.isfinite(vals)]
    return np.percentile(vals, [2.5, 97.5]) if len(vals) else (np.nan, np.nan)


def resid(v, X):
    X = np.column_stack([np.ones(len(v)), X])
    return v - X @ np.linalg.lstsq(X, v, rcond=None)[0]


def partial_sp(d, how):
    """Частичный Спирмен оценка–амплитуда при поправке на возраст.
    linear: ранги оценки и амплитуды очищаются от ранга возраста; bins: от 7 возрастных групп (нелинейная поправка)."""
    rs, ra = stats.rankdata(d.score), stats.rankdata(d.amp_uv)
    if how == "linear":
        X = stats.rankdata(d.age)[:, None]
    else:
        X = pd.get_dummies(d.age_bin, drop_first=True).values.astype(float)
    return stats.pearsonr(resid(rs, X), resid(ra, X))[0]


def sp(a, b):
    return stats.spearmanr(a, b)[0]


# 1) Связи внутри класса
rows = []
for ds, g in f.groupby("dataset"):
    for y, d in g.groupby("y"):
        cls = "патология" if y else "норма"
        d = d.reset_index(drop=True)
        r2 = 1 - resid(d.log_amp.values, pd.get_dummies(d.age_bin, drop_first=True).values.astype(float)).var() / d.log_amp.var()
        est = {
            "возраст–амплитуда": sp(d.age, d.amp_uv),
            "возраст–оценка": sp(d.age, d.score),
            "оценка–амплитуда": sp(d.score, d.amp_uv),
            "оценка–амплитуда | возраст (ранг)": partial_sp(d, "linear"),
            "оценка–амплитуда | возраст (7 групп)": partial_sp(d, "bins"),
            "оценка–амплитуда, дети <18": sp(d[d.child == 1].score, d[d.child == 1].amp_uv),
            "оценка–амплитуда, взрослые": sp(d[d.child == 0].score, d[d.child == 0].amp_uv),
            "R² log(амплитуды) от возраста (7 групп)": r2,
        }
        bs = {k: [] for k in est}
        for idx in boot_idx(d.assign(_s=0), "_s"):
            b = d.iloc[idx].reset_index(drop=True)
            bs["возраст–амплитуда"].append(sp(b.age, b.amp_uv))
            bs["возраст–оценка"].append(sp(b.age, b.score))
            bs["оценка–амплитуда"].append(sp(b.score, b.amp_uv))
            bs["оценка–амплитуда | возраст (ранг)"].append(partial_sp(b, "linear"))
            bs["оценка–амплитуда | возраст (7 групп)"].append(partial_sp(b, "bins"))
            c, a = b[b.child == 1], b[b.child == 0]
            bs["оценка–амплитуда, дети <18"].append(sp(c.score, c.amp_uv) if len(c) > 3 else np.nan)
            bs["оценка–амплитуда, взрослые"].append(sp(a.score, a.amp_uv) if len(a) > 3 else np.nan)
            bs["R² log(амплитуды) от возраста (7 групп)"].append(np.nan)
        for k, v in est.items():
            lo, hi = ci(bs[k])
            n = {"оценка–амплитуда, дети <18": d.child.sum(), "оценка–амплитуда, взрослые": (1 - d.child).sum()}.get(k, len(d))
            rows.append(dict(набор=NAMES[ds], класс=cls, n=int(n), показатель=k, значение=round(v, 3),
                             ДИ="" if np.isnan(lo) else f"{lo:+.2f}…{hi:+.2f}"))
corr = pd.DataFrame(rows)
corr.to_csv(os.path.join(RES, "age_check_correlations.csv"), index=False)

# 2) Возраст у ошибок против верных
rows = []
for ds, g in f[f.dataset != "nmt_all"].groupby("dataset"):
    for y, err, ok in [(0, "FP", "TN"), (1, "FN", "TP")]:
        d = g[g.y == y]
        e, c = d[d.group == err], d[d.group == ok]
        U, p = stats.mannwhitneyu(e.age, c.age)
        bs = []
        for _ in range(B):
            be, bc = rng.choice(e.age.values, len(e)), rng.choice(c.age.values, len(c))
            bs.append(stats.mannwhitneyu(be, bc)[0] / (len(e) * len(c)))
        lo, hi = ci(bs)
        rows.append(dict(набор=NAMES[ds], класс="норма" if y == 0 else "патология", сравнение=f"{err} против {ok}",
                         n_ошибок=len(e), n_верных=len(c),
                         медиана_возраста_ошибки=e.age.median(), медиана_возраста_верные=c.age.median(),
                         доля_детей_ошибки=round(e.child.mean(), 2), доля_детей_верные=round(c.child.mean(), 2),
                         MW_эффект=round(U / (len(e) * len(c)), 3), MW_ДИ=f"{lo:.2f}–{hi:.2f}", MW_p=p,
                         медиана_амплитуды_ошибки=round(e.amp_uv.median(), 1), медиана_амплитуды_верные=round(c.amp_uv.median(), 1)))
err_age = pd.DataFrame(rows)
# логистическая проверка: ошибка ~ log амплитуды + возраст (группы); коэффициенты на 1 СКО log амплитуды
import statsmodels.api as sm
lr_rows = []
for ds, g in f[f.dataset != "nmt_all"].groupby("dataset"):
    for y, err in [(0, "FP"), (1, "FN")]:
        d = g[g.y == y]
        z = (d.log_amp - d.log_amp.mean()) / d.log_amp.std()
        for name, X in [("только амплитуда", z.to_frame("амплитуда")),
                        ("амплитуда + возраст", pd.concat([z.rename("амплитуда"), d.child.rename("дети")], axis=1))]:
            m = sm.Logit((d.group == err).astype(int), sm.add_constant(X.astype(float))).fit(disp=0)
            for k in X.columns:
                lr_rows.append(dict(набор=NAMES[ds], ошибка=err, модель=name, признак=k, OR=round(np.exp(m.params[k]), 2),
                                    OR_ДИ="–".join(f"{v:.2f}" for v in np.exp(m.conf_int().loc[k])), p=m.pvalues[k]))
err_age.to_csv(os.path.join(RES, "age_check_errors.csv"), index=False)
pd.DataFrame(lr_rows).to_csv(os.path.join(RES, "age_check_errors_logit.csv"), index=False)


# 3) ROC AUC и пороговые метрики по возрасту и полу (порог — тот же, что в разборе ошибок)
def metrics(d):
    tp = ((d.pred == 1) & (d.y == 1)).sum(); fn = ((d.pred == 0) & (d.y == 1)).sum()
    tn = ((d.pred == 0) & (d.y == 0)).sum(); fp = ((d.pred == 1) & (d.y == 0)).sum()
    return dict(auc=roc_auc_score(d.y, d.score) if d.y.nunique() == 2 else np.nan,
                recall=tp / max(tp + fn, 1), spec=tn / max(tn + fp, 1))

rows = []
for ds, g in f[f.dataset != "nmt_all"].groupby("dataset"):
    g = g.reset_index(drop=True)
    subsets = {"все": g, "дети <18": g[g.child == 1], "взрослые ≥18": g[g.child == 0],
               "мужчины": g[g.gender == "male"], "женщины": g[g.gender == "female"]}
    boots = {k: [] for k in subsets}
    diff_age, diff_sex = [], []
    for idx in boot_idx(g):
        b = g.iloc[idx]
        masks = {"все": np.ones(len(b), bool), "дети <18": b.child.values == 1, "взрослые ≥18": b.child.values == 0,
                 "мужчины": b.gender.values == "male", "женщины": b.gender.values == "female"}
        r = {}
        for k in subsets:
            s = b[masks[k]]
            r[k] = metrics(s) if s.y.nunique() == 2 else dict(auc=np.nan, recall=np.nan, spec=np.nan)
            boots[k].append(r[k])
        diff_age.append(r["дети <18"]["auc"] - r["взрослые ≥18"]["auc"])
        diff_sex.append(r["мужчины"]["auc"] - r["женщины"]["auc"])
    for k, s in subsets.items():
        m = metrics(s)
        row = dict(набор=NAMES[ds], группа=k, n=len(s), n_пат=int(s.y.sum()), n_норм=int((1 - s.y).sum()))
        for mk, mn in [("auc", "ROC_AUC"), ("recall", "чувствительность"), ("spec", "специфичность")]:
            lo, hi = ci([x[mk] for x in boots[k]])
            row[mn] = round(m[mk], 3); row[mn + "_ДИ"] = f"{lo:.3f}–{hi:.3f}"
        rows.append(row)
    for name, dd, a, b_ in [("разница AUC дети − взрослые", diff_age, "дети <18", "взрослые ≥18"),
                            ("разница AUC мужчины − женщины", diff_sex, "мужчины", "женщины")]:
        lo, hi = ci(dd)
        est = metrics(subsets[a])["auc"] - metrics(subsets[b_])["auc"]
        rows.append(dict(набор=NAMES[ds], группа=name, ROC_AUC=round(est, 3), ROC_AUC_ДИ=f"{lo:+.3f}…{hi:+.3f}"))
auc = pd.DataFrame(rows)
auc.to_csv(os.path.join(RES, "age_check_auc_groups.csv"), index=False)

# 4) Пол: амплитуда и оценка внутри класса (NMT val+eval)
rows = []
for y, d in f[f.dataset == "nmt_all"].groupby("y"):
    m, w = d[d.gender == "male"], d[d.gender == "female"]
    for k in ["amp_uv", "score", "age"]:
        U, p = stats.mannwhitneyu(m[k], w[k])
        rows.append(dict(класс="патология" if y else "норма", показатель=k, n_муж=len(m), n_жен=len(w),
                         медиана_муж=round(m[k].median(), 3), медиана_жен=round(w[k].median(), 3),
                         MW_эффект_муж_больше=round(U / (len(m) * len(w)), 3), MW_p=p))
pd.DataFrame(rows).to_csv(os.path.join(RES, "age_check_sex.csv"), index=False)

# 5) Сводка по возрастным группам (NMT val+eval): медианы амплитуды и оценки
grp = (f[f.dataset == "nmt_all"].groupby(["y", "age_bin"], observed=True)
       .agg(n=("id", "size"), амплитуда_медиана=("amp_uv", "median"), оценка_медиана=("score", "median"),
            доля_ошибок=("group", lambda s: s.isin(["FP", "FN"]).mean())).round(3).reset_index())
grp["класс"] = grp.y.map({0: "норма", 1: "патология"})
grp.drop(columns="y").to_csv(os.path.join(RES, "age_check_by_agebin.csv"), index=False)

# Рисунок
a = f[f.dataset == "nmt_all"]
fig, ax = plt.subplots(1, 3, figsize=(16, 4.8))
col = {0: "#3b7dd8", 1: "#d8573b"}
for y in (0, 1):
    d = a[a.y == y]
    ax[0].scatter(d.age, d.amp_uv, s=10, alpha=.5, c=col[y], label="патология" if y else "норма")
ax[0].set_yscale("log"); ax[0].set_xlabel("возраст, лет"); ax[0].set_ylabel("амплитуда (СКО), мкВ"); ax[0].legend()
ax[0].set_title("Амплитуда от возраста (NMT val+eval)")
for y in (0, 1):
    for ch, mk in [(1, "o"), (0, "^")]:
        d = a[(a.y == y) & (a.child == ch)]
        ax[1].scatter(d.amp_uv, d.score, s=12, alpha=.5, c=col[y], marker=mk,
                      label=f"{'патология' if y else 'норма'}, {'дети' if ch else 'взрослые'}")
ax[1].set_xscale("log"); ax[1].set_xlabel("амплитуда (СКО), мкВ"); ax[1].set_ylabel("оценка REVE v4"); ax[1].legend(fontsize=7)
ax[1].set_title("Оценка от амплитуды: дети (●) и взрослые (▲)")
order = ["TN", "FP", "TP", "FN"]
ax[2].boxplot([a[a.group == g].age for g in order], labels=order, showfliers=False)
for i, g in enumerate(order):
    v = a[a.group == g].age
    ax[2].scatter(np.full(len(v), i + 1) + rng.uniform(-.15, .15, len(v)), v, s=6, alpha=.4, c="gray")
ax[2].axhline(18, ls="--", c="k", lw=.8); ax[2].set_ylabel("возраст, лет"); ax[2].set_title("Возраст по исходам (NMT val+eval)")
plt.tight_layout(); plt.savefig(os.path.join(FIG, "age_check.png"), dpi=130)

pd.set_option("display.width", 250, "display.max_columns", 30, "display.max_colwidth", 40)
print(corr.to_string(index=False)); print(err_age.to_string(index=False)); print(pd.DataFrame(lr_rows).to_string(index=False))
print(auc.to_string(index=False)); print(pd.read_csv(os.path.join(RES, "age_check_sex.csv")).to_string(index=False))
print(grp.to_string(index=False))
