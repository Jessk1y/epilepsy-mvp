import sys, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from scipy.stats import wilcoxon, spearmanr
F = pd.read_csv(sys.argv[1]); out = sys.argv[2]
FE = [c for c in F.columns if c.startswith(("acorr_", "wpli_"))] + ["std"]
def agg(g):
    g = g.sort_values("p", ascending=False); k = max(1, int(np.ceil(0.2 * len(g))))
    o = {"y": g.y.iloc[0], "reve": g.p.iloc[:k].mean(), "nwin": len(g)}
    for f in FE:
        o[f"top_{f}"] = g[f].iloc[:k].mean(); o[f"rest_{f}"] = g[f].iloc[k:].mean() if len(g) > k else np.nan; o[f"all_{f}"] = g[f].mean()
    return pd.Series(o)
P = F.groupby(["dataset", "person"])[[c for c in F.columns if c not in ("dataset", "person")]].apply(agg).reset_index()
P.to_csv(f"{out}/wincorr_per_person.csv", index=False)
DS = ["nmt_eval", "turkey", "epieeg60"]
print(P.groupby("dataset").agg(n=("y", "size"), больных=("y", "sum"), окон_медиана=("nwin", "median")))
pd.set_option("display.width", 300)
T1 = []
for f in FE:
    r = {"признак": f}
    for d in DS:
        q = P[P.dataset == d]
        for part in ["top", "all"]:
            qq = q.dropna(subset=[f"{part}_{f}"]); r[f"{d}:{part}"] = round(roc_auc_score(qq.y, qq[f"{part}_{f}"]), 2)
    T1.append(r)
T1 = pd.DataFrame(T1); T1.to_csv(f"{out}/wincorr_auc.csv", index=False)
print("\n1) AUC больные против здоровых (>0,5 у больных выше); top = верхние 20 % окон REVE, all = все окна\n", T1.to_string(index=False))
T2 = []
for f in FE:
    for d in DS:
        for cl in [1, 0]:
            q = P[(P.dataset == d) & (P.y == cl)].dropna(subset=[f"rest_{f}"]); dif = q[f"top_{f}"] - q[f"rest_{f}"]
            T2.append(dict(признак=f, набор=d, класс="больн" if cl else "здор", n=len(q), рост_доля=round((dif > 0).mean(), 2), p=f"{wilcoxon(dif).pvalue:.0e}"))
T2 = pd.DataFrame(T2); T2.to_csv(f"{out}/wincorr_top_vs_rest.csv", index=False)
print("\n2) верхние окна против остальных внутри человека: доля людей, у кого в верхних окнах выше\n",
      T2.pivot_table(index="признак", columns=["набор", "класс"], values="рост_доля", aggfunc="first").to_string())
print("\n3) Спирмен признака окна с СКО окна, внутри класса")
for f in FE[:-1]:
    print(f"{f:12s}", {f"{d[:4]}:{cl}": round(spearmanr(F[(F.dataset == d) & (F.y == cl)][f], F[(F.dataset == d) & (F.y == cl)]["std"])[0], 2) for d in DS for cl in [0, 1]})
THR = {"nmt_eval": 0.832, "turkey": 0.348, "epieeg60": 0.810}
T4 = []
for f in FE:
    r = {"признак": f}
    for d in DS:
        q = P[P.dataset == d]; q = q[q.reve >= THR[d]]
        r[f"{d} (n={len(q)}, верных {int(q.y.sum())})"] = round(roc_auc_score(q.y, q[f"top_{f}"]), 2)
    T4.append(r)
T4 = pd.DataFrame(T4); T4.to_csv(f"{out}/wincorr_second_voice.csv", index=False)
print("\n4) среди отмеченных REVE «болен»: отличает ли признак (в верхних окнах) верные тревоги от ложных\n", T4.to_string(index=False))
for d in DS:
    q = P[P.dataset == d]; print(d, "AUC REVE (верхние 20 % всех окон):", round(roc_auc_score(q.y, q.reve), 3))
