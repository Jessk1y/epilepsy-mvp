# Г1: порог для новой больницы без разметки — медиана оценок её же записей.
# Честное деление: порог по медиане одной половины людей (метки не нужны), метрики — на другой половине; 500 повторов.
# Кривая по доле больных: из набора берётся подвыборка с заданной долей больных, дальше то же самое.
# Вход: results/eegdino_v2_per_person.csv (REVE v4 и пара REVE v4 + EEG-DINO v2, сборка — верхние 20 % окон). Без GPU, секунды.
import numpy as np, pandas as pd, pathlib
R = pathlib.Path(__file__).resolve().parents[1] / "results"
P = pd.read_csv(R / "eegdino_v2_per_person.csv")
MODELS = {"REVE v4": "reve4_ens_prob", "REVE v4 + EEG-DINO v2": "ens_reve4_dino2"}
NMT_THR = {"REVE v4": 0.832}
rng = np.random.default_rng(0)

def metr(y, s, t):
    p = s >= t; tp = (p & (y == 1)).sum(); tn = (~p & (y == 0)).sum()
    se = tp / max((y == 1).sum(), 1); sp = tn / max((y == 0).sum(), 1)
    return se, sp, (se + sp) / 2

def run(y, s, n_rep=500):
    out = []
    for _ in range(n_rep):
        k = rng.permutation(len(y)); a, b = k[: len(y) // 2], k[len(y) // 2:]
        t = np.median(s[a]); out.append(metr(y[b], s[b], t))
    o = np.array(out)
    return o.mean(0), np.percentile(o[:, 2], [2.5, 97.5])

rows, curve = [], []
for ds in ["turkey", "epieeg60", "nmt_eval"]:
    g = P[P.dataset == ds]
    for m, c in MODELS.items():
        y, s = g.y.values, g[c].values
        (se, sp, ba), (lo, hi) = run(y, s)
        rows.append(dict(dataset=ds, model=m, n=len(y), prevalence=round(y.mean(), 2), threshold="медиана половины",
                         recall=round(se, 3), specificity=round(sp, 3), bal_acc=round(ba, 3), ci_low=round(lo, 3), ci_high=round(hi, 3)))
        if m in NMT_THR:
            se, sp, ba = metr(y, s, NMT_THR[m])
            rows.append(dict(dataset=ds, model=m, n=len(y), prevalence=round(y.mean(), 2), threshold="с NMT (0,832)",
                             recall=round(se, 3), specificity=round(sp, 3), bal_acc=round(ba, 3)))
        pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
        for prev in [0.1, 0.2, 0.3, 0.4, 0.5]:
            nn = len(neg); npos = int(round(prev / (1 - prev) * nn))
            if npos > len(pos): nn = int(round(len(pos) * (1 - prev) / prev)); npos = len(pos)
            res = []
            for _ in range(300):
                k = np.r_[rng.choice(pos, npos, replace=False), rng.choice(neg, nn, replace=False)]
                (se, sp, ba), _ = run(y[k], s[k], n_rep=1); res.append((se, sp, ba))
            r = np.array(res).mean(0)
            curve.append(dict(dataset=ds, model=m, prevalence=prev, n=npos + nn, recall=round(r[0], 3), specificity=round(r[1], 3), bal_acc=round(r[2], 3)))

T = pd.DataFrame(rows); C = pd.DataFrame(curve)
T.to_csv(R / "unlabeled_threshold_metrics.csv", index=False); C.to_csv(R / "unlabeled_threshold_prevalence.csv", index=False)
print(T.to_string()); print(C.to_string())
