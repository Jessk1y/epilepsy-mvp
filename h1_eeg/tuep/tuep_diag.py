"""Разбор результата TUEP после основного прогона (поисковый, не меняет PROTOCOL.md): почему ROC AUC около 0,5.

    python tuep_diag.py --work ~/data/tuep_work --root ~/data/TUEP

Проверяет: (1) реагируют ли оценки на патологию вообще — сеансы с приступами против сеансов без них у тех же больных;
(2) контроль с инсультом / деменцией против остального контроля (модель обучена «патология / норма» на NMT);
(3) возраст из заголовков EDF: связь с оценкой и сам по себе; (4) распределения оценок по группам и доля выше порога NMT.
Выход — только сводные цифры (out/tuep_diag.csv), без идентификаторов пациентов.
"""
import argparse, os, re, sys
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tuep_protocol as TP


def edf_age(path):
    """Возраст из заголовка EDF TUH: поле пациента содержит «Age:NN»."""
    with open(path, "rb") as f:
        h = f.read(88).decode("latin-1")
    m = re.search(r"Age:(\d+)", h)
    return int(m.group(1)) if m else np.nan


def auc(y, s):
    y, s = np.asarray(y), np.asarray(s, float)
    m = ~np.isnan(s)
    return round(roc_auc_score(y[m], s[m]), 3) if len(set(y[m])) == 2 else np.nan


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--work", required=True); ap.add_argument("--root", required=True)
    a = ap.parse_args()
    W = os.path.expanduser(a.work); R = os.path.expanduser(a.root)
    rec = pd.read_csv(f"{W}/out/tuep_record_scores.csv")
    meta = pd.read_csv(f"{W}/prep/h1_shared_meta.csv")
    inv = pd.read_csv(f"{W}/inventory/inventory_files.csv")
    rec = rec.drop(columns=[c for c in ("n_seiz", "year", "montage") if c in rec]).merge(meta[["id", "n_seiz", "year", "montage", "sfreq_orig"]], on="id", how="left")
    S = TP.session_table(rec, ["reve4", "dino2", "ens"])
    # возраст по сеансу — из первого EDF сеанса
    inv["skey"] = inv.patient + "/" + inv.session
    first = inv.groupby("skey").path.first()
    S["age"] = [edf_age(os.path.join(R, first.get(f"{p}/{s}", ""))) if f"{p}/{s}" in first else np.nan for p, s in zip(S.patient, S.session)]
    # метаданные (по сеансу: «aaaaaanr_s001»)
    md = pd.read_excel(os.path.join(R, "DOCS", "metadata_v00r.xlsx"))
    md["key"] = md["Subject ID"].astype(str).str.strip()
    S["key"] = S.patient + "_" + S.session.str[:4]
    for c in ("Stroke", "Dementia", "TBI", "Epilepsy", "Electrographic Seizure"):
        if c in md:
            S[c] = S.key.map(md.set_index("key")[c].astype(str).str.upper().eq("Y"))
    rows = []
    add = lambda k, v, n="": rows.append(dict(проверка=k, значение=v, n=n))
    # (4) распределения
    for g, d in S.groupby("y"):
        add(f"медиана оценки пары, сеансы {'эпилепсии' if g else 'контроля'}", round(d.ens.median(), 3), len(d))
        add(f"доля сеансов выше порога NMT 0,755 (пара), {'эпилепсия' if g else 'контроль'}", round((d.ens >= 0.755).mean(), 3), len(d))
    # (1) приступы: у больных с приступами — сеансы с приступами против их же сеансов без приступов
    E = S[S.y == 1]; pz = E[E.has_seiz].patient.unique(); Z = E[E.patient.isin(pz)]
    add("AUC: сеансы с приступами против сеансов без них (те же больные), пара", auc(Z.has_seiz.astype(int), Z.ens), len(Z))
    # (2) контроль: инсульт / деменция против остального контроля
    C = S[S.y == 0]
    for c in ("Stroke", "Dementia"):
        if c in C:
            add(f"AUC: контроль с «{c}» против остального контроля, пара", auc(C[c].astype(int), C.ens), f"{int(C[c].sum())} из {len(C)}")
    # эпилепсия против «чистого» контроля без инсульта и деменции (по сеансам, все сеансы — поисково)
    if "Stroke" in S and "Dementia" in S:
        clean = S[(S.y == 1) | (~S.Stroke & ~S.Dementia)]
        add("AUC по сеансам: эпилепсия против контроля без инсульта и деменции, пара", auc(clean.y, clean.ens), len(clean))
    add("AUC по сеансам: эпилепсия против всего контроля, пара", auc(S.y, S.ens), len(S))
    # (3) возраст
    m = S.age.notna()
    add("сеансов с возрастом в заголовке", int(m.sum()), len(S))
    if m.sum() > 10:
        add("медиана возраста, эпилепсия / контроль", f"{S[m & (S.y == 1)].age.median():.0f} / {S[m & (S.y == 0)].age.median():.0f}")
        add("AUC: возраст сам по себе (выше — эпилепсия)", auc(S.y[m], S.age[m]), int(m.sum()))
        r, p = spearmanr(S.age[m], S.ens[m]); add("Спирмен: возраст и оценка пары (все сеансы)", f"{r:.2f} (p={p:.3f})", int(m.sum()))
        for g in (0, 1):
            mm = m & (S.y == g); r, p = spearmanr(S.age[mm], S.ens[mm])
            add(f"Спирмен: возраст и оценка, {'эпилепсия' if g else 'контроль'}", f"{r:.2f} (p={p:.3f})", int(mm.sum()))
    T = pd.DataFrame(rows); T.to_csv(f"{W}/out/tuep_diag.csv", index=False)
    pd.set_option("display.width", 200); print(T.to_string(index=False))


if __name__ == "__main__":
    main()
