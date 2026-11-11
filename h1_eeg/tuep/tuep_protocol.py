"""Задача 1.4: протокол оценки на TUEP — разбиение по пациентам, «один пациент — одна оценка», метрики, утечки, подгруппа «без разрядов».

    py tuep_protocol.py folds    --meta prep/h1_shared_meta.csv --out protocol/tuep_folds.csv      # 5 фолдов по пациентам
    py tuep_protocol.py leakage  --out protocol/model_leakage.csv                                 # таблица «предобучена на TUH?»
    py tuep_protocol.py subgroup --root v3.1.0 --meta prep/h1_shared_meta.csv --out protocol/subgroup_ied.csv [--ied-csv …]
    py tuep_protocol.py eval     --scores out/tuep_record_scores.csv --meta prep/h1_shared_meta.csv --folds protocol/tuep_folds.csv --results ../results --out out/

Правила — PROTOCOL.md (записаны до прогона, 11.11.2026). Коротко: один сеанс на пациента (случайный, seed 0), сеансы с приступами исключены;
проверка утечки через год записи и монтаж; все сеансы пациента — в одном фолде; ROC AUC и пороговые метрики с 95 % ДИ — бутстреп по людям;
порог не по тесту: с отложенных 15 % NMT train (Юден и чувствительность 80 %), перенесён без изменений;
дополнительно — калибровка порога на k случайных людях TUEP, метрики на остальных (500 повторов).
"""
import argparse, glob, os, re, sys
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold

# ---------------------------------------------------------------- утечки
# Данные по предобучению — со слов задачи (02.11.2026); первоисточники по каждой модели НЕ перепроверялись. Перед отчётом свериться со статьями.
LEAKAGE = [
    ("REVE", "да (предобучена на корпусе, включающем TUH)", "утечка возможна", "брать как «верхнюю оценку», не как независимый результат"),
    ("EEG-DINO", "да (предобучена на TUH)", "утечка возможна", "то же"),
    ("LaBraM", "да; видела сам TUEP", "утечка прямая", "не использовать как независимую проверку"),
    ("CBraMod", "да (предобучена на TUH)", "утечка возможна", "то же"),
    ("LUNA", "да (предобучена на TUH)", "утечка возможна", "то же"),
    ("BIOT", "по заданию — основной честный контроль", "проверить", "в статье BIOT есть варианты предобучения, в т. ч. на TUAB/TUEV (это TUH): уточнить, какой вариант и какие веса мы используем; пересечение пациентов TUEP с TUAB/TUEV не проверено"),
    ("EEGPT", "по заданию — основной честный контроль", "проверить", "уточнить корпуса предобучения по статье"),
]


def cmd_leakage(a):
    df = pd.DataFrame(LEAKAGE, columns=["модель", "предобучение на TUH", "вывод", "примечание"])
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    df.to_csv(a.out, index=False); print(df.to_string(index=False))


# ---------------------------------------------------------------- фолды
def patient_table(meta):
    """Одна строка на пациента; метка пациента должна быть единой."""
    g = meta.groupby("patient").y.agg(["min", "max", "size"])
    assert (g["min"] == g["max"]).all(), "пациент одновременно в двух группах: " + str(list(g.index[g["min"] != g["max"]]))
    return g["min"].rename("y").reset_index().merge(g["size"].rename("n_files").reset_index(), on="patient")


def make_folds(meta, n_folds=5, seed=0):
    P = patient_table(meta)
    P["fold"] = -1
    for k, (_, te) in enumerate(StratifiedKFold(n_folds, shuffle=True, random_state=seed).split(P.patient, P.y)):
        P.loc[te, "fold"] = k
    return P


def cmd_folds(a):
    meta = pd.read_csv(a.meta)
    P = make_folds(meta, a.n_folds, a.seed)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    P.to_csv(a.out, index=False)
    print(pd.crosstab(P.fold, P.y, margins=True))
    # проверка: сеансы пациента не разъезжаются (по построению — фолд на пациента), и нет пересечений
    chk = meta.merge(P[["patient", "fold"]], on="patient").groupby("patient").fold.nunique()
    assert (chk == 1).all()
    print("OK: все файлы и сеансы пациента в одном фолде")


# ---------------------------------------------------------------- один пациент — одна оценка (PROTOCOL.md, п. 2–3)
SESSION_RULES = ("random", "first", "last", "all")


def session_table(rec, score_cols):
    """Файлы -> сеансы: оценка сеанса = среднее по файлам с весом длительности; приступ хотя бы в одном файле — сеанс с приступами."""
    rec = rec.copy()
    rec["w"] = rec.dur_s if "dur_s" in rec else 1.0
    for c in ("n_seiz", "year", "montage"):
        if c not in rec:
            rec[c] = np.nan
    out = []
    for (p, s), g in rec.groupby(["patient", "session"]):
        row = dict(patient=p, session=s, y=int(g.y.iloc[0]), dur_s=g.w.sum(), n_files=len(g),
                   has_seiz=bool((g.n_seiz.fillna(0) > 0).any()), seiz_unknown=bool(g.n_seiz.isna().all()),
                   year=g.year.iloc[0], montage="|".join(sorted(set(map(str, g.montage)))))
        for c in score_cols:
            row[c] = np.average(g[c], weights=g.w)
        out.append(row)
    return pd.DataFrame(out)


def select_sessions(S, score_cols, rule="random", seed=0):
    """Один сеанс на пациента. Сеансы с приступами исключаются заранее; пациент без подходящих сеансов выпадает (список — второй выход).
    random — основной расчёт (seed 0): у больных в среднем 5,3 сеанса, у контроля 1,7, а ранние сеансы чаще старого монтажа
    02_tcp_le — выбор «первого» связал бы монтаж и год записи с классом. first / last / all — проверка устойчивости."""
    E = S[~S.has_seiz].sort_values(["patient", "session"])
    dropped = sorted(set(S.patient) - set(E.patient))
    if rule == "all":
        P = E.groupby("patient").agg(y=("y", "first"), dur_s=("dur_s", "sum"), n_sessions=("session", "size"), year=("year", "median"),
                                     montage=("montage", lambda x: "|".join(sorted(set("|".join(x).split("|"))))),
                                     **{c: (c, "mean") for c in score_cols}).reset_index()
        return P, dropped
    rng = np.random.default_rng(seed); pick = []
    for p, g in E.groupby("patient"):
        i = {"random": int(rng.integers(len(g))) if len(g) > 1 else 0, "first": 0, "last": len(g) - 1}[rule]
        pick.append(g.iloc[i])
    P = pd.DataFrame(pick).reset_index(drop=True)
    P["n_sessions"] = 1
    return P, dropped


def stratified_auc(y, s, strata):
    """ROC AUC только по парам «больной — контроль» внутри одного слоя (монтажа): убирает вклад монтажа в оценку."""
    num = den = 0.0
    for k in np.unique(strata):
        m = strata == k; yy, ss_ = y[m], s[m]
        npos, nneg = int(yy.sum()), int((1 - yy).sum())
        if npos and nneg:
            w = npos * nneg; num += roc_auc_score(yy, ss_) * w; den += w
    return num / den if den else np.nan


def confound_check(P):
    """Насколько класс угадывается без ЭЭГ — по году записи и монтажу (PROTOCOL.md, п. 5). Монтаж — доля эпилепсии в слое по той же
    выборке (оптимистично для монтажа, т. е. строго к нам)."""
    rows = []
    m = P.year.notna()
    if m.sum() and P.y[m].nunique() == 2:
        a = roc_auc_score(P.y[m], P.year[m].astype(float))
        rows.append(dict(признак="год записи", n=int(m.sum()), auc=round(a, 3), auc_any_direction=round(max(a, 1 - a), 3)))
    share = P.groupby("montage").y.transform("mean")
    a = roc_auc_score(P.y, share)
    rows.append(dict(признак="монтаж", n=len(P), auc=round(a, 3), auc_any_direction=round(max(a, 1 - a), 3)))
    return pd.DataFrame(rows), pd.crosstab(P.montage, P.y, margins=True)


# ---------------------------------------------------------------- метрики (как в блокнотах: ROC AUC + пороговые, бутстреп по людям)
def metr(y, pr):
    tp, tn = ((pr == 1) & (y == 1)).sum(), ((pr == 0) & (y == 0)).sum()
    fp, fn = ((pr == 1) & (y == 0)).sum(), ((pr == 0) & (y == 1)).sum()
    rec, spec = tp / max(tp + fn, 1), tn / max(tn + fp, 1)
    return dict(acc=(tp + tn) / len(y), bacc=(rec + spec) / 2, prec=tp / (tp + fp) if tp + fp else np.nan, rec=rec, spec=spec)


def boot_auc(y, s, n=2000, seed=0):
    r, v = np.random.default_rng(seed), []
    for _ in range(n):
        i = r.integers(0, len(y), len(y))
        if len(set(y[i])) == 2:
            v.append(roc_auc_score(y[i], s[i]))
    return roc_auc_score(y, s), *np.percentile(v, [2.5, 97.5])


def metr_ci(y, s, t, n=2000, seed=0):
    pr = (s >= t).astype(int); m0 = metr(y, pr); r = np.random.default_rng(seed); B = []
    for _ in range(n):
        i = r.integers(0, len(y), len(y)); B.append(metr(y[i], pr[i]))
    return {k: f"{m0[k]:.2f} ({np.nanpercentile([b[k] for b in B], 2.5):.2f}-{np.nanpercentile([b[k] for b in B], 97.5):.2f})" for k in m0}


def thr_youden(y, s):
    f, t, th = roc_curve(y, s); return float(th[np.argmax(t - f)])


def thr_recall80(y, s):
    f, t, th = roc_curve(y, s); return float(th[np.argmax(t >= 0.8)])


RULES = {"Юден": thr_youden, "чувствительность 80 %": thr_recall80}


def nmt_thresholds(results_dir):
    """Пороги из отложенных 15 % NMT train: берутся из сохранённых оценок по людям (eegdino_v2_per_person.csv: dino2_ens_prob, reve4_ens_prob,
    ens_reve4_dino2 — основной вид «60 окон, верхние 20 %»), пересчитываются, а не переписываются руками."""
    d = pd.read_csv(os.path.join(results_dir, "eegdino_v2_per_person.csv"))
    v = d[d.dataset == "nmt_val"]
    cols = {"reve4": "reve4_ens_prob", "dino2": "dino2_ens_prob", "ens": "ens_reve4_dino2"}
    return {m: {r: f(v.y.values, v[c].values) for r, f in RULES.items()} for m, c in cols.items()}


def calib_eval(y, S, k, n_rep=500, seed=0):
    """Порог по k случайным людям (правила как выше), метрики на остальных; одни и те же выборки для всех моделей. S — словарь модель -> оценки."""
    rng, res = np.random.default_rng(seed), []
    for _ in range(n_rep):
        idx = rng.permutation(len(y)); cal, test = idx[:k], idx[k:]
        if len(set(y[cal])) < 2:
            continue
        for m, s in S.items():
            for rn, f in RULES.items():
                res.append(dict(модель=m, порог=rn, **metr(y[test], (s[test] >= f(y[cal], s[cal])).astype(int))))
    R = pd.DataFrame(res); g = R.groupby(["модель", "порог"])
    return (g.mean().round(2).astype(str) + " (" + g.quantile(.025).round(2).astype(str) + "-" + g.quantile(.975).round(2).astype(str) + ")").reset_index()


def cmd_eval(a):
    rec = pd.read_csv(a.scores)
    if a.meta:                                         # год, приступы, монтаж — из описи подготовки, если их нет в оценках
        m = pd.read_csv(a.meta)
        add = [c for c in ("year", "n_seiz", "montage") if c in m and (c not in rec or rec[c].isna().all())]
        rec = rec.drop(columns=[c for c in add if c in rec]).merge(m[["id"] + add], on="id", how="left")
    score_cols = [c for c in ["reve4", "dino2", "ens"] if c in rec.columns]
    names = {"reve4": "REVE v4", "dino2": "EEG-DINO v2", "ens": "REVE v4 + EEG-DINO v2"}
    os.makedirs(a.out, exist_ok=True)
    S = session_table(rec, score_cols)
    S.to_csv(os.path.join(a.out, "tuep_session_scores.csv"), index=False)
    if S.seiz_unknown.any():
        print("ВНИМАНИЕ: у", int(S.seiz_unknown.sum()), "сеансов нет разметки приступов — они считаются без приступов")
    # устойчивость к выбору сеанса
    sens = []
    for rule in SESSION_RULES:
        Pr, _ = select_sessions(S, score_cols, rule, a.seed)
        for c in score_cols:
            sens.append(dict(сеанс=rule, модель=names[c], n=len(Pr), n_эпилепсия=int(Pr.y.sum()), auc=round(roc_auc_score(Pr.y, Pr[c]), 3)))
    pd.DataFrame(sens).to_csv(os.path.join(a.out, "tuep_session_rule_sensitivity.csv"), index=False)
    print(pd.DataFrame(sens).pivot(index="модель", columns="сеанс", values="auc").to_string())
    P, dropped = select_sessions(S, score_cols, a.sessions, a.seed)
    print(f"сеанс на пациента: {a.sessions}; пациентов {len(P)} ({int(P.y.sum())} с эпилепсией); выпали (все сеансы с приступами): {len(dropped)} {dropped}")
    if a.folds:
        P = P.merge(pd.read_csv(a.folds)[["patient", "fold"]], on="patient", how="left")
    P.to_csv(os.path.join(a.out, "tuep_patient_scores.csv"), index=False)       # предсказания по людям — пересчитывать метрики без повторного прогона
    # утечка через год и монтаж
    Cf, ct = confound_check(P)
    Cf.to_csv(os.path.join(a.out, "tuep_confounds.csv"), index=False); ct.to_csv(os.path.join(a.out, "tuep_montage_by_class.csv"))
    print(Cf.to_string(index=False)); print(ct.to_string())
    thr = nmt_thresholds(a.results)
    y = P.y.to_numpy(); rows = []
    for c in score_cols:
        rows.append(dict(подгруппа="все пациенты, пары внутри монтажа", модель=names[c], порог="—", n=len(P), n_эпилепсия=int(y.sum()),
                         **{"ROC AUC": f"{stratified_auc(y, P[c].to_numpy(), P.montage.to_numpy()):.3f}"}))
    sub = {"все пациенты": np.ones(len(P), bool)}
    for mt, g in P.groupby("montage"):
        if g.y.sum() >= 5 and (1 - g.y).sum() >= 5:
            sub[f"монтаж {mt}"] = (P.montage == mt).to_numpy()
    if a.subgroup:
        sg = pd.read_csv(a.subgroup).set_index("patient").ied_group
        P["ied_group"] = P.patient.map(sg)
        # «без разрядов» сравнивается со всеми контролями: ROC AUC = больные без разрядов против здоровых
        sub["эпилепсия без разрядов + контроль"] = ((P.y == 0) | (P.ied_group == "без разрядов")).to_numpy()
        sub["эпилепсия с разрядами + контроль"] = ((P.y == 0) | (P.ied_group == "с разрядами")).to_numpy()
    for sname, mask in sub.items():
        for c in score_cols:
            s, yy = P[c].to_numpy()[mask], y[mask]
            auc, lo, hi = boot_auc(yy, s)
            rows.append(dict(подгруппа=sname, модель=names[c], порог="—", **{"значение порога": np.nan}, n=len(yy), n_эпилепсия=int(yy.sum()),
                             **{"ROC AUC": f"{auc:.3f} ({lo:.2f}-{hi:.2f})"}))
            for rn in RULES:
                t = thr[c][rn]; mm = metr_ci(yy, s, t)
                rows.append(dict(подгруппа=sname, модель=names[c], порог=rn, **{"значение порога": round(t, 3)}, n=len(yy), n_эпилепсия=int(yy.sum()),
                                 **{"ROC AUC": f"{auc:.3f} ({lo:.2f}-{hi:.2f})", "accuracy": mm["acc"], "сбаланс. accuracy": mm["bacc"], "precision": mm["prec"],
                                    "recall (чувств.)": mm["rec"], "специфичность": mm["spec"], "примечание": "порог: отложенные 15 % NMT train, перенесён без изменений"}))
    T = pd.DataFrame(rows); T.to_csv(os.path.join(a.out, "tuep_metrics.csv"), index=False)
    print(T.drop(columns=[c for c in ["примечание"] if c in T]).to_string(index=False))
    for k in a.calib_k:
        C = calib_eval(y, {names[c]: P[c].to_numpy() for c in score_cols}, k)
        C.insert(0, "калибровка людей", k); C.to_csv(os.path.join(a.out, f"tuep_calibration_k{k}.csv"), index=False); print(C.to_string(index=False))
    # порог без разметки: медиана оценок половины пациентов (метки не нужны), метрики на другой половине, 500 повторов
    rng, un = np.random.default_rng(a.seed), []
    for _ in range(500):
        idx = rng.permutation(len(y)); h, t = idx[: len(y) // 2], idx[len(y) // 2:]
        for c in score_cols:
            s = P[c].to_numpy(); un.append(dict(модель=names[c], **metr(y[t], (s[t] >= np.median(s[h])).astype(int))))
    U = pd.DataFrame(un).groupby("модель")
    U = (U.mean().round(2).astype(str) + " (" + U.quantile(.025).round(2).astype(str) + "-" + U.quantile(.975).round(2).astype(str) + ")").reset_index()
    U.insert(1, "порог", "без разметки: медиана половины"); U.to_csv(os.path.join(a.out, "tuep_unlabeled_threshold.csv"), index=False); print(U.to_string(index=False))
    # по фолдам — разброс (модели не обучались на TUEP, фолды нужны для будущего дообучения и для разброса оценки)
    if "fold" in P:
        fr = [dict(fold=f, n=len(g), auc=roc_auc_score(g.y, g[c])) for f, g in P.groupby("fold") for c in score_cols[:1]]
        print("ROC AUC по фолдам (", names[score_cols[0]], "):", [round(r["auc"], 3) for r in fr])


# ---------------------------------------------------------------- подгруппа «без разрядов»
NEG = re.compile(r"\b(no|without|absence of|absent|not (?:seen|identified|observed))\b[^.\n]{0,60}\b(epileptiform|spike|sharp|discharge|seizure)", re.I)
POS = re.compile(r"\b(epileptiform|spikes?|sharp waves?|polyspikes?|spike[- ]and[- ]wave|spike[- ]wave|pleds?|gpeds?|lpds|gpds|discharges?|seizures?)\b", re.I)


def classify_report(text):
    """Грубое правило по тексту заключения: «с разрядами» / «без разрядов» / «неясно». Только для первичной разметки — проверить глазами выборку."""
    imp = re.split(r"(?i)impression\s*:?", text, maxsplit=1)
    t = imp[1] if len(imp) == 2 else text
    t_pos = NEG.sub(" ", t)
    if POS.search(t_pos):
        return "с разрядами"
    if NEG.search(t) or re.search(r"(?i)\bnormal\b|within normal limits", t):
        return "без разрядов"
    return "неясно"


def cmd_subgroup(a):
    """Что действительно есть в TUEP, выяснится по описи (n_txt в inventory_sessions.csv). Два источника:
    1) тексты заключений рядом с сеансами (*.txt) — если они входят в TUEP v3.1.0;
    2) внешний csv с числом разрядов по пациентам (--ied-csv: patient, n_ied) — результат детектора (в проекте есть IED-детектор, см. ied-senualab-turkish).
    Разрядов в самом TUEP нет (разметка только приступов, по описанию корпуса)."""
    meta = pd.read_csv(a.meta); P = patient_table(meta); rows = []
    if a.ied_csv:
        d = pd.read_csv(a.ied_csv).groupby("patient").n_ied.sum()
        P["n_ied"] = P.patient.map(d)
        P["ied_group"] = np.where(P.n_ied.isna(), "неясно", np.where(P.n_ied > 0, "с разрядами", "без разрядов")); P["source"] = "детектор"
    else:
        sess = meta[["patient", "session"]].drop_duplicates()
        for r in sess.itertuples():
            txts = glob.glob(os.path.join(a.root, "**", r.patient, r.session, "**", "*.txt"), recursive=True)
            for t in txts:
                rows.append(dict(patient=r.patient, session=r.session, file=os.path.relpath(t, a.root), cls=classify_report(open(t, encoding="utf-8", errors="replace").read())))
        R = pd.DataFrame(rows, columns=["patient", "session", "file", "cls"])
        R.to_csv(os.path.splitext(a.out)[0] + "_reports.csv", index=False)
        print("отчётов найдено:", len(R), "| классы:", R.cls.value_counts().to_dict())
        def agg(g):
            c = set(g)
            return "с разрядами" if "с разрядами" in c else ("без разрядов" if c == {"без разрядов"} else "неясно")
        P["ied_group"] = P.patient.map(R.groupby("patient").cls.agg(agg)).fillna("нет отчёта"); P["source"] = "отчёты (правило по тексту)"
    P.to_csv(a.out, index=False)
    print(pd.crosstab(P.y, P.ied_group))
    print("Правило по тексту — грубое: перед использованием проверить вручную 20–30 отчётов из *_reports.csv.")


def main():
    ap = argparse.ArgumentParser(); sp = ap.add_subparsers(dest="cmd", required=True)
    f = sp.add_parser("folds"); f.add_argument("--meta", required=True); f.add_argument("--out", required=True); f.add_argument("--n-folds", type=int, default=5); f.add_argument("--seed", type=int, default=0)
    l = sp.add_parser("leakage"); l.add_argument("--out", required=True)
    s = sp.add_parser("subgroup"); s.add_argument("--meta", required=True); s.add_argument("--root", default=""); s.add_argument("--out", required=True); s.add_argument("--ied-csv", default="")
    e = sp.add_parser("eval"); e.add_argument("--scores", required=True); e.add_argument("--folds", default=""); e.add_argument("--results", required=True)
    e.add_argument("--out", required=True); e.add_argument("--subgroup", default="")
    e.add_argument("--sessions", default="random", choices=SESSION_RULES, help="один сеанс на пациента (основной расчёт — random)")
    e.add_argument("--seed", type=int, default=0); e.add_argument("--meta", default="", help="h1_shared_meta.csv: год, приступы, монтаж")
    e.add_argument("--calib-k", type=int, nargs="*", default=[10, 20])
    a = ap.parse_args()
    {"folds": cmd_folds, "leakage": cmd_leakage, "subgroup": cmd_subgroup, "eval": cmd_eval}[a.cmd](a)


if __name__ == "__main__":
    main()
