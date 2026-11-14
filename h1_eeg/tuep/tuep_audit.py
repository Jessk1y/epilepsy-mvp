"""Проверка прогона TUEP на настоящих данных (ревью 14.11.2026): не сбой ли подготовки дал ROC AUC 0,52. Только процессор, 1–3 мин.

    python tuep_audit.py --work ~/data/tuep_work --root ~/data/TUEP [--sample 60]

Печатает только сводные цифры (без кодов пациентов) и пишет их в <work>/out/tuep_audit.csv:
 1. амплитуда подготовленных записей (СКО, мкВ) по классам, монтажам и частотам — рядом с NMT и Турцией из общего набора;
 2. на выборке файлов: средний референт, доля «плоских» окон, сеть 60 Гц и область 50 Гц, доля мощности выше 45 Гц по частотам записи;
 3. перекос технических признаков по классу на уровне файлов (в отчёте он посчитан только по одному сеансу на пациента);
 4. если есть оценки моделей: связь оценки с амплитудой; сборка «верхние 20 % по всем окнам сеанса» вместо «по файлам»;
    сеансы с приступом по заключению врача (metadata_v00r.xlsx), но без разметки seiz — и ROC AUC без таких сеансов.
Пишет <work>/protocol/exclude_report_seizures.csv (patient, session) — для --exclude-sessions в tuep_finetune.py. Этот файл в git не класть.
"""
import argparse, os, sys
import numpy as np, pandas as pd, scipy.signal as ss
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import tuep_protocol as TP
import tuep_finetune as FT
from shared_data import read_record, windows

REF = "общий набор Г1, медиана СКО записи: NMT норма 18,5 / патология 31,6; Турция здоровые 12,7 / эпилепсия 27,2 мкВ"


def auc(y, s):
    y, s = np.asarray(y), np.asarray(s, float); m = ~np.isnan(s)
    return round(roc_auc_score(y[m], s[m]), 3) if len(set(y[m])) == 2 else np.nan


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--work", required=True); ap.add_argument("--root", default=""); ap.add_argument("--sample", type=int, default=60)
    a = ap.parse_args()
    W, R = FT.expand(a.work), FT.expand(a.root)
    prep = os.path.join(W, "prep")
    meta = FT.load_prep(prep); S = FT.session_frame(meta)
    rows = []
    add = lambda k, v, n="": (rows.append(dict(проверка=k, значение=v, n=n)), print(f"{k}: {v}" + (f"  [n={n}]" if n != "" else "")))
    # 1. амплитуда
    print("==", REF)
    q = lambda x: f"{x.median():.1f} ({x.quantile(.25):.1f}–{x.quantile(.75):.1f})"
    for g, d in meta.groupby("y"):
        add(f"СКО записи, мкВ, медиана (квартиль), {'эпилепсия' if g else 'без эпилепсии'}", q(d.std_uv), len(d))
    for g, d in meta.groupby("montage"):
        add(f"СКО записи, монтаж {g}", q(d.std_uv), len(d))
    for g, d in meta.groupby(meta.sfreq_orig.round().astype(int)):
        add(f"СКО записи, частота {g} Гц", q(d.std_uv), len(d))
    add("файлов с СКО < 2 мкВ / > 200 мкВ", f"{int((meta.std_uv < 2).sum())} / {int((meta.std_uv > 200).sum())}", len(meta))
    add("обрезано отсчётов (±60 000)", int(meta.clipped.sum()))
    # 2. выборка файлов
    pick = meta.groupby(["y", "montage"], group_keys=False).apply(lambda d: d.sample(min(len(d), max(4, a.sample // 6)), random_state=0))
    st = []
    for r in pick.itertuples():
        X = read_record(prep, r._asdict()).astype(np.float64)
        Wn = windows(X, FT.L); Wn = Wn[FT.spread(len(Wn), 60)]
        f, P = ss.welch(Wn, 200, nperseg=400, axis=-1); P = P.mean((0, 1))
        b = lambda lo, hi: P[(f >= lo) & (f < hi)].mean()
        st.append(dict(y=r.y, montage=r.montage, fs=int(round(r.sfreq_orig)), ref=float(np.abs(X.mean(0)).max()), flat=float((Wn.std((1, 2)) < 0.5).mean()),
                       r60=b(59, 61.5) / b(54, 57), r50=b(49, 51.5) / b(44, 47), hi=P[(f >= 45) & (f < 75)].sum() / P[(f >= 1) & (f < 75)].sum(),
                       alpha=b(8, 13) / b(1, 45), wstd=float(np.median(Wn.std((1, 2))))))
    T = pd.DataFrame(st)
    add("выборка: макс. |среднее по 19 каналам|, мкВ (должно быть ~0)", f"{T.ref.max():.3f}", len(T))
    add("выборка: доля плоских окон (СКО < 0,5 мкВ), средняя / макс. по файлам", f"{T.flat.mean():.3f} / {T.flat.max():.2f}")
    add("выборка: мощность 60 Гц к 55 Гц (режекция есть — меньше 1), медиана", f"{T.r60.median():.2f}")
    add("выборка: мощность 50 Гц к 45 Гц (при обучении на NMT здесь была режекция), медиана", f"{T.r50.median():.2f}")
    for g, d in T.groupby("fs"):
        add(f"выборка: доля мощности 45–75 Гц, частота {g} Гц, медиана", f"{d.hi.median():.4f}", len(d))
    for g, d in T.groupby("y"):
        add(f"выборка: доля мощности 45–75 Гц, {'эпилепсия' if g else 'без эпилепсии'}", f"{d.hi.median():.4f}", len(d))
    # 3. технические признаки по классу — файлы и сеансы
    for name, col, D in (("файлы: частота", meta.sfreq_orig.round().astype(int), meta), ("файлы: монтаж", meta.montage, meta), ("сеансы: слой монтаж|частота", S.stratum, S)):
        ct = pd.crosstab(col, D.y); print(f"-- {name} (столбцы: без эпилепсии, эпилепсия)\n{ct.to_string()}")
        add(f"ROC AUC класса по признаку «{name}» (доля класса в слое, на той же выборке)", auc(D.y, D.groupby(col.to_numpy()).y.transform("mean")), len(D))
    # заключения врачей: приступ в сеансе без разметки seiz
    ex = None
    if R and os.path.exists(os.path.join(R, "DOCS", "metadata_v00r.xlsx")):
        md = pd.read_excel(os.path.join(R, "DOCS", "metadata_v00r.xlsx"))
        key = md["Subject ID"].astype(str).str.strip()
        rep = (md["Electrographic Seizure"].astype(str).str.upper().eq("Y") | md["Clinical Seizure"].astype(str).str.upper().eq("Y"))
        rep = dict(zip(key, rep))
        S["rep_seiz"] = [bool(rep.get(f"{p}_{s[:4]}", False)) for p, s in zip(S.patient, S.session)]
        add("сеансов с приступом по заключению, но без разметки seiz (остались в расчёте)", int((S.rep_seiz & ~S.has_seiz).sum()), int((~S.has_seiz).sum()))
        ex = S[S.rep_seiz & ~S.has_seiz][["patient", "session"]]
        os.makedirs(os.path.join(W, "protocol"), exist_ok=True)
        ex.to_csv(os.path.join(W, "protocol", "exclude_report_seizures.csv"), index=False)
    # 4. оценки моделей
    sp = os.path.join(W, "out", "tuep_record_scores.csv")
    if os.path.exists(sp):
        rec = pd.read_csv(sp)
        rec = rec.drop(columns=[c for c in ("n_seiz", "year", "montage", "sfreq_orig", "std_uv") if c in rec]).merge(meta[["id", "n_seiz", "year", "montage", "sfreq_orig", "std_uv"]], on="id")
        cols = [c for c in ("reve4", "dino2", "ens") if c in rec]
        for g, d in rec.groupby("y"):
            add(f"Спирмен оценка пары — СКО записи, {'эпилепсия' if g else 'без эпилепсии'} (на Турции у REVE v4 было 0,59)", round(spearmanr(d.ens, d.std_uv).correlation, 2), len(d))
        Ss = TP.session_table(rec, cols)
        P, _ = TP.select_sessions(Ss, cols, "random", 0)
        add("ROC AUC пары по протоколу (сверка с отчётом: 0,523)", auc(P.y, P.ens), len(P))
        if ex is not None:
            Ss2 = Ss.copy(); k = set(zip(ex.patient, ex.session)); Ss2["has_seiz"] = Ss2.has_seiz | [(p, s) in k for p, s in zip(Ss2.patient, Ss2.session)]
            P2, dr = TP.select_sessions(Ss2, cols, "random", 0)
            add("ROC AUC пары без сеансов с приступом по заключению", auc(P2.y, P2.ens), f"{len(P2)}, выпало пациентов {len(dr)}")
            e = Ss.merge(S[["patient", "session", "rep_seiz"]], on=["patient", "session"]); e = e[(e.y == 1) & ~e.has_seiz]
            add("положительный контроль: у больных сеансы с приступом по заключению против остальных (без разметки seiz), пара", auc(e.rep_seiz.astype(int), e.ens), len(e))
        wp = os.path.join(W, "out", "tuep_window_scores.npz")
        if os.path.exists(wp):                                  # сборка по всем окнам сеанса (как у NMT: одна запись = один набор окон)
            Z = np.load(wp); ch = dict(zip(P.patient, P.session)); out = []
            for (p, s), g in rec.groupby(["patient", "session"]):
                if ch.get(p) != s:
                    continue
                v = []
                for fam in ("reve4", "dino2"):
                    ks = [[k for k in (f"{fam}_seed{i}__{rid}" for i in range(3)) if k in Z.files] for rid in g.id]
                    if all(ks):
                        v.append(np.mean([FT.top20(np.concatenate([Z[k[i]] for k in ks])) for i in range(min(map(len, ks)))]))
                if len(v) == 2:
                    out.append((g.y.iloc[0], np.mean(v)))
            if out:
                o = np.array(out); add("ROC AUC пары: верхние 20 % по всем окнам сеанса (вместо среднего по файлам)", auc(o[:, 0], o[:, 1]), len(o))
    pd.DataFrame(rows).to_csv(os.path.join(W, "out", "tuep_audit.csv") if os.path.isdir(os.path.join(W, "out")) else os.path.join(W, "tuep_audit.csv"), index=False)


if __name__ == "__main__":
    main()
