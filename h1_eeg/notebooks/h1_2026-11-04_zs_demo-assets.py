"""Г1, демонстрация REVE v4: небольшие файлы для набора Kaggle zakharsann/h1-demo-assets.

Только готовые результаты (csv и картинки из h1_eeg/results), без сигналов и весов.
Запуск из h1_eeg: py notebooks/h1_2026-11-04_zs_demo-assets.py <папка вывода>
Затем: py -m kaggle datasets create -p <папка вывода>   (или datasets version -m "...")
"""
import ast, json, os, re, shutil, sys
import numpy as np, pandas as pd

OUT = sys.argv[1]
os.makedirs(OUT, exist_ok=True)
R = "results"
VIEW = ("все окна", "верхние 20 %")
THR_NMT = 0.832

# 1. Оценки записей REVE v4 (все окна, среднее верхних 20 %, среднее вероятностей 3 запусков): NMT eval, Турция, EpiEEG-60
pp = pd.read_csv(f"{R}/reve_v4_per_person.csv")
pp = pp[(pp["окна"] == VIEW[0]) & (pp["сборка"] == VIEW[1]) & pp.dataset.isin(["nmt_eval", "turkey"])]
fac = pd.read_csv(f"{R}/reve_v4_errors_factors.csv")[["dataset", "id", "amp_uv", "muscle", "blink", "art_frac"]]
a = pp.merge(fac, on=["dataset", "id"], how="left")
a = pd.DataFrame(dict(dataset=a.dataset, id=a.id, person=a.id, y=a.y, score=a.reve4_ens_prob, seed0=a.reve4_seed0, seed1=a.reve4_seed1,
                      seed2=a.reve4_seed2, dur_min=a.dur_s / 60, amp_uv=a.amp_uv, muscle=a.muscle, n_ied_marks=np.nan))
ep = pd.read_csv(f"{R}/epieeg60_per_person.csv")
ep = ep[(ep.iloc[:, 1] == VIEW[0]) & (ep.iloc[:, 2] == VIEW[1])]
ids = pd.read_csv(f"{R}/epieeg60_window_scores.csv.gz", usecols=["id", "person"]).drop_duplicates()
ep = ep.merge(ids, on="person")
b = pd.DataFrame(dict(dataset="epieeg60", id=ep.id, person=ep.person, y=ep.y, score=ep.reve4_ens_prob, seed0=ep.reve4_seed0, seed1=ep.reve4_seed1,
                      seed2=ep.reve4_seed2, dur_min=ep.dur_min, amp_uv=ep.std_uv, muscle=np.nan, n_ied_marks=ep.n_ied_marks))
scores = pd.concat([a, b], ignore_index=True)
assert scores.groupby("dataset").size().to_dict() == {"epieeg60": 63, "nmt_eval": 185, "turkey": 121}, scores.groupby("dataset").size()
scores.round(5).to_csv(f"{OUT}/reve_v4_scores.csv", index=False)

# 2. Оценки окон (для кривых вдоль записи без запуска модели)
w = pd.read_csv(f"{R}/reve_v4_window_scores.csv.gz")
w = w[w.dataset.isin(["nmt_eval", "turkey"])][["dataset", "id", "window", "seed", "p"]]
we = pd.read_csv(f"{R}/epieeg60_window_scores.csv.gz").assign(dataset="epieeg60")[["dataset", "id", "window", "seed", "p"]]
pd.concat([w, we]).round(4).to_csv(f"{OUT}/reve_v4_window_scores.csv.gz", index=False, compression="gzip")


# 3. Порог «по 20 людям этой больницы»: из 500 случайных выборок 10 больных + 10 здоровых берём ту,
#    у которой порог Юдена ближе всего к медиане порогов (типичная выборка). Тот же код — в блокноте.
def youden(y, s):
    from sklearn.metrics import roc_curve
    f, t, th = roc_curve(y, s)
    return float(th[np.argmax(t - f)])


def calib(d, k=10, n=500):
    draws = []
    for s in range(n):
        g = np.random.default_rng(s)
        cal = list(g.choice(d[d.y == 1].id, k, replace=False)) + list(g.choice(d[d.y == 0].id, k, replace=False))
        draws.append((youden(d[d.id.isin(cal)].y, d[d.id.isin(cal)].score), cal))
    ths = np.array([t for t, _ in draws])
    j = int(np.argmin(np.abs(ths - np.median(ths))))
    return draws[j][0], draws[j][1]


LOCAL = {}
for ds in ["turkey", "epieeg60"]:
    thr, cal = calib(scores[scores.dataset == ds])
    LOCAL[ds] = (thr, cal)
    print(ds, "порог по 20 людям:", round(thr, 3), sorted(cal))

# 4. Записи для показа: выбраны по готовым оценкам, с запасом от порога, не из 20 людей калибровки
CASES = [
    ("TP1", "turkey", "1_epilepsy (18).mat", "верно найден больной (выше обоих порогов)"),
    ("TP2", "turkey", "1_epilepsy (16).mat", "верно найден больной"),
    ("TN1", "turkey", "2_Normal (51).mat", "верно распознан здоровый"),
    ("TN2", "turkey", "2_Normal (46).mat", "верно распознан здоровый"),
    ("FP1", "turkey", "2_Normal (5).mat", "ложная тревога: высокая амплитуда, медленные волны в лобных отведениях"),
    ("FP2", "turkey", "2_Normal (54).mat", "ложная тревога: короткий подозрительный отрезок около 10-й минуты"),
    ("FN1", "turkey", "1_epilepsy (7).mat", "пропуск: низкая амплитуда, оценка низкая по всей записи"),
    ("FN2", "turkey", "1_epilepsy (12).mat", "пропуск"),
    ("EP1", "epieeg60", "Abnormal_Patient_40.EEG_Patient_40", "EpiEEG-60: больной с размеченными разрядами, найден"),
    ("EP2", "epieeg60", "Abnormal_Patient_39.EEG_Patient_39", "EpiEEG-60: больной без размеченных разрядов, пропуск"),
]
rows = []
for key, ds, rid, note in CASES:
    r = scores[(scores.dataset == ds) & (scores.id == rid)].iloc[0]
    thr, cal = LOCAL[ds]
    assert rid not in cal, (rid, "в выборке калибровки")
    p_loc, p_nmt = int(r.score >= thr), int(r.score >= THR_NMT)
    grp = {(1, 1): "TP", (0, 0): "TN", (0, 1): "FP", (1, 0): "FN"}[(int(r.y), p_loc)]
    if ds == "turkey":
        assert grp == key[:2], (key, grp)
        assert abs(r.score - thr) > 0.08, (key, r.score, thr)
    rows.append(dict(key=key, dataset=ds, record=rid, y=int(r.y), score_saved=round(float(r.score), 3),
                     verdict_nmt=p_nmt, verdict_local=p_loc, group_local=grp, dur_min=round(float(r.dur_min), 1),
                     amp_uv=round(float(r.amp_uv), 1), n_ied_marks=r.n_ied_marks, note=note))
cases = pd.DataFrame(rows)
cases.to_csv(f"{OUT}/demo_cases.csv", index=False)
print(cases.drop(columns="note").to_string(index=False))

# 5. Истории обучения: (потери обучения, ROC AUC отложенной части NMT) по эпохам, в длинном виде
SRC = [("REVE v4 (основная)", "reve_v4_runs.csv", None), ("REVE v2", "reve_v2_runs.csv", None), ("REVE (первая)", "reve_finetune_runs.csv", None),
       ("EEG-DINO", "eegdino_finetune_runs.csv", None), ("LaBraM", "labram_finetune_runs.csv", None), ("EEGPT", "eegpt_finetune_runs.csv", None),
       ("LUNA", "luna_finetune_runs.csv", None), ("BIOT", "biot_finetune_runs.csv", None),
       ("CBraMod", "cbramod_vs_deepepilepsy_runs.csv", "cb"), ("DeepEpilepsy v2", "cbramod_vs_deepepilepsy_runs.csv", "de")]
hist = []
for name, f, sel in SRC:
    d = pd.read_csv(f"{R}/{f}")
    if sel:
        d = d[d.model == sel]
    for _, r in d.iterrows():
        h = ast.literal_eval(re.sub(r"np\.float64\(([^)]*)\)", r"\1", r["hist"]))
        auc = [x[1] for x in h]
        for e, (loss, v) in enumerate(h):
            hist.append(dict(model=name, seed=int(r.seed), epoch=e + 1, train_loss=loss, val_auc=v, chosen=int(e == int(np.argmax(auc)))))
hist = pd.DataFrame(hist)
hist.to_csv(f"{OUT}/train_history.csv", index=False)
print(hist[hist.chosen == 1].groupby("model").epoch.apply(list).to_string())
shutil.copy(f"{R}/eegdino_v2_curves.csv", OUT)

# 6. Разбор ошибок и возраст: таблицы и картинки из отчёта 2026-10-27_reve_errors.md
for f in ["reve_v4_errors_factors.csv", "age_check_errors.csv", "age_check_by_agebin.csv"]:
    shutil.copy(f"{R}/{f}", OUT)
for f in ["reve_v4_errors/01_factors_turkey.png", "reve_v4_errors/03_spearman.png", "age_check.png", "epieeg60_reve_v4.png", "eegdino_v2_curves.png"]:
    shutil.copy(f"{R}/figures/{f}", f"{OUT}/{os.path.basename(f)}")

# 7. Дополнения к консультации 05.11: пара с EEG-DINO v2, потери REVE на отложенной части, кривая адаптации
for f in ["eegdino_v2_per_person.csv", "reve_v4_loss_curves.csv"]:
    shutil.copy(f"{R}/{f}", OUT)
shutil.copy(f"{R}/figures/reve_fewshot_curve.png", OUT)

json.dump({"title": "h1-demo-assets", "id": "zakharsann/h1-demo-assets", "licenses": [{"name": "other"}]},
          open(f"{OUT}/dataset-metadata.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("файлы:", sorted(os.listdir(OUT)), f"{sum(os.path.getsize(f'{OUT}/{f}') for f in os.listdir(OUT)) / 1e6:.1f} МБ")
