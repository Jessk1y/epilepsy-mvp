"""DeepEpilepsy авторов (Lemoine et al., Brain Communications 2025; веса HF emilelemoine/DeepEpilepsy_v1, MIT) на TUEP без дообучения.

    CUDA_VISIBLE_DEVICES=0 python tuep_deepepilepsy.py --prep ~/data/tuep_work/prep --out ~/data/tuep_work/out_de \
        --sessions-from ~/data/tuep_work/out/tuep_patient_scores.csv [--limit 20]
    python tuep_protocol.py eval --scores ~/data/tuep_work/out_de/tuep_record_scores.csv --meta ~/data/tuep_work/prep/h1_shared_meta.csv \
        --results ../results --out ~/data/tuep_work/out_de

Модель обучена ровно на нашей задаче: «эпилепсия или нет» по рутинной ЭЭГ, контроль — другие пациенты клиники (CHUM, Монреаль).
Вход как у авторов (статья, раздел «Automated processing»; блокноты h1_2026-10-19/20 *deepepilepsy*): 19 каналов в среднем референте,
200 Гц, окна 30 с (19 × 6000), порядок каналов авторов; каждое окно нормируется по каналу (как в наших блокнотах); оценка записи —
медиана вероятностей окон (как у авторов). Окна — с перекрытием 50 %, не больше --max-win на файл, равномерно.
Код модели — из репозитория авторов (gitlab chum-epilepsy/dl_epilepsy_reeg), клонируется в --repo при первом запуске.
Выход — tuep_record_scores.csv со столбцом de (формат tuep_infer.py), дальше tuep_protocol.py eval. Повторный запуск продолжает.

Проверка референта и фильтра (15.11). У авторов запись сохранена относительно среднего A1 и A2 («average referential montage (A1–A2)»,
в коде EEGReference = "A1A2") и подаётся в сеть без программных фильтров; у нас — средний по 19 каналам и 0,1–75 Гц + режекция 60 Гц.
--ref ears читает EDF заново (--root, --inventory) и вычитает (A1 + A2) / 2: монтаж 01_tcp_ar — по каналам A1 и A2 файла, 02_tcp_le —
как есть (уже связанные уши), 03_tcp_ar_a — пропуск (ушных электродов нет). --filt none — без фильтров, только 200 Гц.
--compare <tuep_record_scores.csv другого варианта> печатает ROC AUC обоих вариантов на общих пациентах и разность с ДИ:

    CUDA_VISIBLE_DEVICES=0 python tuep_deepepilepsy.py --prep ~/data/tuep_work/prep --out ~/data/tuep_work/out_de_ears --ref ears \
        --root ~/data/TUEP --sessions-from ~/data/tuep_work/out/tuep_patient_scores.csv --compare ~/data/tuep_work/out_de/tuep_record_scores.csv
"""
import argparse, importlib.util, os, subprocess, sys, time
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import tuep_common as C                                    # noqa: F401  (подключает ../src)
from shared_data import load_meta, read_record, EL

AUTH = ["Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4", "F7", "F8", "T3", "T4", "T5", "T6", "O1", "O2", "Fz", "Cz", "Pz"]
IDX = [EL.index(e) for e in AUTH]                         # из порядка общего набора в порядок авторов
SF, L = 200, 200 * 30


def load_model(repo, dev, torch):
    if not os.path.isdir(repo):
        subprocess.run(["git", "clone", "--depth", "1", "https://gitlab.com/chum-epilepsy/dl_epilepsy_reeg.git", repo], check=True)
    spec = importlib.util.spec_from_file_location("de_transformers", os.path.join(repo, "src/epileptology/ml/models/transformers.py"))
    de = importlib.util.module_from_spec(spec); spec.loader.exec_module(de)
    from huggingface_hub import hf_hub_download
    w = hf_hub_download("emilelemoine/DeepEpilepsy_v1", "DeepEpilepsy_large_50_30s_augment.pt")
    m = de.ViT1d(n_class=2, in_channels=19, n_timepoints=L, patch_size=50, tokenizer="conv1d",
                 n_layers=6, n_heads=8, dim_hidden=512, dim_mlp=512, dropout=0.25, attention_dropout=0.25)
    m.load_state_dict(torch.load(w, map_location="cpu", weights_only=True), strict=True)
    return m.to(dev).eval()


def windows30(X, overlap, max_win):
    n = X.shape[1]
    if n < L:
        return np.zeros((0, 19, L), np.float32)
    step = max(1, int(L * (1 - overlap)))
    starts = np.arange(0, n - L + 1, step)
    if len(starts) > max_win:
        starts = starts[np.unique(np.linspace(0, len(starts) - 1, max_win).round().astype(int))]
    return np.stack([X[:, s:s + L] for s in starts]).astype(np.float32)


def load_edf(path, ref, filt):
    """Запись прямо из EDF: (19, n) float32 в порядке EL, 200 Гц, или (None, причина). ref: avg — средний по 19 (как в prep),
    ears — относительно (A1 + A2) / 2; filt: prep — 0,1–75 Гц + режекция 60 Гц (как в prep), none — без фильтров."""
    try:
        _, _, labels = C.read_header(path)
        idx = C.channel_index(labels)
        if idx is None:
            return None, "нет 19 каналов"
        ears = []
        if ref == "ears" and C.ref_kind(labels) != "LE":          # -LE: запись уже относительно связанных ушей
            pos = {}
            for i, l in enumerate(labels):
                pos.setdefault(C.norm_channel(l), i)
            if "A1" not in pos or "A2" not in pos:
                return None, "нет A1/A2"
            ears = [pos["A1"], pos["A2"]]
        X, fs = C.read_edf_channels(path, idx + ears)
    except Exception as e:
        return None, f"чтение: {type(e).__name__}"
    X = C.to_fs200(X, fs)
    if ref == "avg":
        X = X - X.mean(0, keepdims=True)
    elif ears:
        X = X[:19] - X[19:].mean(0, keepdims=True)
    return (C.filt(X) if filt == "prep" else X).astype(np.float32), ""


def patient_auc(d, col="de"):
    """Файлы -> пациент (среднее с весом длительности) -> таблица patient, y, оценка."""
    g = d.assign(ws=d[col] * d.dur_s).groupby("patient").agg(y=("y", "first"), ws=("ws", "sum"), w=("dur_s", "sum"))
    return (g.ws / g.w).rename("s").to_frame().assign(y=g.y)


def zscore(x):
    return (x - x.mean(-1, keepdims=True)) / (x.std(-1, keepdims=True) + 1e-20)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prep", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--repo", default="~/data/weights/dl_epilepsy_reeg")
    ap.add_argument("--sessions-from", default="", help="tuep_patient_scores.csv: только выбранные сеансы (как в основном прогоне)")
    ap.add_argument("--overlap", type=float, default=0.5); ap.add_argument("--max-win", type=int, default=120)
    ap.add_argument("--no-zscore", action="store_true", help="подавать мкВ без нормировки окна (проверка чувствительности)")
    ap.add_argument("--ref", default="prep", choices=["prep", "avg", "ears"], help="prep — подготовленные записи; avg / ears — заново из EDF (нужен --root)")
    ap.add_argument("--filt", default="prep", choices=["prep", "none"], help="для --ref avg / ears: фильтры как в prep или без них")
    ap.add_argument("--root", default="", help="папка TUEP с EDF"); ap.add_argument("--inventory", default="", help="inventory_files.csv (по умолчанию <prep>/../inventory/)")
    ap.add_argument("--compare", default="", help="tuep_record_scores.csv другого варианта: ROC AUC на общих пациентах и разность")
    ap.add_argument("--bs", type=int, default=32); ap.add_argument("--device", default="cuda:0"); ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    prep, out, repo = (os.path.expanduser(x) for x in (a.prep, a.out, a.repo))
    import torch
    os.makedirs(out, exist_ok=True)
    meta, _ = load_meta(prep)
    if a.sessions_from:
        keep = pd.read_csv(os.path.expanduser(a.sessions_from))[["patient", "session"]]
        meta = meta.merge(keep, on=["patient", "session"])
    if a.limit:
        meta = meta.head(a.limit)
    edf = {}
    if a.ref != "prep":
        assert a.root, "для --ref avg / ears нужен --root (папка TUEP с EDF)"
        inv = pd.read_csv(os.path.expanduser(a.inventory) if a.inventory else os.path.join(os.path.dirname(prep.rstrip("/")), "inventory", "inventory_files.csv"))
        edf = {os.path.splitext(f)[0]: os.path.join(os.path.expanduser(a.root), q) for f, q in zip(inv.file, inv.path)}
    dev = a.device if torch.cuda.is_available() else "cpu"
    print("записей:", len(meta), "| пациентов:", meta.patient.nunique(), "|", torch.cuda.get_device_name(0) if dev != "cpu" else "процессор", flush=True)
    model = load_model(repo, dev, torch)
    path = os.path.join(out, "tuep_record_scores.csv")
    rows = pd.read_csv(path).to_dict("records") if os.path.exists(path) else []
    seen = {r["id"] for r in rows}
    t0, skipped = time.time(), {}
    for j, r in enumerate(meta.itertuples()):
        if r.id in seen:
            continue
        if a.ref == "prep":
            X = read_record(prep, r._asdict()).astype(np.float32)
        else:
            X, why = load_edf(edf[r.id], a.ref, a.filt)
            if X is None:
                skipped[why] = skipped.get(why, 0) + 1; continue
        W = windows30(X[IDX], a.overlap, a.max_win)
        if len(W) == 0:
            skipped["короче 30 с"] = skipped.get("короче 30 с", 0) + 1; continue      # у авторов таких окон не бывает
        if not a.no_zscore:
            W = zscore(W)
        p = []
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=dev != "cpu"):
            for b in range(0, len(W), a.bs):
                p.append(torch.softmax(model(torch.from_numpy(W[b:b + a.bs]).to(dev)).float(), 1)[:, 1].cpu().numpy())
        p = np.concatenate(p)
        rows.append(dict(id=r.id, patient=r.patient, session=r.session, y=r.y, dur_s=r.dur_s, montage=getattr(r, "montage", ""),
                         year=getattr(r, "year", None), n_seiz=getattr(r, "n_seiz", None), n_windows=len(W), de=float(np.median(p))))
        if (j + 1) % 50 == 0 or j + 1 == len(meta):
            pd.DataFrame(rows).to_csv(path, index=False); print(f"{j+1}/{len(meta)} {time.time()-t0:.0f} с", flush=True)
    pd.DataFrame(rows).to_csv(path, index=False)
    d = pd.DataFrame(rows)
    print("файлов с оценкой:", len(d), "| пропущено в этом запуске:", skipped or 0, f"| референт: {a.ref}, фильтр: {a.filt if a.ref != 'prep' else 'prep'}")
    print("медиана оценки: без эпилепсии", round(d[d.y == 0].de.median(), 3), "| эпилепсия", round(d[d.y == 1].de.median(), 3))
    print("→", path)
    if a.sessions_from and d.y.nunique() == 2:
        import tuep_protocol as TP
        P = patient_auc(d)
        print("пациентов с оценкой: %d (эпилепсия %d) | ROC AUC %.3f (%.2f–%.2f)" % (len(P), int(P.y.sum()), *TP.boot_auc(P.y.to_numpy(), P.s.to_numpy())))
        if a.compare:
            from sklearn.metrics import roc_auc_score
            o = pd.read_csv(os.path.expanduser(a.compare))
            ids = set(d.id) & set(o.id)                         # сравнение — на одних и тех же файлах
            A, B = patient_auc(d[d.id.isin(ids)]), patient_auc(o[o.id.isin(ids)])
            B = B.loc[A.index]; y, s1, s2 = A.y.to_numpy(), A.s.to_numpy(), B.s.to_numpy()
            rg, v = np.random.default_rng(0), []
            for _ in range(2000):
                i = rg.integers(0, len(y), len(y))
                if len(set(y[i])) == 2:
                    v.append(roc_auc_score(y[i], s1[i]) - roc_auc_score(y[i], s2[i]))
            print("на общих %d пациентах: этот вариант %.3f, вариант --compare %.3f, разность %+.3f (%+.3f…%+.3f)"
                  % (len(y), roc_auc_score(y, s1), roc_auc_score(y, s2), roc_auc_score(y, s1) - roc_auc_score(y, s2), *np.percentile(v, [2.5, 97.5])))


if __name__ == "__main__":
    main()
