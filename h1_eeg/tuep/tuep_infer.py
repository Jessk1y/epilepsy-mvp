"""Шаг 5: вывод готовых моделей (REVE v4, EEG-DINO v2) на подготовленном TUEP — без дообучения, на ОДНОЙ карте.

    CUDA_VISIBLE_DEVICES=0 py tuep_infer.py --prep /data/tuep_work/prep --out /data/tuep_work/out \
        --reve-dir /data/weights/h1-reve-v4 --dino-dir /data/weights/h1-eegdino-v2

Как в блокнотах: запись -> ФВЧ 0,3 Гц по записи целиком -> окна 10 с подряд с начала -> 60 окон равномерно -> оценка записи =
среднее верхних 20 % вероятностей окон (основной вид REVE v4 / EEG-DINO v2); 3 запуска (seed 0/1/2) -> среднее вероятностей;
ансамбль REVE v4 + EEG-DINO v2 = среднее двух вероятностей. Выход: tuep_record_scores.csv (по файлам), tuep_window_scores.npz.
Дальше: tuep_protocol.py eval (оценка по пациентам, метрики, пороги).

Веса: `kaggle kernels output zakharsann/h1-reve-v4 -p <reve-dir>` (reve_v4_seed{0,1,2}.pt, reve_v4_model.py),
      `kaggle kernels output zakharsann/h1-eegdino-v2 -p <dino-dir>` (eegdino_v2_seed{0,1,2}.pt, eegdino_v2_model.py).
Нужны также веса основ с HF (brain-bzh/reve-base, brain-bzh/reve-positions; код авторов eegdino/EEG-DINO) — заранее в кэш HF,
если с машины HF недоступен (HF_HUB_OFFLINE=1 и --dino-src с папкой, где лежит models/ авторов).
"""
import argparse, os, sys, time
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import tuep_common as C                                    # noqa: F401  (подключает ../src в sys.path)
from shared_data import load_meta, read_record, windows, highpass

L, HP, CAP_WIN, TOP = 2000, 0.3, 60, 0.2


def spread(n, k):
    return np.unique(np.linspace(0, n - 1, k).astype(int)) if n > k else np.arange(n)


def top20(p):
    return float(np.sort(p)[::-1][:int(np.ceil(TOP * len(p)))].mean())


def load_models(a, torch, dev):
    """Список (имя, функция окна -> вероятности) по запускам."""
    models = {}
    if a.reve_dir:
        sys.path.insert(0, a.reve_dir)
        from reve_v4_model import RVNet, rv_input, positions
        pos = positions(dev)
        for s in a.seeds:
            m = RVNet(pos); m.load_state_dict(torch.load(os.path.join(a.reve_dir, f"reve_v4_seed{s}.pt"), map_location="cpu", weights_only=False)["state_dict"])
            models[f"reve4_seed{s}"] = (m.to(dev).eval(), rv_input)
    if a.dino_dir:
        sys.path.insert(0, a.dino_dir)
        from eegdino_v2_model import DinoNet, dino_input
        for s in a.seeds:
            m = DinoNet(src_dir=a.dino_src or None); m.load_state_dict(torch.load(os.path.join(a.dino_dir, f"eegdino_v2_seed{s}.pt"), map_location="cpu", weights_only=False)["state_dict"])
            models[f"dino2_seed{s}"] = (m.to(dev).eval(), dino_input)
    assert models, "укажите --reve-dir и/или --dino-dir"
    return models


def predict(model, prep, W, torch, dev, bs):
    out = []
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16, enabled=dev.startswith("cuda")):
        for b in range(0, len(W), bs):
            out.append(torch.sigmoid(model(prep(W[b:b + bs]).to(dev)).float()).cpu().numpy())
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prep", required=True, help="папка с h1_shared_meta.csv (выход tuep_prepare.py)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--reve-dir", default=""); ap.add_argument("--dino-dir", default=""); ap.add_argument("--dino-src", default="")
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    ap.add_argument("--device", default="cuda:0", help="одна карта; номер карты задавать через CUDA_VISIBLE_DEVICES")
    ap.add_argument("--bs", type=int, default=30)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    import torch
    os.makedirs(a.out, exist_ok=True)
    meta, info = load_meta(a.prep)
    if a.limit:
        meta = meta.head(a.limit)
    print("записей:", len(meta), "| пациентов:", meta.patient.nunique(), "|", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "процессор")
    models = load_models(a, torch, a.device)

    path = os.path.join(a.out, "tuep_record_scores.csv")
    done = pd.read_csv(path) if os.path.exists(path) else pd.DataFrame()
    rows = done.to_dict("records"); seen = set(done["id"]) if len(done) else set()
    wins = dict(np.load(os.path.join(a.out, "tuep_window_scores.npz"))) if os.path.exists(os.path.join(a.out, "tuep_window_scores.npz")) and seen else {}
    t0 = time.time()
    for j, r in enumerate(meta.itertuples()):
        if r.id in seen:
            continue
        X = highpass(read_record(a.prep, r._asdict()), HP).astype(np.float32)
        W = windows(X, L)
        W = W[spread(len(W), CAP_WIN)]
        row = dict(id=r.id, patient=r.patient, session=r.session, y=r.y, dur_s=r.dur_s, montage=getattr(r, "montage", ""),
                   year=getattr(r, "year", None), n_seiz=getattr(r, "n_seiz", None), n_windows=len(W))
        for name, (m, prep) in models.items():
            p = predict(m, prep, W, torch, a.device, a.bs)
            wins[f"{name}__{r.id}"] = p.astype(np.float32)
            row[name] = top20(p)
        for fam in ("reve4", "dino2"):
            cols = [c for c in row if c.startswith(fam + "_seed")]
            if cols:
                row[fam] = float(np.mean([row[c] for c in cols]))
        if "reve4" in row and "dino2" in row:
            row["ens"] = (row["reve4"] + row["dino2"]) / 2
        rows.append(row)
        if (j + 1) % 50 == 0 or j + 1 == len(meta):
            pd.DataFrame(rows).to_csv(path, index=False); np.savez_compressed(os.path.join(a.out, "tuep_window_scores.npz"), **wins)
            print(f"{j+1}/{len(meta)} {time.time()-t0:.0f} с", flush=True)
    pd.DataFrame(rows).to_csv(path, index=False)
    np.savez_compressed(os.path.join(a.out, "tuep_window_scores.npz"), **wins)
    print("→", path)


if __name__ == "__main__":
    main()
