"""Шаг 3: подготовка записей TUEP в формат общего набора Г1 (читается теми же функциями shared_data.py).

    py tuep_prepare.py --root /data/tuep/v3.1.0 --out /data/tuep_work/prep [--montages 01_tcp_ar,02_tcp_le,03_tcp_ar_a]

Выход (как у zakharsann/h1-prep-shared):
  h1_shared_meta.csv  — строка на файл EDF: dataset=tuep, part=tuep, split=tuep, id=<пациент>_<сеанс>_<файл>, y (1 — эпилепсия),
                        patient, session, montage, in_tusz (если дана опись), sfreq_orig, dur_full_s, dur_s, std_uv, std_raw_uv, clipped,
                        shard200/off200/n200
  h1_shared_info.json — параметры подготовки
  h1_shared_failed.csv — файлы, которые не удалось подготовить (нет 19 каналов, битый заголовок, короче 10 с)
  fs200/tuep_NNN.npy  — плоские float16 по ~1 ГБ: запись (19, n) подряд

Чтение: shared_data.load_meta(root), shared_data.read_record(root, row) / read_all (на локальном диске read_record через mmap быстро).
Всего ~627 ч -> ~17 ГБ float16 при 200 Гц. Не в git и не на Kaggle.
"""
import argparse, glob, json, os, sys, time
from concurrent.futures import ThreadPoolExecutor
import numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tuep_common as C

SHARD_BYTES = 1.0e9


class Shards:
    """Тот же формат, что в блокноте общего набора: плоские float16-массивы по ~1 ГБ."""
    def __init__(self, out_dir, part="tuep"):
        self.dir, self.part, self.k, self.buf, self.n_el = os.path.join(out_dir, "fs200"), part, 0, [], 0
        os.makedirs(self.dir, exist_ok=True)

    def add(self, X16):
        if self.buf and (self.n_el + X16.size) * 2 > SHARD_BYTES:
            self.flush()
        off = self.n_el
        self.buf.append(X16.ravel()); self.n_el += X16.size
        return f"{self.part}_{self.k:03d}.npy", off, X16.shape[1]

    def flush(self):
        if self.buf:
            np.save(os.path.join(self.dir, f"{self.part}_{self.k:03d}.npy"), np.concatenate(self.buf))
            self.k, self.buf, self.n_el = self.k + 1, [], 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--montages", default="", help="через запятую; пусто — все (01_tcp_ar,02_tcp_le,03_tcp_ar_a)")
    ap.add_argument("--inventory", default="", help="inventory_files.csv из tuep_inventory.py: берём группу и признак TUSZ оттуда")
    ap.add_argument("--groups-csv", default="", help="csv patient,y, если группы нет ни в пути, ни в описи")
    ap.add_argument("--also-notch50", action="store_true", help="добавить режекцию 50 Гц (как при обучении моделей) к 60 Гц")
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0, help="только первые N файлов (проверка)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    paths = sorted(glob.glob(os.path.join(a.root, "**", "*.edf"), recursive=True))
    rows = []
    for p in paths:
        r = C.parse_path(p); r["path"] = p
        rows.append(r)
    D = pd.DataFrame(rows)
    if a.montages:
        D = D[D.montage.isin(a.montages.split(","))]
    if a.inventory:
        inv = pd.read_csv(a.inventory)[["path", "group", "in_tusz"]]
        D["rel"] = [os.path.relpath(p, a.root) for p in D.path]
        D = D.drop(columns="group").merge(inv.rename(columns={"path": "rel"}), on="rel", how="left").drop(columns="rel")
    if a.groups_csv:
        g = pd.read_csv(a.groups_csv).set_index("patient").y
        D["group"] = D.group.fillna(D.patient.map(g))
    assert D.patient.notna().all(), "часть файлов не разобралась по шаблону aaaaaaaa_sNNN_tNNN.edf"
    assert D.group.notna().all(), "у части файлов нет группы: дайте --inventory или --groups-csv"
    D = D.sort_values(["patient", "session", "file"]).reset_index(drop=True)
    if a.limit:
        D = D.head(a.limit)
    print("файлов к подготовке:", len(D), "| пациентов:", D.patient.nunique())

    extra = (50.0,) if a.also_notch50 else ()
    W = Shards(a.out)
    out_rows, failed = [], []
    t0 = time.time()

    def job(i):
        r = D.iloc[i]
        X, info = C.load_record(r.path, extra_notch=extra)
        return i, X, info

    with ThreadPoolExecutor(a.threads) as ex:
        for j, (i, X, info) in enumerate(ex.map(job, range(len(D)))):
            r = D.iloc[i]
            if X is None:
                failed.append(dict(id=r.file, patient=r.patient, session=r.session, error=info["error"]))
            else:
                sh, off, n = W.add(X)
                row = dict(dataset="tuep", part="tuep", split="tuep", id=os.path.splitext(r.file)[0], y=int(r.group),
                           label="epilepsy" if r.group == 1 else "no_epilepsy", patient=r.patient, session=r.session, montage=r.montage,
                           scale_k=1.0, short_lt60s=bool(info["dur_s"] < 60), **info, shard200=sh, off200=off, n200=n)
                if "in_tusz" in D.columns:
                    row["in_tusz"] = r.in_tusz
                out_rows.append(row)
            if (j + 1) % 100 == 0 or j + 1 == len(D):
                print(f"{j+1}/{len(D)} {time.time()-t0:.0f} с, не прочитано {len(failed)}", flush=True)
    W.flush()

    meta = pd.DataFrame(out_rows)
    first = ["dataset", "part", "split", "id", "y", "label", "sfreq_orig", "dur_full_s", "dur_s", "scale_k", "std_uv", "std_raw_uv", "clipped"]
    meta = meta[first + [c for c in meta.columns if c not in first]]
    meta.to_csv(os.path.join(a.out, "h1_shared_meta.csv"), index=False)
    pd.DataFrame(failed, columns=["id", "patient", "session", "error"]).to_csv(os.path.join(a.out, "h1_shared_failed.csv"), index=False)
    total = sum(os.path.getsize(f) for f in glob.glob(os.path.join(a.out, "fs200", "*.npy")))
    info = dict(version=1, source="TUEP v3.1.0 (TUH, NEDC)", channels=C.EL, fs=[200],
                reference="средний референт по 19 каналам 10–20 (для любого монтажа TUH)",
                filter=dict(band_hz=list(C.BAND), butter_order=4, sosfiltfilt=True, notch_hz=C.NOTCH_HZ, notch_q=C.NOTCH_Q, extra_notch_hz=list(extra)),
                dtype="float16", unit="мкВ", f16_clip=C.F16_MAX, scale_k=1.0, size_gb=round(total / 1e9, 2), n_records=len(meta),
                n_patients=int(meta.patient.nunique()), n_failed=len(failed), montages=a.montages or "все", seconds=round(time.time() - t0))
    json.dump(info, open(os.path.join(a.out, "h1_shared_info.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(info)
    print(pd.crosstab(meta.label, meta.montage))


if __name__ == "__main__":
    main()
