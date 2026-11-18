"""Г1, TUEP: новые варианты предобработки и признаков с ЧЕСТНЫМ (вложенным) выбором лучшего. Схема — те же фолды и правила,
что в tuep_baseline.py / tuep_finetune.py: тест — фолд k, отложенная часть — фолд k+1, обучение — остальные; веса сеансов и
выравнивание слоёв «монтаж|частота»; один случайный сеанс без приступов на пациента (seed 0); порог — по отложенной части.

    # 1) признаки (процессор; кэш на диск, повторный запуск не пересчитывает)
    python tuep_variants.py features --prep ~/data/tuep_work/prep --out ~/data/tuep_work/ft --procs 12 --limit 40   # проба: время
    python tuep_variants.py features --prep ~/data/tuep_work/prep --out ~/data/tuep_work/ft --procs 12
    # 2) признаки REVE v4 (видеокарта; веса reve_v4_seed{0,1,2}.pt — готовая REVE v4 после NMT, без TUEP)
    CUDA_VISIBLE_DEVICES=0 python tuep_variants.py embed --prep ~/data/tuep_work/prep --out ~/data/tuep_work/ft --reve-dir ~/data/weights/h1-reve-v4
    # 3) модели по фолдам, вложенный выбор, сводка (процессор, минуты)
    python tuep_variants.py run --prep ~/data/tuep_work/prep --out ~/data/tuep_work/ft

Всё пишется в <out>/variants/: feats_<настройка>.npz, reve_emb.npz, runs/<вариант>_f<k>_s0.csv (формат runs tuep_finetune.py),
tuep_variants_summary.md и *.csv без кодов пациентов (их можно в git), tuep_variants_patient_scores.csv — с кодами (остаётся на машине).
Прежние запуски REVE v4 и эталона (reve, bpampw, tech) берутся из <out>/runs (--ref-runs).

Настройки признаков (<монтаж>_<выброс>_<окна>):
  монтаж: avg — средний референт (как раньше, 19 каналов); bip — продольный биполярный «двойной банан», 18 отведений
          (разности каналов подготовки; средний референт в разности сокращается, поэтому пересчёт точный);
  выброс: r0 — все окна; r1 — выброс плохих окон ДО расчёта признаков. Пороги заданы заранее (18.11, до запуска на TUEP):
          размах любого канала > 500 мкВ; мышечная помеха — у ≥ 3 каналов доля мощности 30–70 Гц (без 57–63 Гц) в 1–70 Гц > 0,30;
          плоский участок — СКО любого канала < 0,5 мкВ. Решение о выбросе — всегда по окну в среднем референте. Если осталось
          меньше max(3, 25 %) окон файла — берутся все окна (такие файлы считаются и печатаются);
  окна:   w60 — 60 равномерных окон 10 с (как раньше, ~10 минут записи); all — все окна 10 с записи подряд.
Сигнал окна — ФВЧ 0,3 Гц по куску «окно ± 5 с» (FT.eval_windows, без ФНЧ), как в эталоне.

Группы признаков (по окну; по окнам файла — медиана и 90-й процентиль, кроме paf):
  rel — lg доли мощности δ 1–4, θ 4–8, α 8–13, β 13–30, γ 30–45 Гц от 1–45 Гц по каналам (как bp);  amp — lg СКО канала;
  wpli_mean — средний wPLI по парам в 5 диапазонах;  wpli_node_theta — сила узла в θ (как bpampw);
  ratio — lg θ/α и lg δ/α по каналам;  aper — наклон и смещение спектра 1/f: прямая в log-log на 2–40 Гц без 7–14 Гц;
  paf — частота пика альфа по каналу (на уровне файла: максимум остатка над 1/f в 7–14 Гц среднего по окнам спектра);
  hjorth — подвижность и сложность Хьорта;  shape — lg (длина линии / СКО), эксцесс, асимметрия распределения (грубый след острых волн);
  asym — |разность| полушарий по 8 парам каналов (lg доли 5 диапазонов и lg СКО);  wpli_node_all — сила узла во всех 5 диапазонах.
Наборы: B = rel + amp + wpli_mean + wpli_node_theta (= прежний эталон bpampw); B+NEW = rel + amp + wpli_mean + все новые группы.
"""
import argparse, json, os, sys, time
from concurrent.futures import ProcessPoolExecutor
from types import SimpleNamespace
import numpy as np, pandas as pd, scipy.signal as ss, scipy.stats as st
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import tuep_finetune as FT                                 # noqa: E402
import tuep_protocol as TP                                 # noqa: E402
from shared_data import EL                                 # noqa: E402

FS = FT.FS
BANDS = [(1, 4), (4, 8), (8, 13), (13, 30), (30, 45)]
CS = [0.003, 0.03, 0.3]                                    # как в tuep_baseline.py
HGB_GRID = [dict(learning_rate=lr, max_depth=d) for lr in (0.03, 0.1) for d in (2, 3)]
REJ_PTP, REJ_MUSCLE, REJ_MUSCLE_NCH, REJ_FLAT = 500.0, 0.30, 3, 0.5
ALL_WIN = 10 ** 9
CONFIGS = ["avg_r0_w60", "bip_r0_w60", "avg_r1_w60", "avg_r0_all", "avg_r1_all", "bip_r1_all"]
_I = {e: i for i, e in enumerate(EL)}
BIP = [("Fp1", "F7"), ("F7", "T3"), ("T3", "T5"), ("T5", "O1"), ("Fp2", "F8"), ("F8", "T4"), ("T4", "T6"), ("T6", "O2"),
       ("Fp1", "F3"), ("F3", "C3"), ("C3", "P3"), ("P3", "O1"), ("Fp2", "F4"), ("F4", "C4"), ("C4", "P4"), ("P4", "O2"),
       ("Fz", "Cz"), ("Cz", "Pz")]
BIP_A, BIP_B = [_I[a] for a, _ in BIP], [_I[b] for _, b in BIP]
PAIRS = {"avg": [(_I[a], _I[b]) for a, b in (("Fp1", "Fp2"), ("F3", "F4"), ("C3", "C4"), ("P3", "P4"), ("O1", "O2"), ("F7", "F8"), ("T3", "T4"), ("T5", "T6"))],
         "bip": [(i, i + 4) for i in (0, 1, 2, 3, 8, 9, 10, 11)]}       # левая цепочка — правая цепочка
B = ["rel", "amp", "wpli_mean", "wpli_node_theta"]
NEW = ["ratio", "aper", "paf", "hjorth", "shape", "asym", "wpli_node_all"]
BNEW = ["rel", "amp", "wpli_mean"] + NEW


# ---------------------------------------------------------------- признаки
def parse_cfg(cfg):
    mont, rej, win = cfg.split("_")
    assert mont in ("avg", "bip") and rej in ("r0", "r1") and win in ("w60", "all"), cfg
    return mont, rej == "r1", 60 if win == "w60" else ALL_WIN


def window_ok(W):
    """(k, 19, L) мкВ, средний референт -> (k,) bool: окно годное. Пороги — константы REJ_* (заданы заранее)."""
    big = (W.max(-1) - W.min(-1) > REJ_PTP).any(1)
    f, P = ss.welch(W, FS, nperseg=400, axis=-1)
    notch = (f >= 57) & (f <= 63)
    hi, tot = (f >= 30) & (f < 70) & ~notch, (f >= 1) & (f < 70) & ~notch
    muscle = ((P[..., hi].sum(-1) / (P[..., tot].sum(-1) + 1e-12)) > REJ_MUSCLE).sum(1) >= REJ_MUSCLE_NCH
    flat = (W.std(-1) < REJ_FLAT).any(1)
    return ~(big | muscle | flat)


def _band_sos():
    return [ss.butter(4, b, btype="bandpass", fs=FS, output="sos") for b in BANDS]


def window_features(W, mont, band_sos):
    """Признаки окон: словарь группа -> (k, d); плюс средний lg-спектр (C, F) для paf."""
    k, C, n = W.shape
    f, P = ss.welch(W, FS, nperseg=400, axis=-1)
    tot = P[..., (f >= 1) & (f < 45)].sum(-1) + 1e-12
    bp = np.stack([P[..., (f >= lo) & (f < hi)].sum(-1) for lo, hi in BANDS], -1)            # (k, C, 5)
    rel = np.log10(bp / tot[..., None] + 1e-9)
    sd = W.std(-1)
    amp = np.log10(sd + 1e-3)
    out = {"rel": rel.reshape(k, -1), "amp": amp}
    out["ratio"] = np.concatenate([np.log10((bp[..., 1] + 1e-12) / (bp[..., 2] + 1e-12)), np.log10((bp[..., 0] + 1e-12) / (bp[..., 2] + 1e-12))], 1)
    fm = (f >= 2) & (f <= 40) & ~((f >= 7) & (f <= 14))
    x = np.log10(f[fm]); yv = np.log10(P[..., fm] + 1e-12)
    xc = x - x.mean()
    slope = (yv * xc).sum(-1) / (xc ** 2).sum()
    out["aper"] = np.concatenate([slope, yv.mean(-1) - slope * x.mean()], 1)
    d1 = np.diff(W, axis=-1); d2 = np.diff(d1, axis=-1)
    v0, v1, v2 = W.var(-1) + 1e-12, d1.var(-1) + 1e-12, d2.var(-1) + 1e-12
    mob = np.sqrt(v1 / v0)
    out["hjorth"] = np.concatenate([mob, np.sqrt(v2 / v1) / mob], 1)
    out["shape"] = np.concatenate([np.log10(np.abs(d1).mean(-1) / (sd + 1e-6) + 1e-9), st.kurtosis(W, axis=-1), st.skew(W, axis=-1)], 1)
    pr = PAIRS[mont]
    L_, R_ = [a for a, _ in pr], [b for _, b in pr]
    out["asym"] = np.concatenate([np.abs(rel[:, L_] - rel[:, R_]).reshape(k, -1), np.abs(amp[:, L_] - amp[:, R_])], 1)
    iu = np.triu_indices(C, 1)
    wm, wn = np.zeros((k, 5)), np.zeros((k, 5, C))
    for s0 in range(0, k, 16):                                                               # кусками: память при «всей записи»
        Wc = W[s0:s0 + 16]
        for b, sb in enumerate(band_sos):
            z = ss.hilbert(ss.sosfiltfilt(sb, Wc, axis=-1), axis=-1)
            im = (z[:, iu[0]] * np.conj(z[:, iu[1]])).imag
            wp = np.abs(im.mean(-1)) / (np.abs(im).mean(-1) + 1e-12)                          # (kc, пары)
            wm[s0:s0 + 16, b] = wp.mean(-1)
            M = np.zeros((len(Wc), C, C)); M[:, iu[0], iu[1]] = wp; M = M + M.transpose(0, 2, 1)
            wn[s0:s0 + 16, b] = M.sum(-1) / (C - 1)
    out["wpli_mean"], out["wpli_node_theta"], out["wpli_node_all"] = wm, wn[:, 1], wn.reshape(k, -1)
    return out, f, np.log10(P + 1e-12).mean(0)


def paf_from_spectrum(f, lp):
    """Средний lg-спектр (C, F) -> частота пика альфа по каналам: максимум остатка над прямой 1/f в 7–14 Гц."""
    fm = (f >= 2) & (f <= 40) & ~((f >= 7) & (f <= 14))
    x = np.log10(f[fm]); xc = x - x.mean()
    slope = (lp[:, fm] * xc).sum(-1) / (xc ** 2).sum(); off = lp[:, fm].mean(-1) - slope * x.mean()
    am = (f >= 7) & (f <= 14)
    res = lp[:, am] - (off[:, None] + slope[:, None] * np.log10(f[am]))
    return f[am][np.argmax(res, 1)]


def file_features(prep, r, cfg, sos=None, band_sos=None):
    """Признаки файла: (словарь группа -> вектор, сведения). Файл короче окна -> (None, сведения)."""
    mont, rej, n_win = parse_cfg(cfg)
    sos = FT.make_sos(0) if sos is None else sos
    band_sos = _band_sos() if band_sos is None else band_sos
    W = FT.eval_windows(prep, r, sos, n_win).astype(np.float64)
    info = dict(n_win=len(W), n_kept=len(W), fallback=0)
    if not len(W):
        return None, info
    if rej:
        ok = window_ok(W)
        if ok.sum() >= max(3, int(np.ceil(0.25 * len(W)))):
            W = W[ok]; info["n_kept"] = int(ok.sum())
        else:
            info["fallback"] = 1
    if mont == "bip":
        W = W[:, BIP_A] - W[:, BIP_B]
    Xw, f, lp = window_features(W, mont, band_sos)
    feats = {g: np.concatenate([np.median(v, 0), np.percentile(v, 90, 0)]) for g, v in Xw.items()}
    feats["paf"] = paf_from_spectrum(f, lp)
    return feats, info


def _job(args):
    prep, rid, shard, off, n, cfg = args
    return file_features(prep, SimpleNamespace(id=rid, shard200=shard, off200=off, n200=n), cfg)


def vdir(out):
    return os.path.join(out, "variants")


def cmd_features(a):
    meta = FT.load_prep(a.prep).reset_index(drop=True)
    os.makedirs(vdir(a.out), exist_ok=True)
    for cfg in a.configs:
        path = os.path.join(vdir(a.out), f"feats_{cfg}.npz")
        if os.path.exists(path) and not a.limit and list(np.load(path, allow_pickle=True)["ids"]) == list(meta.id):
            print(f"{cfg}: уже есть {path}"); continue
        rows = [(a.prep, r.id, r.shard200, int(r.off200), int(r.n200), cfg) for r in meta.itertuples()]
        if a.limit:
            rows = rows[:a.limit]
        t0, res = time.time(), []
        with ProcessPoolExecutor(a.procs) as ex:
            for i, x in enumerate(ex.map(_job, rows, chunksize=2)):
                res.append(x)
                if (i + 1) % 200 == 0:
                    el = time.time() - t0
                    print(f"{cfg}: {i + 1}/{len(rows)} файлов, {el / 60:.1f} мин, осталось ~{el / (i + 1) * (len(rows) - i - 1) / 60:.0f} мин", flush=True)
        el = time.time() - t0
        if a.limit:
            print(f"{cfg}: {len(rows)} файлов за {el:.0f} с; оценка на все {len(meta)}: ~{el / len(rows) * len(meta) / 60:.0f} мин ({a.procs} процессов)")
            continue
        ref = next(f for f, _ in res if f is not None)
        out = {g: np.stack([f[g] if f is not None else np.full(len(v), np.nan) for f, _ in res]) for g, v in ref.items()}
        inf = pd.DataFrame([i for _, i in res])
        np.savez_compressed(path, ids=meta.id.to_numpy(), n_win=inf.n_win.to_numpy(), n_kept=inf.n_kept.to_numpy(), fallback=inf.fallback.to_numpy(),
                            **{"g_" + g: v for g, v in out.items()})
        msg = f"{cfg}: {len(meta)} файлов за {el / 60:.1f} мин, без окон {int((inf.n_win == 0).sum())}"
        if parse_cfg(cfg)[1]:
            m = inf.n_win > 0
            msg += f"; выброшено окон {1 - inf.n_kept[m].sum() / inf.n_win[m].sum():.1%}, файлов «осталось мало — взяты все окна» {int(inf.fallback.sum())}"
        print(msg, "→", path, flush=True)


def load_cfg(out, cfg, ids):
    path = os.path.join(vdir(out), f"feats_{cfg}.npz")
    if not os.path.exists(path):
        return None
    z = np.load(path, allow_pickle=True)
    assert list(z["ids"]) == list(ids), f"{path}: другой список файлов — удалите и посчитайте заново"
    d = {k[2:]: z[k] for k in z.files if k.startswith("g_")}
    d["_info"] = pd.DataFrame({"n_win": z["n_win"], "n_kept": z["n_kept"], "fallback": z["fallback"]})
    return d


# ---------------------------------------------------------------- признаки REVE v4 (видеокарта)
def cmd_embed(a):
    import torch
    meta = FT.load_prep(a.prep).reset_index(drop=True)
    path = os.path.join(vdir(a.out), "reve_emb.npz" if a.model == "reve" else f"{a.model}_emb.npz")
    if os.path.exists(path) and not a.limit:
        print("уже есть", path); return
    os.makedirs(vdir(a.out), exist_ok=True)
    if not torch.cuda.is_available() and a.device.startswith("cuda"):
        print("CUDA недоступна — считаю на процессоре (годится только для игрушечной модели)"); a.device = "cpu"
    dev = a.device
    ns = SimpleNamespace(reve_dir=FT.expand(a.reve_dir), init="nmt" if a.model == "reve" else "base", freeze_backbone=False, lr_scale=1.0)
    models = []
    for seed in a.seeds:
        m, inp, *_ = FT.build_model(a.model, ns, torch, dev, seed)
        models.append((m.eval(), inp))

    def emb(m, x):
        if a.model == "toy":
            h = m.backbone(x).abs().mean(-1)
            return h, m.head(h).squeeze(-1)
        o = m.backbone(x, m.pos.expand(x.size(0), -1, -1))
        if isinstance(o, (tuple, list)):
            o = o[0]
        if hasattr(o, "last_hidden_state"):
            o = o.last_hidden_state
        h = m.head[:3](o.float().mean(2).flatten(1))                     # предпоследний слой головы (256), отсев в eval не действует
        return h, m.head[3:](h).squeeze(-1)

    sos, rows = FT.make_sos(0), list(meta.itertuples())
    if a.limit:
        rows = rows[:a.limit]
    E, Sc, t0 = [], [], time.time()
    amp = dict(device_type="cuda" if dev.startswith("cuda") else "cpu", dtype=torch.float16, enabled=dev.startswith("cuda"))
    for i, r in enumerate(rows):
        W = FT.eval_windows(a.prep, r, sos, FT.CAP_WIN)
        e, s = [], []
        for m, inp in models:
            if not len(W):
                e.append(None); s.append(np.nan); continue
            hs, ps = [], []
            with torch.no_grad(), torch.autocast(**amp):
                for b in range(0, len(W), a.bs):
                    h, lg = emb(m, inp(W[b:b + a.bs]).to(dev))
                    hs.append(h.float().cpu().numpy()); ps.append(torch.sigmoid(lg.float()).cpu().numpy())
            e.append(np.concatenate(hs).mean(0)); s.append(FT.top20(np.concatenate(ps)))
        E.append(e); Sc.append(s)
        if (i + 1) % 200 == 0:
            el = time.time() - t0
            print(f"{i + 1}/{len(rows)} файлов, {el / 60:.1f} мин, осталось ~{el / (i + 1) * (len(rows) - i - 1) / 60:.0f} мин", flush=True)
    el = time.time() - t0
    if a.limit:
        print(f"{len(rows)} файлов за {el:.0f} с; оценка на все {len(meta)}: ~{el / len(rows) * len(meta) / 60:.0f} мин"); return
    d = next(len(x) for e in E for x in e if x is not None)
    X = np.stack([np.concatenate([x if x is not None else np.full(d, np.nan) for x in e]) for e in E])
    np.savez_compressed(path, ids=meta.id.to_numpy(), X=X, score=np.array(Sc, float), seeds=np.array(a.seeds))
    print(f"признаки {a.model}: {X.shape} ({len(a.seeds)} запуска × {d}) за {el / 60:.1f} мин → {path}")


# ---------------------------------------------------------------- варианты
def variants():
    """Список вариантов. pool=True — участвует во вложенном выборе (задан заранее); False — только поисковая таблица."""
    V = [dict(name="b0", title="эталон заново (= bpampw): средний референт, 60 окон", cfg="avg_r0_w60", groups=B, model="lr", pool=True),
         dict(name="v1_bip", title="биполярный монтаж (18 отведений)", cfg="bip_r0_w60", groups=B, model="lr", pool=True),
         dict(name="v2_rej", title="выброс плохих окон", cfg="avg_r1_w60", groups=B, model="lr", pool=True),
         dict(name="v3_all", title="вся запись", cfg="avg_r0_all", groups=B, model="lr", pool=True),
         dict(name="v4_new", title="+ все новые признаки", cfg="avg_r0_w60", groups=BNEW, model="lr", pool=True),
         dict(name="v5_hgb", title="градиентный бустинг, эталонные признаки", cfg="avg_r0_w60", groups=B, model="hgb", pool=True),
         dict(name="v5_hgb_new", title="градиентный бустинг, + новые признаки", cfg="avg_r0_w60", groups=BNEW, model="hgb", pool=True),
         dict(name="v7_comb", title="выброс + вся запись + новые признаки", cfg="avg_r1_all", groups=BNEW, model="lr", pool=True),
         dict(name="v7_comb_bip", title="биполярный + выброс + вся запись + новые признаки", cfg="bip_r1_all", groups=BNEW, model="lr", pool=True),
         dict(name="v6_emb", title="признаки REVE v4 + эталонные в одной лог. регрессии", cfg="avg_r0_w60", groups=B, model="lr", emb=True, pool=True),
         dict(name="v6_emb_new", title="признаки REVE v4 + эталонные + новые в одной лог. регрессии", cfg="avg_r0_w60", groups=BNEW, model="lr", emb=True, pool=True),
         dict(name="x_emb_only", title="только признаки REVE v4 (без дообучения)", cfg="avg_r0_w60", groups=[], model="lr", emb=True, pool=False)]
    for g in NEW:
        gr = [x for x in B if x != "wpli_node_theta"] + [g] if g == "wpli_node_all" else B + [g]
        V.append(dict(name=f"x_{g}", title=f"эталон + {g}", cfg="avg_r0_w60", groups=gr, model="lr", pool=False))
    return V


def file_weights(meta, S, F, chosen, fold, balance, min_pat):
    """Маски обучения / отложенной части / теста и веса файлов — как в tuep_baseline.fit_predict."""
    fit_p, val_p, test_p = FT.split_patients(F, fold)
    S_fit = S[S.patient.isin(fit_p)].reset_index(drop=True)
    w, _ = FT.session_weights(S_fit, balance, min_pat)
    ws = {(p, s): wi for p, s, wi in zip(S_fit.patient, S_fit.session, w)}
    dur = meta.groupby(["patient", "session"]).dur_s.transform("sum").to_numpy()
    wf = np.array([ws.get(k, 0.0) for k in zip(meta.patient, meta.session)]) * meta.dur_s.to_numpy() / dur
    tr = wf > 0
    va = (meta.patient.isin(val_p) & np.array([chosen.get(p) == s for p, s in zip(meta.patient, meta.session)])).to_numpy()
    te = meta.patient.isin(test_p).to_numpy()
    assert not (tr & te).any() and not (tr & va).any() and not (va & te).any()
    return tr, va, te, wf


def fit_predict(X, kind, meta, S, F, chosen, fold, balance, min_pat):
    """Одна модель на фолд; гиперпараметры (C или сетка бустинга) — по ROC AUC отложенной части (по пациентам)."""
    tr, va, te, wf = file_weights(meta, S, F, chosen, fold, balance, min_pat)
    y, sw = meta.y.to_numpy(), wf[tr] / wf[tr].mean()
    if kind == "lr":
        sc = StandardScaler().fit(X[tr]); Xs = np.nan_to_num(sc.transform(X))
        grid = [dict(C=c) for c in CS]
        make = lambda h: LogisticRegression(C=h["C"], max_iter=3000)
    else:
        Xs = X
        grid = HGB_GRID
        make = lambda h: HistGradientBoostingClassifier(max_iter=200, min_samples_leaf=20, l2_regularization=1.0, random_state=0, **h)
    best = None
    for h in grid:
        m = make(h).fit(Xs[tr], y[tr], sample_weight=sw)
        pv = m.predict_proba(Xs[va])[:, 1]
        V = FT.patient_scores(meta[va].assign(score=pv), chosen)
        auc = roc_auc_score(V.y, V.score)
        if best is None or auc > best[0]:
            best = (auc, h, pv, m.predict_proba(Xs[te])[:, 1])
    auc, h, pv, pt = best
    cols = ["id", "patient", "session", "y", "dur_s"]
    d = pd.concat([meta.loc[va, cols].assign(split="val", score=pv), meta.loc[te, cols].assign(split="test", score=pt)]).assign(fold=fold, seed=0)
    return d, h, auc


def build_X(v, feats, emb):
    parts = [feats[v["cfg"]][g] for g in v["groups"]]
    if v.get("emb"):
        parts.append(emb)
    return np.concatenate(parts, 1)


# ---------------------------------------------------------------- оценка (как cmd_report в tuep_finetune.py)
def evaluate(d, chosen, F, info, title):
    te, va = d[d.split == "test"], d[d.split == "val"]
    P = FT.patient_scores(te, chosen).merge(F[["patient", "fold"]], on="patient")
    P["session"] = P.patient.map(chosen)
    P = P.join(info, on=["patient", "session"])
    y, s = P.y.to_numpy(), P.score.to_numpy()
    assert P.patient.is_unique
    auc, lo, hi = TP.boot_auc(y, s)
    per_fold = [roc_auc_score(g.y, g.score) for _, g in P.groupby("fold") if g.y.nunique() == 2]
    vauc = [roc_auc_score(V.y, V.score) for k in sorted(va.fold.unique()) for V in [FT.patient_scores(va[va.fold == k], chosen)] if V.y.nunique() == 2]
    strat = TP.stratified_auc(y, s, P.stratum.to_numpy())
    mm = (P.montage == "01_tcp_ar").to_numpy()
    row = {"вариант": title, "n": len(P), "ROC AUC (95 % ДИ)": f"{auc:.3f} ({lo:.2f}-{hi:.2f})", "по фолдам": f"{np.mean(per_fold):.3f} ± {np.std(per_fold):.3f}",
           "внутри слоя": f"{strat:.3f}", "только 01_tcp_ar": f"{roc_auc_score(y[mm], s[mm]):.3f} (n={int(mm.sum())})" if mm.sum() and len(set(y[mm])) == 2 else "—",
           "отложенные, среднее": f"{np.mean(vauc):.3f}" if vauc else "—", "_auc": auc, "_strat": strat}
    thr = []
    for rn, f in TP.RULES.items():
        pred = np.full(len(P), -1)
        for k in sorted(te.fold.unique()):
            V = FT.patient_scores(va[va.fold == k], chosen)
            if V.y.nunique() < 2:
                continue
            pred[(P.fold == k).to_numpy()] = (P.score[P.fold == k].to_numpy() >= f(V.y.to_numpy(), V.score.to_numpy())).astype(int)
        ok = pred >= 0
        if ok.any():
            P[f"pred_{'youden' if rn == 'Юден' else 'sens80'}"] = pred
            mt = FT.boot_metr(y[ok], pred[ok])
            thr.append({"вариант": title, "порог": rn, "n": int(ok.sum()), "accuracy": mt["acc"], "сбаланс. accuracy": mt["bacc"],
                        "precision": mt["prec"], "recall": mt["rec"], "специфичность": mt["spec"]})
    return row, thr, P


def pair(R, a, b):
    key = ["split", "fold", "id", "patient", "session", "y", "dur_s"]
    d = pd.concat([R[a], R[b]]).groupby(key, as_index=False).agg(score=("score", "mean"), k=("score", "size"))
    return d[d.k == 2].drop(columns="k")


def val_auc(d, k, chosen):
    V = FT.patient_scores(d[(d.split == "val") & (d.fold == k)], chosen)
    return roc_auc_score(V.y, V.score) if V.y.nunique() == 2 else -1.0


def cmd_run(a):
    meta = FT.load_prep(a.prep).reset_index(drop=True)
    S = FT.session_frame(meta, a.exclude_sessions); F = FT.get_folds(meta, a.folds); chosen, dropped = FT.chosen_sessions(S)
    folds = sorted(F.fold.unique())
    rdir = os.path.join(vdir(a.out), "runs"); os.makedirs(rdir, exist_ok=True)
    feats, info_rej = {}, {}
    for cfg in CONFIGS:
        z = load_cfg(a.out, cfg, meta.id)
        if z is not None:
            info_rej[cfg] = z.pop("_info"); feats[cfg] = z
    ep = os.path.join(vdir(a.out), "reve_emb.npz" if not a.emb_file else a.emb_file)
    emb = None
    if os.path.exists(ep):
        z = np.load(ep, allow_pickle=True)
        assert list(z["ids"]) == list(meta.id), f"{ep}: другой список файлов"
        emb = z["X"]
    names = {}
    for v in variants():
        if a.variants and v["name"] not in a.variants:
            continue
        if v["cfg"] not in feats or (v.get("emb") and emb is None):
            print(f"{v['name']}: пропущен — нет", "признаков " + v["cfg"] if v["cfg"] not in feats else "признаков REVE (команда embed)"); continue
        names[v["name"]] = v
        X = None
        for k in folds:
            p = os.path.join(rdir, f"{v['name']}_f{k}_s0.csv")
            if os.path.exists(p):
                continue
            X = build_X(v, feats, emb) if X is None else X
            d, h, auc = fit_predict(X, v["model"], meta, S, F, chosen, int(k), a.balance, a.min_pat)
            d.to_csv(p + ".tmp", index=False); os.replace(p + ".tmp", p)
            json.dump(dict(variant=v["name"], fold=int(k), hp=h, val_auc=auc, n_features=int(X.shape[1])), open(p[:-4] + ".json", "w", encoding="utf-8"), ensure_ascii=False)
            print(f"{v['name']}: фолд {k}, {h}, признаков {X.shape[1]}, отложенные ROC AUC {auc:.3f}", flush=True)
    R = {m: d for m, d in FT.load_runs(rdir).items() if m in names}
    ref = FT.load_runs(FT.expand(a.ref_runs)) if a.ref_runs and os.path.isdir(FT.expand(a.ref_runs)) else {}
    title = {m: f"{m}: {v['title']}" + (" [бустинг]" if v["model"] == "hgb" else "") for m, v in names.items()}
    for m in ("reve", "bpampw", "tech"):
        if m in ref:
            R["ref_" + m] = ref[m]; title["ref_" + m] = f"прежний {m}: " + FT.NAMES.get(m, m)
    have_reve = "ref_reve" in R
    if have_reve and "ref_bpampw" in R:
        R["ref_reve+bpampw"] = pair(R, "ref_reve", "ref_bpampw"); title["ref_reve+bpampw"] = "прежняя пара REVE v4 + эталон с wPLI (0,661)"
    pool = [m for m in ("ref_bpampw", "ref_reve+bpampw") if m in R] + [m for m, v in names.items() if v["pool"]]
    if have_reve:
        for m, v in names.items():
            if v["pool"]:
                R[f"reve+{m}"] = pair(R, "ref_reve", m); title[f"reve+{m}"] = f"пара REVE v4 + {m}"; pool.append(f"reve+{m}")
    else:
        print("нет прежних запусков REVE в", a.ref_runs, "— пары с REVE не строятся")
    # вложенный выбор: в каждом фолде — кандидат с наибольшим ROC AUC отложенной части; при равенстве — раньше в списке
    info = S.set_index(["patient", "session"])[["montage", "sfreq", "stratum", "has_seiz", "n_files"]]
    choice, parts = [], []
    for k in folds:
        va = {m: val_auc(R[m], k, chosen) for m in pool if (R[m].fold == k).any()}
        best = max(va, key=lambda m: (va[m], -pool.index(m)))
        choice.append({"фолд": int(k), "выбран": best, "отложенные ROC AUC": round(va[best], 3),
                       "второй": sorted(va, key=lambda m: -va[m])[1] if len(va) > 1 else "", "кандидатов": len(va)})
        parts.append(R[best][R[best].fold == k])
    R["nested"] = pd.concat(parts); title["nested"] = "**вложенный выбор (итог)**"
    rows, thr, P_all = [], [], None
    for m in ["nested"] + [x for x in R if x != "nested"]:
        row, th, P = evaluate(R[m], chosen, F, info, title[m])
        row["код"] = m; row["в выборе"] = "да" if m in pool else ""
        rows.append(row); thr += th
        P_all = P[["patient", "y", "fold", "stratum"]].copy() if P_all is None else P_all
        P_all = P_all.merge(P[["patient", "score"] + [c for c in P if c.startswith("pred_")]].rename(columns=lambda c: c if c == "patient" else f"{m}_{c}"), on="patient", how="outer")
    T, TH, CH = pd.DataFrame(rows), pd.DataFrame(thr), pd.DataFrame(choice)
    od = vdir(a.out)
    P_all.to_csv(os.path.join(od, "tuep_variants_patient_scores.csv"), index=False)            # с кодами пациентов — только на рабочей машине
    T.drop(columns=["_auc", "_strat"]).to_csv(os.path.join(od, "tuep_variants_metrics.csv"), index=False)
    TH.to_csv(os.path.join(od, "tuep_variants_thresholds.csv"), index=False); CH.to_csv(os.path.join(od, "tuep_variants_nested_choice.csv"), index=False)
    # правило вывода (задано заранее)
    tn = T.set_index("код")
    r_auc, r_st = (tn.loc["ref_reve+bpampw", "_auc"], tn.loc["ref_reve+bpampw", "_strat"]) if "ref_reve+bpampw" in tn.index else (0.661, 0.633)
    n_auc, n_st = tn.loc["nested", "_auc"], tn.loc["nested", "_strat"]
    Pn = P_all.dropna(subset=["nested_score"] + (["ref_reve+bpampw_score"] if "ref_reve+bpampw" in tn.index else []))
    diff = ""
    if "ref_reve+bpampw" in tn.index:
        dv, dl, dh = FT.paired_diff(Pn.y.to_numpy(), Pn.nested_score.to_numpy(), Pn["ref_reve+bpampw_score"].to_numpy())
        diff = f"Разница «вложенный выбор − прежняя пара» на тех же {len(Pn)} пациентах: {dv:+.3f} (95 % ДИ {dl:+.3f}…{dh:+.3f})."
    better = n_auc > r_auc and n_st > r_st
    verdict = ("**Улучшение по правилу**: вложенный выбор выше пары и по общему ROC AUC, и внутри слоя." if better else
               "**Признак утечки**: общий ROC AUC выше пары, а внутри слоя — нет. Не принимается." if n_auc > r_auc else
               "**Улучшения нет**: вложенный выбор не выше прежней пары. Лучшим остаётся пара REVE v4 + эталон с wPLI.")
    if abs(n_auc - r_auc) < 0.05:
        verdict += " Разница меньше 0,05 — на 199 пациентах статистически не различима."
    rej_lines = []
    for cfg, inf in info_rej.items():
        if parse_cfg(cfg)[1]:
            share = pd.Series((1 - inf.n_kept / inf.n_win.clip(lower=1)).to_numpy(), index=meta.index)
            Q = FT.patient_scores(meta.assign(score=share)[inf.n_win.to_numpy() > 0], chosen)
            rej_lines.append(f"- `{cfg}`: выброшено окон {1 - inf.n_kept.sum() / max(inf.n_win.sum(), 1):.1%}; «мало окон — взяты все» в {int(inf.fallback.sum())} файлах; "
                             f"доля выброшенного у выбранного сеанса: эпилепсия {Q.score[Q.y == 1].mean():.1%}, без {Q.score[Q.y == 0].mean():.1%}, "
                             f"ROC AUC этой доли как предсказателя {roc_auc_score(Q.y, Q.score):.3f}")
    cols = ["вариант", "n", "ROC AUC (95 % ДИ)", "по фолдам", "внутри слоя", "только 01_tcp_ar", "отложенные, среднее"]
    main_codes = ["nested"] + [c for c in ("ref_reve+bpampw", "ref_bpampw", "ref_tech") if c in tn.index]
    md = ["# TUEP: новые варианты предобработки и признаков — сводка", "",
          f"Пациентов: {len(P_all)} (выпали — все сеансы с приступами: {len(dropped)}). Тест — фолд k, отложенная часть — фолд k+1, обучение — остальные. "
          "Один случайный сеанс без приступов (seed 0). Порог — по отложенной части фолда. ДИ — бутстреп по пациентам (2000).", "",
          "## Итог: вложенный выбор (лучший вариант в каждом фолде — по отложенной части)", "",
          T[T["код"].isin(main_codes)].set_index("код").loc[main_codes][cols].to_markdown(index=False), "",
          "Выбор по фолдам:", "", CH.to_markdown(index=False), "", verdict + (" " + diff if diff else ""), "",
          "Правило (задано заранее): улучшение — только если вложенный выбор выше пары 0,661 и внутри слоя выше 0,633; разница < ~0,05 не различима; "
          "рост общего AUC без роста внутри слоя — признак утечки. Оговорка: отложенная часть у REVE уже использована для выбора эпохи, "
          "а у лог. регрессии и бустинга — для выбора гиперпараметров, поэтому её ROC AUC завышен у всех кандидатов (у REVE — сильнее).", "",
          "## Пороговые метрики (итог)", "", TH[TH["вариант"].isin([title[c] for c in main_codes])].to_markdown(index=False), "",
          "## Все варианты на тесте — поисково, для понимания (тест «изношен», выбирать по этой таблице нельзя)", "",
          T[~T["код"].isin(main_codes)][cols + ["в выборе"]].to_markdown(index=False), "",
          "## Пороговые метрики всех вариантов — поисково", "", TH[~TH["вариант"].isin([title[c] for c in main_codes])].to_markdown(index=False), ""]
    if rej_lines:
        md += ["## Выброс плохих окон", ""] + rej_lines + [""]
    hp = []
    for p in sorted(os.listdir(rdir)):
        if p.endswith(".json"):
            j = json.load(open(os.path.join(rdir, p), encoding="utf-8"))
            if j["variant"] in names:
                hp.append(dict(вариант=j["variant"], фолд=j["fold"], гиперпараметры=json.dumps(j["hp"]), признаков=j["n_features"], отложенные=round(j["val_auc"], 3)))
    if hp:
        md += ["## Гиперпараметры по фолдам", "", pd.DataFrame(hp).to_markdown(index=False), ""]
    txt = "\n".join(md)
    assert not any(p in txt for p in chosen), "в сводке не должно быть кодов пациентов"
    open(os.path.join(od, "tuep_variants_summary.md"), "w", encoding="utf-8").write(txt)
    print(txt); print("→", od, "(в git — только tuep_variants_summary.md, *_metrics.csv, *_thresholds.csv, *_nested_choice.csv)")


def main():
    ap = argparse.ArgumentParser(); sp = ap.add_subparsers(dest="cmd", required=True)
    f = sp.add_parser("features"); f.add_argument("--prep", required=True); f.add_argument("--out", required=True)
    f.add_argument("--configs", nargs="*", default=CONFIGS, choices=CONFIGS); f.add_argument("--procs", type=int, default=8)
    f.add_argument("--limit", type=int, default=0, help="только первые N файлов: время и оценка полного расчёта, без записи")
    e = sp.add_parser("embed"); e.add_argument("--prep", required=True); e.add_argument("--out", required=True)
    e.add_argument("--reve-dir", default=""); e.add_argument("--model", default="reve", choices=["reve", "toy"]); e.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    e.add_argument("--bs", type=int, default=64); e.add_argument("--device", default="cuda:0"); e.add_argument("--limit", type=int, default=0)
    r = sp.add_parser("run"); r.add_argument("--prep", required=True); r.add_argument("--out", required=True)
    r.add_argument("--ref-runs", default="", help="папка runs прежних запусков (reve, bpampw, tech); по умолчанию <out>/runs")
    r.add_argument("--folds", default=""); r.add_argument("--exclude-sessions", default="")
    r.add_argument("--balance", default="strata", choices=["strata", "class"]); r.add_argument("--min-pat", type=int, default=3)
    r.add_argument("--variants", nargs="*", default=[], help="только эти варианты (по умолчанию все)")
    r.add_argument("--emb-file", default="", help=argparse.SUPPRESS)
    a = ap.parse_args()
    for k in ("prep", "out", "folds", "exclude_sessions", "ref_runs", "emb_file"):
        if hasattr(a, k):
            setattr(a, k, FT.expand(getattr(a, k)))
    if a.cmd == "run" and not a.ref_runs:
        a.ref_runs = os.path.join(a.out, "runs")
    {"features": cmd_features, "embed": cmd_embed, "run": cmd_run}[a.cmd](a)


if __name__ == "__main__":
    main()
