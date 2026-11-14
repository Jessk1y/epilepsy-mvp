"""Задача 1.6: дообучение REVE v4 / EEG-DINO v2 на самом TUEP по 5 фолдам (разбиение по пациентам) и сводный отчёт.

    # пробный режим (минуты): 1 фолд, 1 запуск, 1 короткая эпоха; печатает скорость и оценку времени полного прогона
    CUDA_VISIBLE_DEVICES=0 python tuep_finetune.py train --model reve --prep ~/data/tuep_work/prep --out ~/data/tuep_work/ft \
        --reve-dir ~/data/weights/h1-reve-v4 --smoke
    # полный прогон (можно прерывать: готовые запуски пропускаются, незаконченный продолжается с последней эпохи)
    CUDA_VISIBLE_DEVICES=0 python tuep_finetune.py train --model reve --prep ~/data/tuep_work/prep --out ~/data/tuep_work/ft \
        --reve-dir ~/data/weights/h1-reve-v4
    python tuep_finetune.py report --prep ~/data/tuep_work/prep --out ~/data/tuep_work/ft --zero-shot ~/data/tuep_work/out/tuep_patient_scores.csv

Схема (подробно — README.md, раздел 6):
- внешние 5 фолдов по пациентам (tuep_protocol.make_folds, seed 0 — те же, что `tuep_protocol.py folds`); для фолда k: тест — фолд k,
  отложенная часть — фолд (k+1) % 5 (выбор эпохи и порога), обучение — остальные три. Тест фолда не участвует ни в чём до конца;
- один пациент — одна оценка: тот же случайный сеанс без приступов (seed 0), что в прогоне без дообучения; оценка файла — среднее
  верхних 20 % вероятностей 60 равномерных окон 10 с, оценка сеанса — среднее по файлам с весом длительности;
- обучение: все сеансы без приступов пациентов обучающей части; окно выбирается так: класс 50/50 -> слой «монтаж|частота» с ОДИНАКОВЫМ
  для обоих классов распределением -> пациент равновероятно -> сеанс равновероятно -> файл по длине -> случайное начало.
  Слои, где в одном из классов меньше --min-pat пациентов (03_tcp_ar_a, 512 Гц), в обучение не идут: по ним класс угадывается без ЭЭГ;
- сигнал: ФВЧ 0,3 Гц (как раньше) + ФНЧ 45 Гц (--lowpass; убирает область режекции 50/60 Гц и верх спектра, где видна аппаратура);
  фильтр — по куску «окно ± 5 с», одинаково в обучении и в оценке;
- метрики: ROC AUC и пороговые с 95 % ДИ (бутстреп по пациентам); порог — по отложенной части своего фолда; предсказания по людям — csv.

DeepEpilepsy авторов (--model de, 15.11): та же схема, но окна 30 с (20 окон на файл — те же 10 минут записи), вход и аугментации
авторов (каналы в их порядке, standard_scale по каналу, RandAugment m = 7), оценка файла — медиана вероятностей окон (как в статье).
Файлы короче 30 с оценки не получают. Подробно — README.md, раздел 8.

Данные TUH и выходы с кодами пациентов остаются на рабочей машине; в git — только tuep_ft_summary.md / tuep_ft_metrics.csv.
"""
import argparse, glob, importlib.util, json, os, sys, time
from concurrent.futures import ThreadPoolExecutor
import numpy as np, pandas as pd, scipy.signal as ss
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import tuep_common as C                                    # noqa: F401,E402  (подключает ../src)
import tuep_protocol as TP                                 # noqa: E402
from shared_data import EL, _shard                         # noqa: E402

FS, L, HP, CAP_WIN, TOP, MARGIN, N_FOLDS = 200, 2000, 0.3, 60, 0.2, 1000, 5
L10 = 2000                                                 # окно 10 с: в нём задаётся --n-win (60 окон = 10 минут записи)
WIN_S = {"de": 30}                                         # длина окна модели, с; у остальных 10
DE_ORDER = ["Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4", "F7", "F8", "T3", "T4", "T5", "T6", "O1", "O2", "Fz", "Cz", "Pz"]   # settings.ELECTRODES авторов
DE_IDX = [EL.index(e) for e in DE_ORDER]
MIRROR_PAIRS = {"Fp1": "Fp2", "F3": "F4", "C3": "C4", "P3": "P4", "O1": "O2", "F7": "F8", "T3": "T4", "T5": "T6"}
MIRROR_PAIRS.update({v: k for k, v in list(MIRROR_PAIRS.items())})
MIRROR = [EL.index(MIRROR_PAIRS.get(e, e)) for e in EL]
NAMES = {"dea": "DeepEpilepsy в условиях авторов (уши, без ФНЧ и режекции), дообучена на TUEP",
         "deafz": "DeepEpilepsy в условиях авторов (уши, без ФНЧ и режекции), учится только голова",
         "de": "DeepEpilepsy авторов, дообучена на TUEP", "defz": "DeepEpilepsy авторов, на TUEP учится только голова",
         "revefz": "REVE v4, на TUEP учится только голова", "bpampw": "эталон: мощности, амплитуда, wPLI + лог. регрессия", "bpampwd": "эталон: мощности, амплитуда, wPLI, возраст, пол + лог. регрессия", "demo": "только возраст и пол + лог. регрессия", "reve": "REVE v4, дообучена на TUEP", "dino": "EEG-DINO v2, дообучена на TUEP", "toy": "игрушечная сеть (тест)",
         "bp": "эталон: мощности ритмов + лог. регрессия", "bpamp": "эталон: мощности ритмов и амплитуда + лог. регрессия",
         "tech": "контроль утечки: только монтаж, частота, длительность, год (без ЭЭГ)"}


# ---------------------------------------------------------------- данные: сеансы, фолды, веса сеансов
def expand(p):
    return os.path.abspath(os.path.expanduser(p)) if p else p


def load_prep(prep):
    meta = pd.read_csv(os.path.join(prep, "h1_shared_meta.csv"))
    need = {"id", "y", "patient", "session", "montage", "sfreq_orig", "dur_s", "n_seiz", "shard200", "off200", "n200"}
    assert need <= set(meta.columns), f"в h1_shared_meta.csv нет столбцов: {sorted(need - set(meta.columns))}"
    return meta


def session_frame(meta, exclude_csv=""):
    """Сеансы: класс, приступы, монтаж, частота, слой «монтаж|частота». exclude_csv (patient, session) — сеансы, которые считаем
    сеансами с приступами дополнительно (например, по заключениям врачей из metadata_v00r.xlsx, см. tuep_audit.py)."""
    S = TP.session_table(meta, [])
    if exclude_csv:
        ex = pd.read_csv(exclude_csv)
        keys = set(zip(ex.patient.astype(str), ex.session.astype(str).str[:4]))
        add = np.array([(p, s[:4]) in keys for p, s in zip(S.patient, S.session)])
        print(f"дополнительно исключено сеансов по списку: {int((add & ~S.has_seiz).sum())}")
        S["has_seiz"] = S.has_seiz | add
    S["stratum"] = S.montage.astype(str) + "|" + S.sfreq.astype(str)
    return S


def chosen_sessions(S, seed=0):
    """Один случайный сеанс без приступов на пациента — то же правило и тот же seed, что в tuep_protocol.py eval (PROTOCOL.md, п. 2)."""
    P, dropped = TP.select_sessions(S, [], "random", seed)
    return dict(zip(P.patient, P.session)), dropped


def get_folds(meta, folds_csv=""):
    F = pd.read_csv(folds_csv)[["patient", "y", "fold"]] if folds_csv else TP.make_folds(meta, N_FOLDS, 0)[["patient", "y", "fold"]]
    assert set(F.patient) == set(meta.patient) and F.patient.is_unique, "фолды не соответствуют списку пациентов"
    return F


def split_patients(F, k):
    """(обучение, отложенная часть, тест) для внешнего фолда k."""
    n = int(F.fold.max()) + 1
    assert n >= 3, "нужно не меньше 3 фолдов"
    te, va = set(F.patient[F.fold == k]), set(F.patient[F.fold == (k + 1) % n])
    fit = set(F.patient) - te - va
    assert not (te & va) and not (fit & te) and not (fit & va)
    return fit, va, te


def session_weights(S, balance="strata", min_pat=3):
    """Вероятность выбрать сеанс при обучении (сумма 1). Сеансы с приступами — 0.
    class:  класс 50/50 -> пациент равновероятно -> сеанс равновероятно (у больных сеансов втрое больше, вклад пациента от этого не растёт).
    strata: то же, но сначала слой «монтаж|частота» с одним распределением q для обоих классов, q ~ min(пациентов класса 0, класса 1);
            слои, где в одном из классов меньше min_pat пациентов, получают q = 0. Тогда слой не несёт сведений о классе."""
    S = S.reset_index(drop=True)
    ok = ~S.has_seiz.to_numpy()
    st = S.stratum.to_numpy() if balance == "strata" else np.array(["все"] * len(S))
    T = pd.DataFrame({"st": st, "y": S.y, "patient": S.patient})[ok]
    npat = T.groupby(["st", "y"]).patient.nunique().unstack(fill_value=0).reindex(columns=[0, 1], fill_value=0)
    q = npat.min(axis=1).astype(float)
    q[q < (min_pat if balance == "strata" else 1)] = 0.0
    assert q.sum() > 0, "нет слоя, где хватает пациентов обоих классов"
    q = q / q.sum()
    n_sess = T.groupby(["patient", "st"]).size()
    w = np.zeros(len(S))
    for i in np.where(ok)[0]:
        if q[st[i]] > 0:
            w[i] = 0.5 * q[st[i]] / npat.loc[st[i], S.y[i]] / n_sess[(S.patient[i], st[i])]
    tab = npat.rename(columns={0: "пациентов без эпилепсии", 1: "пациентов с эпилепсией"}).assign(доля_в_обучении=q.round(3))
    return w / w.sum(), tab


# ---------------------------------------------------------------- сигнал: окно с полями, фильтр, кэш окон для оценки
def make_sos(lowpass):
    sos = ss.butter(4, HP, btype="highpass", fs=FS, output="sos")
    if lowpass:
        sos = np.vstack([sos, ss.butter(6, lowpass, btype="lowpass", fs=FS, output="sos")])
    return sos


def set_window(model):
    """Длина окна зависит от модели: 10 с (REVE, EEG-DINO, эталон) или 30 с (DeepEpilepsy). Меняет L для всех функций ниже."""
    global L
    L = FS * WIN_S.get(model, 10)
    return L


def n_windows(n_win):
    """--n-win задан в окнах по 10 с; для более длинного окна — столько же секунд записи (60 × 10 с = 20 × 30 с)."""
    return max(1, n_win * L10 // L)


def read_window(prep, shard, off, n, start, sos):
    """Окно L отсчётов: читается кусок с полями ±5 с (сколько есть), фильтруется, вырезается середина. Одинаково в обучении и в оценке."""
    a, b = max(0, start - MARGIN), min(n, start + L + MARGIN)
    X = np.asarray(_shard(prep, FS, shard)[off:off + 19 * n].reshape(19, n)[:, a:b], np.float64)
    return ss.sosfiltfilt(sos, X, axis=-1)[:, start - a:start - a + L].astype(np.float32)


def spread(n, k):
    return np.unique(np.linspace(0, n - 1, k).astype(int)) if n > k else np.arange(n)


def top20(p):
    return float(np.sort(p)[::-1][:int(np.ceil(TOP * len(p)))].mean())


def median(p):
    return float(np.median(p))


def eval_windows(prep, r, sos, n_win, cache_dir=""):
    """Окна файла для оценки: до n_win равномерно из окон длины L подряд с начала записи (как в блокнотах и tuep_infer.py). (k, 19, L) float32.
    Файл короче окна — пустой массив (k = 0). Кэш на диске (float16, ~4,6 МБ на файл при 10 минутах записи): фильтрация делается один раз
    для всех эпох, запусков и моделей с этой длиной окна."""
    n = int(r.n200)
    if n < L:
        return np.zeros((0, 19, L), np.float32)
    p = os.path.join(cache_dir, f"{r.id}.npy") if cache_dir else ""
    if p and os.path.exists(p):
        return np.load(p).astype(np.float32)
    W = np.stack([read_window(prep, r.shard200, int(r.off200), n, int(s) * L, sos) for s in spread(n // L, n_win)])
    if p:
        tmp = p + f".{os.getpid()}.tmp.npy"
        np.save(tmp, np.clip(W, -C.F16_MAX, C.F16_MAX).astype(np.float16)); os.replace(tmp, p)
    return W


def score_files(prob_fn, prep, rows, sos, n_win, cache_dir="", workers=4, agg=top20):
    """Оценка каждого файла: agg (среднее верхних 20 % или медиана) вероятностей окон. prob_fn: (k, 19, L) -> (k,).
    Возвращает {id: оценка} и число окон; файлов короче окна в словаре нет."""
    out, n = {}, 0
    with ThreadPoolExecutor(workers) as ex:
        for r, W in zip(rows, ex.map(lambda r: eval_windows(prep, r, sos, n_win, cache_dir), rows)):
            if len(W):
                out[r.id] = agg(prob_fn(W)); n += len(W)
    return out, n


def patient_scores(rec, chosen):
    """Файлы -> выбранный сеанс пациента: среднее по файлам с весом длительности. rec: patient, session, y, dur_s, score."""
    d = rec[[chosen.get(p) == s for p, s in zip(rec.patient, rec.session)]]
    g = d.assign(ws=d.score * d.dur_s).groupby("patient").agg(y=("y", "first"), ws=("ws", "sum"), w=("dur_s", "sum"))
    return (g.ws / g.w).rename("score").to_frame().assign(y=g.y).reset_index()


class WindowSampler:
    """Случайные окна обучения: сеанс — по весам session_weights, файл — пропорционально числу возможных начал, начало — любой отсчёт."""
    def __init__(self, meta_fit, S_fit, w):
        self.w = w
        files = {k: g for k, g in meta_fit.groupby(["patient", "session"])}
        self.sess = []
        for p, s, y, wi in zip(S_fit.patient, S_fit.session, S_fit.y, w):
            if wi > 0:
                g = files[(p, s)]
                nst = (g.n200.to_numpy() - L + 1).clip(min=0).astype(float)
                self.sess.append((int(y), list(zip(g.shard200, g.off200.astype(int), g.n200.astype(int))), nst / nst.sum()))
        self.p = w[w > 0] / w[w > 0].sum()

    def draw(self, B, g):
        items = []
        for i in g.choice(len(self.sess), size=B, p=self.p):
            y, fl, pf = self.sess[i]
            shard, off, n = fl[int(g.choice(len(fl), p=pf))]
            items.append((shard, off, n, int(g.integers(0, n - L + 1)), y))
        return items


# ---------------------------------------------------------------- модели
_POS = {}


def norm_input(W, torch):
    """Как rv_input REVE v4: из канала вычесть среднее по окну, окно поделить на общее СКО, обрезка ±15."""
    x = np.asarray(W, np.float32)
    x = x - x.mean(-1, keepdims=True)
    return torch.from_numpy(np.clip(x / (x.std(axis=(1, 2), keepdims=True) + 1e-6), -15, 15))


def de_input(W, torch):
    """Вход DeepEpilepsy: каналы в порядке авторов, standard_scale — каждый канал окна к нулевому среднему и единичному СКО (augmentation.py)."""
    x = np.asarray(W, np.float32)[:, DE_IDX]
    return torch.from_numpy((x - x.mean(-1, keepdims=True)) / (x.std(-1, keepdims=True) + 1e-20))


def de_randaugment(x, g, m=7):
    """RandAugment авторов DeepEpilepsy (ml/augmentation.py, MIT; n = 1, m = 7 — proj_config.yaml), перенесён на генератор g, чтобы обучение
    продолжалось после обрыва с того же места. Одно окно (19, n) в мкВ, ДО standard_scale. Выбирается один приём из шести, срабатывает
    с вероятностью 0,5: режекция случайной полосы 21 Гц, ФНЧ 30–50 Гц, зануление прямоугольника «каналы × время», зануление каналов
    (каждый с p = 7/12), масштаб амплитуды части каналов, шум в части каналов."""
    op, on = int(g.integers(0, 6)), g.random() < 0.5
    if not on or op == 4:                                     # масштаб амплитуды канала стирается нормировкой по каналу — приём пустой
        return x
    nc, nt = x.shape
    if op == 0:
        bw = 0.05 * FS + (FS / 2 - 0.1 * FS) / m
        lo = g.uniform(1, FS / 2 - bw)
        return ss.sosfilt(ss.butter(4, [lo, min(lo + bw, FS / 2 - 0.01)], btype="bandstop", fs=FS, output="sos"), x, axis=-1)
    if op == 1:
        f = max(min(FS / 2 - (FS / 2 - 15) * m / 10 + g.uniform(-FS / 20, FS / 20), FS / 2 - 1), 15)
        return ss.sosfilt(ss.butter(4, f, btype="lowpass", fs=FS, output="sos"), x, axis=-1)
    x = np.array(x, np.float64)
    if op == 2:
        h = min(int(g.integers(int((m - 1) / 10 * nc / 2) + 1, int(m / 10 * nc) + 1)), nc)
        w = min(int(g.integers(int((m - 1) / 10 * nt / 2) + 1, int(m / 10 * nt) + 1)), nt)
        c0, t0 = int(g.integers(0, nc - h + 1)), int(g.integers(0, nt - w + 1))
        x[c0:c0 + h, t0:t0 + w] = 0
    elif op == 3:
        x[g.random(nc) < m / 12] = 0
    else:
        mask = g.random(nc) < m / 12
        x[mask] += g.normal(0, m / 10 * 0.1 * np.ptp(x, axis=-1).mean(), (int(mask.sum()), nt))
    return x


def de_vit(a, torch):
    """ViT1d_Conv1d_large_50 авторов с их весами (код — клон gitlab chum-epilepsy/dl_epilepsy_reeg, веса — HF emilelemoine/DeepEpilepsy_v1)."""
    nn = torch.nn
    if a.de_toy:                                              # синтетический тест: тот же путь данных, без кода и весов авторов
        class Tiny(nn.Module):
            def __init__(self):
                super().__init__(); self.stem = nn.Conv1d(19, 8, 50, stride=25); self.mlp_head = nn.Linear(8, 2)
            def forward(self, x):
                return self.mlp_head(self.stem(x).abs().mean(-1))
        return Tiny()
    src = os.path.join(a.de_repo, "src", "epileptology", "ml", "models", "transformers.py")
    assert os.path.exists(src), f"нет кода авторов: {src} (git clone https://gitlab.com/chum-epilepsy/dl_epilepsy_reeg.git {a.de_repo})"
    spec = importlib.util.spec_from_file_location("de_transformers", src)
    de = importlib.util.module_from_spec(spec); spec.loader.exec_module(de)
    vit = de.ViT1d(n_class=2, in_channels=19, n_timepoints=L, patch_size=50, tokenizer="conv1d",
                   n_layers=6, n_heads=8, dim_hidden=512, dim_mlp=512, dropout=0.25, attention_dropout=0.25)
    w = a.de_weights
    if not w:
        from huggingface_hub import hf_hub_download
        w = hf_hub_download("emilelemoine/DeepEpilepsy_v1", "DeepEpilepsy_large_50_30s_augment.pt")
    vit.load_state_dict(torch.load(w, map_location="cpu", weights_only=True), strict=True)
    return vit


def build_model(name, a, torch, dev, seed):
    """Модель, функция входа, группы параметров оптимизатора (как в блокнотах обучения на NMT), доля разогрева, имя основы,
    функция «окна обучения + генератор -> вход с аугментациями на устройстве»."""
    nn = torch.nn
    torch.manual_seed(seed)
    train_in = None
    if name == "toy":                                         # для синтетического теста: без весов и без интернета
        class Toy(nn.Module):
            def __init__(self):
                super().__init__(); self.backbone = nn.Conv1d(19, 8, 50, stride=25); self.head = nn.Linear(8, 1)
            def forward(self, x):
                return self.head(self.backbone(x).abs().mean(-1)).squeeze(-1)
        m = Toy(); inp = lambda W: norm_input(W, torch); back, noise, warm = "backbone", 0.1, 0.05
        groups = lambda mm: [dict(params=[p for p in mm.parameters() if p.requires_grad], lr=1e-2 * a.lr_scale, weight_decay=0.0)]
    elif name == "reve":
        sys.path.insert(0, a.reve_dir)
        from reve_v4_model import RVNet, rv_input, positions
        if "pos" not in _POS:
            _POS["pos"] = positions(dev)
        m = RVNet(_POS["pos"]); inp, back, noise, warm = rv_input, "backbone", 0.1, 0.05
        if a.init == "nmt":
            m.load_state_dict(torch.load(os.path.join(a.reve_dir, f"reve_v4_seed{seed}.pt"), map_location="cpu", weights_only=False)["state_dict"])
        def groups(mm):                                       # h1_2026-10-27_zs_reve-v4: основа 1e-5, голова 5e-4, AdamW 0,05
            bk = [p for n, p in mm.named_parameters() if n.startswith("backbone") and p.requires_grad]
            hd = [p for n, p in mm.named_parameters() if not n.startswith("backbone") and p.requires_grad]
            return [g for g in (dict(params=bk, lr=1e-5 * a.lr_scale, weight_decay=0.05), dict(params=hd, lr=5e-4 * a.lr_scale, weight_decay=0.05)) if g["params"]]
    elif name == "dino":
        sys.path.insert(0, a.dino_dir)
        import eegdino_v2_model as DM
        m = DM.DinoNet(src_dir=a.dino_src or None); inp, back, noise, warm = DM.dino_input, "encoder", 0.1 * DM.STD_UV / 100.0, 0.10
        if a.init == "nmt":
            m.load_state_dict(torch.load(os.path.join(a.dino_dir, f"eegdino_v2_seed{seed}.pt"), map_location="cpu", weights_only=False)["state_dict"])
        def groups(mm):                                       # h1_2026-11-02_zs_eegdino-v2: 5e-4, послойное снижение 0,9, нормы и смещения без weight decay
            def scale(n):
                if n.startswith("encoder.encoder_layers."):
                    return 0.9 ** (DM.N_LAYERS - int(n.split(".")[2]))
                return 0.9 ** (DM.N_LAYERS + 1) if n.startswith("encoder.") else 1.0
            gr = {}
            for n, p in mm.named_parameters():
                if p.requires_grad:
                    gr.setdefault((scale(n), p.ndim > 1), []).append(p)
            return [dict(params=ps, lr=5e-4 * a.lr_scale * k, weight_decay=0.05 if dec else 0.0) for (k, dec), ps in gr.items()]
    elif name == "de":
        class DENet(nn.Module):
            """Основа авторов + их же линейная голова на 2 класса. Выход — логит класса 1 (z1 − z0): сигмоида от него равна softmax[:, 1],
            потери BCE равны перекрёстной энтропии авторов."""
            def __init__(self, vit):
                super().__init__(); self.head = vit.mlp_head; vit.mlp_head = nn.Identity(); self.backbone = vit
            def forward(self, x):
                z = self.head(self.backbone(x))
                return z[:, 1] - z[:, 0]
        m = DENet(de_vit(a, torch)); inp, back, noise, warm = (lambda W: de_input(W, torch)), "backbone", 0.0, 0.05
        train_in = lambda X, g: inp(np.stack([de_randaugment(x, g) for x in X])).to(dev)
        def groups(mm):                                       # основа 2e-5 (как при дообучении на NMT, блокнот 20.10), голова 5e-4 (как у REVE), AdamW 0,05
            bk = [p for n, p in mm.named_parameters() if n.startswith("backbone") and p.requires_grad]
            hd = [p for n, p in mm.named_parameters() if not n.startswith("backbone") and p.requires_grad]
            return [g for g in (dict(params=bk, lr=2e-5 * a.lr_scale, weight_decay=0.05), dict(params=hd, lr=5e-4 * a.lr_scale, weight_decay=0.05)) if g["params"]]
    else:
        raise ValueError(name)
    if train_in is None:
        train_in = lambda X, g: augment(inp(X).to(dev), g, torch, noise)
    if a.freeze_backbone:
        for n, p in m.named_parameters():
            if n.startswith(back):
                p.requires_grad = False
    return m.to(dev), inp, groups, warm, back, train_in


def augment(x, g, torch, noise):
    """Как в блокнотах: шум 0,1·СКО, замена полушарий p = 0,5, зануление 1–2 каналов p = 0,5, сдвиг до ±2 с."""
    x = x + noise * torch.randn_like(x)
    m = torch.from_numpy(g.random(len(x)) < 0.5).to(x.device)
    x[m] = x[m][:, MIRROR]
    for b in range(len(x)):
        if g.random() < 0.5:
            x[b, g.choice(19, int(g.integers(1, 3)), replace=False).tolist()] = 0
        x[b] = torch.roll(x[b], int(g.integers(-2 * FS, 2 * FS + 1)), dims=-1)
    return x


# ---------------------------------------------------------------- обучение одного запуска (фолд, seed) с продолжением после обрыва
def save_atomic(torch, obj, path):
    tmp = path + ".tmp"; torch.save(obj, tmp); os.replace(tmp, path)


def run_one(a, torch, meta, S, F, chosen, fold, seed, dirs):
    tag = f"{a.name or a.model}{'fz' if a.freeze_backbone else ''}_f{fold}_s{seed}"          # замороженная основа — отдельное имя (revefz)
    out_csv, ck_path = os.path.join(dirs["runs"], tag + ".csv"), os.path.join(dirs["ckpt"], tag + ".pt")
    if os.path.exists(out_csv):
        print(f"[{tag}] готово, пропуск"); return None
    dev, cuda = a.device, a.device.startswith("cuda")
    fit_p, val_p, test_p = split_patients(F, fold)
    S_fit = S[S.patient.isin(fit_p)].reset_index(drop=True)
    long_ok = set(zip(meta.patient[meta.n200 >= L], meta.session[meta.n200 >= L]))
    short = np.array([(p, s) not in long_ok for p, s in zip(S_fit.patient, S_fit.session)])
    if short.any():                                           # сеансы без единого файла длиной в окно в --prep в обучение не идут (как сеансы с приступами)
        print(f"[{tag}] сеансов обучающей части без файла длиной {L // FS} с в --prep: {int((short & ~S_fit.has_seiz).sum())} — в обучение не идут")
        S_fit = S_fit.assign(has_seiz=S_fit.has_seiz | short)
    w, tab = session_weights(S_fit, a.balance, a.min_pat)
    ok = {(p, s) for p, s, wi in zip(S_fit.patient, S_fit.session, w) if wi > 0}
    meta_fit = meta[[k in ok for k in zip(meta.patient, meta.session)]]
    sampler = WindowSampler(meta_fit, S_fit, w)
    val_rows = [r for r in meta[meta.patient.isin(val_p)].itertuples() if chosen.get(r.patient) == r.session]
    test_rows = list(meta[meta.patient.isin(test_p)].itertuples())
    if a.smoke:
        test_rows = test_rows[:40]
    agg, n_win = (median if a.model == "de" else top20), n_windows(a.n_win)
    n_short = sum(int(r.n200) < L for r in val_rows + test_rows)
    no_score = [p for p in (val_p | test_p) if p in chosen and (p, chosen[p]) not in long_ok]
    if n_short or no_score:
        print(f"[{tag}] файлов короче {L // FS} с (без оценки): {n_short}; пациентов, у которых выбранный сеанс остался без оценки: {len(no_score)}")
    print(f"[{tag}] пациентов: обучение {len(fit_p)}, отложенные {len(val_p)}, тест {len(test_p)} | сеансов в обучении {len(sampler.sess)} из {len(S_fit)} "
          f"| файлов: отложенные {len(val_rows)}, тест {len(test_rows)}")
    print(tab.to_string())

    model, inp, groups, warm, back, train_in = build_model(a.model, a, torch, dev, seed)
    steps_ep = int(np.ceil(a.epoch_windows / a.bs)); total = a.max_epochs * steps_ep
    opt = torch.optim.AdamW(groups(model))
    warm_n = max(1, int(warm * total))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / warm_n if s < warm_n else 0.5 * (1 + np.cos(np.pi * (s - warm_n) / max(1, total - warm_n))) * 0.99 + 0.01)
    scaler = torch.amp.GradScaler("cuda", enabled=cuda)
    amp = dict(device_type="cuda" if cuda else "cpu", dtype=torch.float16 if cuda else torch.bfloat16, enabled=cuda)
    g = np.random.default_rng(1000 * fold + seed)
    st = dict(epoch=0, hist=[], best_auc=-1.0, best_ep=0, best_state=None, best_val=None, sec=0.0, n_bad=0)
    if os.path.exists(ck_path):
        ck = torch.load(ck_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"]); scaler.load_state_dict(ck["scaler"])
        g.bit_generator.state = ck["np_rng"]; torch.set_rng_state(ck["torch_rng"])
        if cuda and ck.get("cuda_rng") is not None:
            torch.cuda.set_rng_state_all(ck["cuda_rng"])
        st = ck["st"]
        print(f"[{tag}] продолжение после эпохи {st['epoch']}")

    def prob_fn(W):
        out = []
        with torch.no_grad(), torch.autocast(**amp):
            for b in range(0, len(W), a.bs_eval):
                out.append(torch.sigmoid(model(inp(W[b:b + a.bs_eval]).to(dev)).float()).cpu().numpy())
        return np.concatenate(out)

    def val_eval():
        model.eval()
        sc, n = score_files(prob_fn, a.prep, val_rows, dirs["sos"], n_win, dirs["cache"], a.workers, agg)
        rec = pd.DataFrame([dict(id=r.id, patient=r.patient, session=r.session, y=r.y, dur_s=r.dur_s, score=sc[r.id]) for r in val_rows if r.id in sc])
        P = patient_scores(rec, chosen)
        return roc_auc_score(P.y, P.score), sc, n

    F_bce = torch.nn.functional.binary_cross_entropy_with_logits
    params = [p for p in model.parameters() if p.requires_grad]
    pool = ThreadPoolExecutor(a.workers)
    t_train = t_val = n_train = n_val = 0
    while st["epoch"] < a.max_epochs and st["epoch"] - st["best_ep"] < a.patience:
        t0 = time.time(); model.train(); losses = []
        if a.freeze_backbone:
            getattr(model, back).eval()
        for _ in range(steps_ep):
            items = sampler.draw(a.bs, g)
            X = np.stack(list(pool.map(lambda it: read_window(a.prep, it[0], it[1], it[2], it[3], dirs["sos"]), items)))
            x = train_in(X, g)
            t = torch.tensor([it[4] for it in items], device=dev, dtype=torch.float32)
            with torch.autocast(**amp):
                out = model(x)
            loss = F_bce(out.float(), t)
            opt.zero_grad()
            if torch.isfinite(loss):
                scaler.scale(loss).backward(); scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(params, 1.0)
                scaler.step(opt); scaler.update()
            else:
                st["n_bad"] += 1
            sched.step(); losses.append(loss.item())
        t1 = time.time(); auc, sc, nv = val_eval(); t2 = time.time()
        t_train += t1 - t0; t_val += t2 - t1; n_train += steps_ep * a.bs; n_val += nv
        st["epoch"] += 1; st["sec"] += t2 - t0
        st["hist"].append(dict(epoch=st["epoch"], loss=round(float(np.nanmean(losses)), 4), val_auc=round(float(auc), 4)))
        if auc > st["best_auc"]:
            st.update(best_auc=float(auc), best_ep=st["epoch"], best_val=sc, best_state={k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
        print(f"[{tag}] эпоха {st['epoch']}: потери {np.nanmean(losses):.3f} | отложенные (по пациентам) ROC AUC {auc:.3f} | лучшая {st['best_ep']} "
              f"| NaN/inf {st['n_bad']} | обучение {t1-t0:.0f} с, оценка {t2-t1:.0f} с", flush=True)
        save_atomic(torch, dict(model=model.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(), scaler=scaler.state_dict(),
                                np_rng=g.bit_generator.state, torch_rng=torch.get_rng_state(),
                                cuda_rng=torch.cuda.get_rng_state_all() if cuda else None, st=st), ck_path)
        if a.debug_stop_epoch and st["epoch"] >= a.debug_stop_epoch:
            print(f"[{tag}] остановка для проверки продолжения"); pool.shutdown(); sys.exit(3)
    pool.shutdown()
    model.load_state_dict(st["best_state"]); model.eval()
    t0 = time.time()
    sc_test, nt = score_files(prob_fn, a.prep, test_rows, dirs["sos"], n_win, dirs["cache"], a.workers, agg)
    t_test = time.time() - t0
    rows = [dict(split="val", fold=fold, seed=seed, id=r.id, patient=r.patient, session=r.session, y=r.y, dur_s=r.dur_s, score=st["best_val"][r.id]) for r in val_rows if r.id in st["best_val"]]
    rows += [dict(split="test", fold=fold, seed=seed, id=r.id, patient=r.patient, session=r.session, y=r.y, dur_s=r.dur_s, score=sc_test[r.id]) for r in test_rows if r.id in sc_test]
    info = dict(tag=tag, model=tag.split("_f")[0], fold=fold, seed=seed, best_epoch=st["best_ep"], best_val_auc=st["best_auc"], hist=st["hist"], n_bad=st["n_bad"],
                minutes=round((st["sec"] + t_test) / 60, 1), init="authors" if a.model == "de" else a.init, win_s=L // FS, n_win=n_win, agg=agg.__name__, freeze_backbone=a.freeze_backbone, balance=a.balance, lowpass=a.lowpass,
                epoch_windows=a.epoch_windows, bs=a.bs, lr_scale=a.lr_scale, n_fit_patients=len(fit_p), n_fit_sessions=len(sampler.sess))
    json.dump(info, open(os.path.join(dirs["runs"], tag + ".json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    if a.save_weights:
        save_atomic(torch, dict(state_dict=st["best_state"], config=info), os.path.join(dirs["weights"], tag + ".pt"))
    tmp = out_csv + ".tmp"; pd.DataFrame(rows).to_csv(tmp, index=False); os.replace(tmp, out_csv)      # csv — признак готовности запуска
    os.remove(ck_path)
    print(f"[{tag}] лучшая эпоха {st['best_ep']} (отложенные {st['best_auc']:.3f}), {info['minutes']} мин → {out_csv}", flush=True)
    return dict(train_wps=n_train / max(t_train, 1e-9) if n_train else np.nan, eval_wps=(n_val + nt) / max(t_val + t_test, 1e-9),
                n_test_files=len(test_rows), n_val_files=len(val_rows))


def cmd_train(a):
    import torch
    for k in ("prep", "out", "reve_dir", "dino_dir", "dino_src", "folds", "exclude_sessions", "de_repo", "de_weights", "proto_prep"):
        setattr(a, k, expand(getattr(a, k)))
    set_window(a.model)
    if a.smoke:
        a.out, a.fold_list, a.seeds, a.max_epochs, a.epoch_windows, a.n_win = os.path.join(a.out, "smoke"), a.fold_list[:1], a.seeds[:1], 1, 10 * a.bs, 12
    if not torch.cuda.is_available() and a.device.startswith("cuda"):
        print("CUDA недоступна — считаю на процессоре (годится только для игрушечной модели)"); a.device = "cpu"
    meta = load_prep(a.prep)
    proto = load_prep(a.proto_prep) if a.proto_prep else meta      # сеансы, фолды и выбранный сеанс — всегда по полному списку файлов
    if a.proto_prep:
        assert set(meta.id) <= set(proto.id), "в --prep есть файлы, которых нет в --proto-prep"
        print(f"протокол (сеансы, слои, фолды, выбранный сеанс) — по {a.proto_prep}; сигнал — из {a.prep}: файлов {len(meta)} из {len(proto)}, "
              f"пациентов {meta.patient.nunique()} из {proto.patient.nunique()}")
    S = session_frame(proto, a.exclude_sessions)
    F = get_folds(proto, a.folds)
    chosen, dropped = chosen_sessions(S)
    lp = f"lp{int(a.lowpass)}" if a.lowpass else "lp0"
    dirs = dict(runs=os.path.join(a.out, "runs"), ckpt=os.path.join(a.out, "ckpt"), weights=os.path.join(a.out, "weights"),
                cache="" if a.no_cache else os.path.join(a.out, f"evalcache_{lp}_w{a.n_win}" + (f"_L{L}" if L != L10 else "")), sos=make_sos(a.lowpass))
    for k in ("runs", "ckpt", "weights", "cache"):
        if dirs[k]:
            os.makedirs(dirs[k], exist_ok=True)
    cfg = dict(lowpass=a.lowpass, balance=a.balance, min_pat=a.min_pat, n_win=a.n_win, exclude_sessions=bool(a.exclude_sessions), smoke=a.smoke)
    if a.proto_prep:
        cfg["signal_prep"] = os.path.basename(a.prep.rstrip("/"))
    cfg_path = os.path.join(a.out, "config.json")
    if os.path.exists(cfg_path):
        old = json.load(open(cfg_path, encoding="utf-8"))
        assert old == cfg, f"в {a.out} уже есть запуски с другими настройками: {old} != {cfg}. Возьмите другую папку --out"
    json.dump(cfg, open(cfg_path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"пациентов {meta.patient.nunique()}, сеансов {len(S)} (с приступами {int(S.has_seiz.sum())}), выпали целиком: {len(dropped)} | устройство: "
          + (torch.cuda.get_device_name(0) if a.device.startswith("cuda") else "процессор"))
    last = None
    for fold in a.fold_list:
        for seed in a.seeds:
            r = run_one(a, torch, meta, S, F, chosen, fold, seed, dirs)
            last = r or last
    if a.smoke and last:
        n_runs = N_FOLDS * 3
        tr = 8 * 4000 / last["train_wps"]
        ev = (8 * len(meta) / N_FOLDS / 3.5 + len(meta) / N_FOLDS) * n_windows(CAP_WIN) / last["eval_wps"]
        print(f"\nскорость: обучение {last['train_wps']:.0f} окон/с, оценка {last['eval_wps']:.0f} окон/с (первый проход оценки включает фильтрацию и запись кэша).")
        print(f"оценка времени полного прогона одной модели (5 фолдов × 3 запуска, до 8 эпох по 4000 окон): не больше ~{n_runs * (tr + ev) / 3600:.1f} ч; "
              f"обычно меньше — ранняя остановка и кэш окон.")
        print("пробный режим пройден; его результаты лежат отдельно в", a.out, "и в отчёт не попадают")


# ---------------------------------------------------------------- сводный отчёт
def boot_metr(y, pr, n=2000, seed=0):
    m0 = TP.metr(y, pr); r = np.random.default_rng(seed); B = []
    for _ in range(n):
        i = r.integers(0, len(y), len(y)); B.append(TP.metr(y[i], pr[i]))
    return {k: f"{m0[k]:.2f} ({np.nanpercentile([b[k] for b in B], 2.5):.2f}-{np.nanpercentile([b[k] for b in B], 97.5):.2f})" for k in m0}


def paired_diff(y, s1, s2, n=2000, seed=0):
    r, v = np.random.default_rng(seed), []
    for _ in range(n):
        i = r.integers(0, len(y), len(y))
        if len(set(y[i])) == 2:
            v.append(roc_auc_score(y[i], s1[i]) - roc_auc_score(y[i], s2[i]))
    return roc_auc_score(y, s1) - roc_auc_score(y, s2), *np.percentile(v, [2.5, 97.5])


def load_runs(run_dir):
    """Все готовые запуски: модель -> таблица (split, fold, id, …, score), оценка файла усреднена по запускам (seed)."""
    out = {}
    for p in sorted(glob.glob(os.path.join(run_dir, "*_f*_s*.csv"))):
        out.setdefault(os.path.basename(p).split("_f")[0], []).append(pd.read_csv(p))
    res = {}
    for m, parts in out.items():
        d = pd.concat(parts)
        res[m] = (d.groupby(["split", "fold", "id", "patient", "session", "y", "dur_s"], as_index=False)
                   .agg(score=("score", "mean"), n_seeds=("seed", "nunique")))
    return res


def cmd_report(a):
    for k in ("prep", "out", "folds", "exclude_sessions", "zero_shot"):
        setattr(a, k, expand(getattr(a, k)))
    meta = load_prep(a.prep)
    S = session_frame(meta, a.exclude_sessions)
    F = get_folds(meta, a.folds)
    chosen, dropped = chosen_sessions(S)
    R = load_runs(os.path.join(a.out, "runs"))
    assert R, "нет готовых запусков в " + os.path.join(a.out, "runs")
    for x in a.extra_runs:                                    # запуски из других папок (эталон, прежние модели) — для сравнения в одной таблице
        for m, d in load_runs(expand(x)).items():
            if m in R:
                print(f"{m}: уже есть в {a.out}, из {x} не берётся")
            else:
                R[m] = d
    for ens in (a.ensemble or [["reve", "dino"]]):            # --ensemble можно повторять: каждая пара — отдельная строка таблицы
        if len(ens) < 2 or any(m not in R for m in ens):
            if len(ens) >= 2:
                print(f"пара {'+'.join(ens)} пропущена: нет запусков", [m for m in ens if m not in R])
            continue
        key = ["split", "fold", "id", "patient", "session", "y", "dur_s"]   # пара: среднее вероятностей по файлам (как ens в tuep_infer.py)
        d = pd.concat([R[m] for m in ens]).groupby(key, as_index=False).agg(score=("score", "mean"), k=("score", "size"), n_seeds=("n_seeds", "min"))
        R["+".join(ens)] = d[d.k == len(ens)].drop(columns="k")              # только файлы с оценкой у обеих моделей
        NAMES["+".join(ens)] = "пара: " + " + ".join(NAMES.get(m, m) for m in ens)
    sfx, note = "", ""
    if a.same_patients:                                       # все строки — на пациентах, у которых есть оценка этой модели
        assert a.same_patients in R, f"нет запусков {a.same_patients}"
        d0 = R[a.same_patients]
        keep = set(patient_scores(d0[d0.split == "test"], chosen).patient)
        R = {m: d[(d.split != "test") | d.patient.isin(keep)] for m, d in R.items()}
        sfx = "_same_" + a.same_patients.replace("+", "_")
        note = f" **Все строки посчитаны на одних и тех же пациентах — тех, у кого есть оценка `{a.same_patients}` ({len(keep)}).**"
    info = S.set_index(["patient", "session"])[["montage", "sfreq", "stratum", "has_seiz", "n_files"]]
    zs = pd.read_csv(a.zero_shot) if a.zero_shot else None
    P_all, rows, thr_rows, sens_rows = None, [], [], []
    for m, d in R.items():
        te, va = d[d.split == "test"], d[d.split == "val"]
        folds_done = sorted(te.fold.unique())
        P = patient_scores(te, chosen).merge(F[["patient", "fold"]], on="patient")
        P["session"] = P.patient.map(chosen)
        P = P.join(info, on=["patient", "session"])
        y, s = P.y.to_numpy(), P.score.to_numpy()
        assert P.patient.is_unique
        auc, lo, hi = TP.boot_auc(y, s)
        per_fold = [roc_auc_score(g.y, g.score) for _, g in P.groupby("fold") if g.y.nunique() == 2]
        row = {"модель": NAMES.get(m, m), "код": m, "фолдов": len(folds_done), "запусков на фолд": int(te.n_seeds.min()), "n": len(P), "n_эпилепсия": int(y.sum()),
               "ROC AUC (95 % ДИ)": f"{auc:.3f} ({lo:.2f}-{hi:.2f})", "по фолдам, среднее ± СКО": f"{np.mean(per_fold):.3f} ± {np.std(per_fold):.3f}",
               "пары внутри слоя монтаж|частота": f"{TP.stratified_auc(y, s, P.stratum.to_numpy()):.3f}",
               "0,76 (DeepEpilepsy) в ДИ": "да" if lo <= 0.76 <= hi else ("ниже 0,76" if hi < 0.76 else "выше 0,76")}
        mm = (P.montage == "01_tcp_ar").to_numpy()
        if mm.sum() and len(set(y[mm])) == 2:
            row["только 01_tcp_ar"] = f"{roc_auc_score(y[mm], s[mm]):.3f} (n={int(mm.sum())})"
        if zs is not None:
            Z = P.merge(zs[["patient"] + [c for c in ("reve4", "dino2", "ens") if c in zs]], on="patient", how="inner")
            ref = {"reve": "reve4", "dino": "dino2"}.get(m, "ens")
            if ref in Z and len(Z):
                dv, dl, dh = paired_diff(Z.y.to_numpy(), Z.score.to_numpy(), Z[ref].to_numpy())
                row[f"Δ к прогону без дообучения ({ref})"] = f"{dv:+.3f} ({dl:+.3f}…{dh:+.3f}), без дообучения {roc_auc_score(Z.y, Z[ref]):.3f}, n={len(Z)}"
        rows.append(row)
        # пороги: по отложенной части своего фолда, применяются к тесту этого фолда; метрики — по всем пациентам вместе
        for rn, f in TP.RULES.items():
            pred = np.full(len(P), -1)
            for k in folds_done:
                V = patient_scores(va[va.fold == k], chosen)
                if V.y.nunique() < 2:
                    continue
                pred[(P.fold == k).to_numpy()] = (P.score[P.fold == k].to_numpy() >= f(V.y.to_numpy(), V.score.to_numpy())).astype(int)
            ok = pred >= 0
            if ok.any():
                P[f"pred_{'youden' if rn == 'Юден' else 'sens80'}"] = pred
                mt = boot_metr(y[ok], pred[ok])
                thr_rows.append({"модель": NAMES.get(m, m), "порог": rn + " (отложенная часть фолда)", "n": int(ok.sum()), "accuracy": mt["acc"], "сбаланс. accuracy": mt["bacc"],
                                 "precision": mt["prec"], "recall (чувств.)": mt["rec"], "специфичность": mt["spec"]})
        # устойчивость к выбору сеанса
        St = TP.session_table(te.merge(meta[["id", "n_seiz", "year", "montage", "sfreq_orig"]], on="id"), ["score"])
        St["has_seiz"] = [bool(info.has_seiz.get((p, s_), False)) for p, s_ in zip(St.patient, St.session)]
        for rule in ("first", "last", "all"):
            Pr, _ = TP.select_sessions(St, ["score"], rule, 0)
            sens_rows.append({"модель": NAMES.get(m, m), "сеанс": rule, "n": len(Pr), "ROC AUC": round(roc_auc_score(Pr.y, Pr.score), 3)})
        sens_rows.append({"модель": NAMES.get(m, m), "сеанс": "random (основной)", "n": len(P), "ROC AUC": round(auc, 3)})
        if P_all is None:
            P_all = pd.DataFrame({"patient": sorted(chosen)})
            P_all["session"] = P_all.patient.map(chosen)
            P_all = P_all.merge(F, on="patient").join(info[["montage", "sfreq"]], on=["patient", "session"])
        P_all = P_all.merge(P[["patient", "score"] + [c for c in P if c.startswith("pred_")]].rename(
            columns=lambda c: c if c == "patient" else f"{m}_{c}"), on="patient", how="outer")
    T, TH, SN = pd.DataFrame(rows), pd.DataFrame(thr_rows), pd.DataFrame(sens_rows)
    P_all.to_csv(os.path.join(a.out, f"tuep_ft_patient_scores{sfx}.csv"), index=False)    # с кодами пациентов — только на рабочей машине
    T.to_csv(os.path.join(a.out, f"tuep_ft_metrics{sfx}.csv"), index=False); TH.to_csv(os.path.join(a.out, f"tuep_ft_threshold_metrics{sfx}.csv"), index=False)
    SN.to_csv(os.path.join(a.out, f"tuep_ft_session_rule_sensitivity{sfx}.csv"), index=False)
    hist = [json.load(open(p, encoding="utf-8")) for p in sorted(glob.glob(os.path.join(a.out, "runs", "*.json")))]
    H = pd.DataFrame([{k: h.get(k) for k in ("model", "fold", "seed", "best_epoch", "best_val_auc", "minutes", "n_bad")} for h in hist])
    cfg = json.load(open(os.path.join(a.out, "config.json"), encoding="utf-8")) if os.path.exists(os.path.join(a.out, "config.json")) else {}
    md = ["# TUEP: дообучение по 5 фолдам — сводка", "",
          f"Пациентов с оценкой: {len(P_all)} (выпали — все сеансы с приступами: {len(dropped)}). Один случайный сеанс без приступов на пациента (seed 0). "
          f"Настройки: {cfg}.{note} Ориентир DeepEpilepsy — 0,76; готовые модели без дообучения — 0,52 (results/tuep_run_2026-11-14.md).", "",
          "## ROC AUC по пациентам (тест всех фолдов вместе; 95 % ДИ — бутстреп по пациентам)", "", T.drop(columns="код").to_markdown(index=False), "",
          "## Пороговые метрики", "", TH.to_markdown(index=False) if len(TH) else "нет", "",
          "## Устойчивость к выбору сеанса", "", SN.pivot(index="модель", columns="сеанс", values="ROC AUC").to_markdown(), ""]
    if len(H):
        md += ["## Запуски", "", H.groupby("model").agg(запусков=("seed", "size"), лучшая_эпоха_медиана=("best_epoch", "median"),
                                                         отложенные_AUC_среднее=("best_val_auc", "mean"), минут_всего=("minutes", "sum")).round(3).to_markdown(), ""]
    open(os.path.join(a.out, f"tuep_ft_summary{sfx}.md"), "w", encoding="utf-8").write("\n".join(md))
    print("\n".join(md)); print("→", a.out, "(в git — только tuep_ft_summary.md и tuep_ft_metrics.csv: в них нет кодов пациентов)")


def main():
    ap = argparse.ArgumentParser(); sp = ap.add_subparsers(dest="cmd", required=True)
    t = sp.add_parser("train")
    t.add_argument("--model", required=True, choices=["reve", "dino", "de", "toy"])
    t.add_argument("--prep", required=True); t.add_argument("--out", required=True)
    t.add_argument("--proto-prep", default="", help="основная подготовка: по ней считаются сеансы, слои, фолды и выбранный сеанс, если в --prep часть файлов отсутствует (prep_ears)")
    t.add_argument("--name", default="", help="код запусков вместо имени модели (например dea), чтобы не смешивать с прежними")
    t.add_argument("--reve-dir", default=""); t.add_argument("--dino-dir", default=""); t.add_argument("--dino-src", default="")
    t.add_argument("--de-repo", default="~/data/weights/dl_epilepsy_reeg", help="клон gitlab chum-epilepsy/dl_epilepsy_reeg (код модели)")
    t.add_argument("--de-weights", default="", help="DeepEpilepsy_large_50_30s_augment.pt; пусто — из кэша HF emilelemoine/DeepEpilepsy_v1")
    t.add_argument("--de-toy", action="store_true", help=argparse.SUPPRESS)
    t.add_argument("--folds", default="", help="tuep_folds.csv; пусто — те же фолды считаются заново (seed 0)")
    t.add_argument("--fold-list", type=int, nargs="*", default=list(range(N_FOLDS))); t.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    t.add_argument("--init", default="nmt", choices=["nmt", "base"], help="nmt — веса после NMT (reve_v4_seed*.pt / eegdino_v2_seed*.pt); base — исходная основа, голова случайная; для de не действует: всегда веса авторов")
    t.add_argument("--freeze-backbone", action="store_true", help="учить только голову")
    t.add_argument("--balance", default="strata", choices=["strata", "class"]); t.add_argument("--min-pat", type=int, default=3)
    t.add_argument("--lowpass", type=float, default=45.0, help="ФНЧ, Гц; 0 — без него")
    t.add_argument("--exclude-sessions", default="", help="csv patient,session: считать сеансами с приступами дополнительно")
    t.add_argument("--max-epochs", type=int, default=8); t.add_argument("--patience", type=int, default=3)
    t.add_argument("--epoch-windows", type=int, default=4000); t.add_argument("--bs", type=int, default=32); t.add_argument("--bs-eval", type=int, default=64)
    t.add_argument("--n-win", type=int, default=CAP_WIN, help="окон по 10 с на файл при оценке; для окна 30 с берётся столько же секунд (20 окон)"); t.add_argument("--lr-scale", type=float, default=1.0)
    t.add_argument("--workers", type=int, default=6); t.add_argument("--device", default="cuda:0")
    t.add_argument("--smoke", action="store_true"); t.add_argument("--save-weights", action="store_true"); t.add_argument("--no-cache", action="store_true")
    t.add_argument("--debug-stop-epoch", type=int, default=0, help=argparse.SUPPRESS)
    r = sp.add_parser("report")
    r.add_argument("--prep", required=True); r.add_argument("--out", required=True); r.add_argument("--folds", default="")
    r.add_argument("--exclude-sessions", default=""); r.add_argument("--zero-shot", default="", help="tuep_patient_scores.csv прогона без дообучения")
    r.add_argument("--ensemble", nargs="*", action="append", help="пара моделей (коды); можно повторять: --ensemble reve bpampw --ensemble de bpampw")
    r.add_argument("--extra-runs", nargs="*", default=[], help="папки runs других прогонов (например ~/data/tuep_work/ft/runs): их модели добавляются в таблицу")
    r.add_argument("--same-patients", default="", help="код модели: все строки считаются только на пациентах, у которых есть её оценка; файлы сводки получают суффикс _same_<код>")
    a = ap.parse_args()
    {"train": cmd_train, "report": cmd_report}[a.cmd](a)


if __name__ == "__main__":
    main()
