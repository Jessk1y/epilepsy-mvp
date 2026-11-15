"""Вариант (а): готовые REVE v4 + EEG-DINO v2 (дообучены на NMT, «патология / норма»), по 3 запуска каждой.

Путь расчёта — как h1_eeg/tuep/tuep_infer.py: ФВЧ 0,3 Гц по записи -> окна 10 с подряд -> 60 окон равномерно -> вероятности окон ->
оценка записи = среднее верхних 20 %; запуски усредняются; пара = среднее двух моделей.
Веса: папки EPI_REVE_DIR и EPI_DINO_DIR (reve_v4_seed{0,1,2}.pt + reve_v4_model.py; eegdino_v2_seed{0,1,2}.pt + eegdino_v2_model.py).
"""
import os, sys
import numpy as np

import record_scoring as RS
from .base import G1Model, G1Output

SEEDS = (0, 1, 2)
SRC = "h1_eeg/SUMMARY.md"


class ReveDino(G1Model):
    id, title = "reve_dino", "REVE v4 + EEG-DINO v2"
    card = dict(
        task="ненормальная ЭЭГ против нормальной (дообучение на NMT: «патология / норма»)",
        trained_on="NMT (Пакистан), 2 225 записей; основы REVE и EEG-DINO предобучены авторами",
        metrics=[dict(data="NMT eval, 185 человек (та же больница)", value="ROC AUC 0,920", source=SRC),
                 dict(data="Турция, 121 человек (эпилепсия / здоровые)", value="ROC AUC 0,936", source=SRC),
                 dict(data="EpiEEG-60, Бангладеш, 63 человека (эпилепсия / здоровые)", value="ROC AUC 0,756", source=SRC),
                 dict(data="TUEP, 199 пациентов (эпилепсия / пациенты без эпилепсии)", value="ROC AUC 0,523 (0,44–0,60) — уровень случайного",
                      source="h1_eeg/results/tuep_improve_2026-11-15.md")],
        default_threshold=0.755, default_threshold_source="индекс Юдена на отложенной части NMT (открытые данные), h1_eeg/results/tuep_improve_2026-11-15.md",
        verdict_above="Признаки, характерные для эпилепсии", verdict_below="Признаков эпилепсии не выявлено",
        limits=["Модель отличает ненормальную ЭЭГ от нормальной. Эпилепсию от других болезней с изменённой ЭЭГ она не отличает (TUEP: 0,52).",
                "Порог с открытых данных в другую клинику не переносится — нужна настройка под клинику."],
        stages=["Чтение и фильтрация сигнала", "Нарезка на окна", "Модель REVE v4", "Модель EEG-DINO v2", "Сведение оценок по записи"],
        eta="Около 1,5 минуты на процессоре, секунды на видеокарте.", demo=False)

    def available(self):
        need = [self.s.reve_dir / f"reve_v4_seed{s}.pt" for s in SEEDS] + [self.s.reve_dir / "reve_v4_model.py"]
        need += [self.s.dino_dir / f"eegdino_v2_seed{s}.pt" for s in SEEDS] + [self.s.dino_dir / "eegdino_v2_model.py"]
        miss = [str(p) for p in need if not p.exists()]
        if miss:
            return False, "нет файлов: " + "; ".join(miss[:3]) + (" …" if len(miss) > 3 else "")
        try:
            import torch, transformers  # noqa: F401
        except Exception as e:
            return False, f"нет библиотеки: {e}"
        return True, ""

    def load(self):
        import torch
        self.torch = torch
        dev = self.s.device if torch.cuda.is_available() else "cpu"
        self.dev = dev
        for d in (self.s.reve_dir, self.s.dino_dir):
            if str(d) not in sys.path:
                sys.path.insert(0, str(d))
        from reve_v4_model import RVNet, rv_input, positions
        from eegdino_v2_model import DinoNet, dino_input
        pos = positions(dev)
        sd = lambda p: torch.load(p, map_location="cpu", weights_only=False)["state_dict"]
        self.fam = {"reve4": ([], rv_input), "dino2": ([], dino_input)}
        for s in SEEDS:
            m = RVNet(pos); m.load_state_dict(sd(os.path.join(self.s.reve_dir, f"reve_v4_seed{s}.pt"))); self.fam["reve4"][0].append(m.to(dev).eval())
        for s in SEEDS:
            m = DinoNet(src_dir=self.s.dino_src or None); m.load_state_dict(sd(os.path.join(self.s.dino_dir, f"eegdino_v2_seed{s}.pt")))
            self.fam["dino2"][0].append(m.to(dev).eval())
        self.loaded = True

    def _probs(self, model, prep, W, bs=30):
        torch, out = self.torch, []
        cuda = str(self.dev).startswith("cuda")
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16, enabled=cuda):
            for b in range(0, len(W), bs):
                out.append(torch.sigmoid(model(prep(W[b:b + bs]).to(self.dev)).float()).cpu().numpy())
        return np.concatenate(out)

    def predict(self, rec, progress=lambda stage, frac: None):
        t = self.timer()
        progress(0, 0.05)
        W, starts = RS.whole_record_windows(rec.x)
        progress(1, 0.1)
        P, done = {}, 0
        for fi, (fam, (models, prep)) in enumerate(self.fam.items()):
            P[fam] = []
            for m in models:
                P[fam].append(self._probs(m, prep, W)); done += 1
                progress(2 + fi, 0.1 + 0.8 * done / 6)
        progress(4, 0.95)
        score_of = lambda idx: float(np.mean([np.mean([RS.top20(p[idx]) for p in P[fam]]) for fam in P]))
        allw = np.arange(len(W))
        comp = {fam: float(np.mean([RS.top20(p) for p in P[fam]])) for fam in P}
        parts = [dict(t0=a, t1=b, score=score_of(idx)) for a, b, idx in self.part_index(starts, rec.x.shape[1])]
        return G1Output(score=score_of(allw), parts=parts, components={"REVE v4": comp["reve4"], "EEG-DINO v2": comp["dino2"]},
                        n_windows=len(W), seconds=t())
