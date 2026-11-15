"""Вариант (б): пара «REVE v4, дообученная на TUEP» + «эталон: мощности ритмов, амплитуда, wPLI + логистическая регрессия».

Лучший результат на задаче «эпилепсия против других пациентов» (TUEP, 5 фолдов по пациентам).
Файлы в папке EPI_TUEP_FINAL_DIR — их создают команды из h1_eeg/tuep/README.md, раздел «Итоговые модели для приложения»:
  reve_tuep_seed{0,1,2}.pt   tuep_finetune.py final
  baseline_bpampw.npz        tuep_baseline.py --save-model
  pair_card.json             tuep_finetune.py card  (пороги и ROC AUC по оценкам вне обучения)
Нужна также папка EPI_REVE_DIR с reve_v4_model.py (класс модели и нормировка окна).
Путь расчёта — как при проверке по фолдам: окна 10 с (60 равномерно), фильтр по куску «окно ± 5 с»; REVE — ФВЧ 0,3 + ФНЧ 45 Гц,
оценка — среднее верхних 20 % окон, среднее по запускам; эталон — ФВЧ 0,3 Гц; пара — среднее двух оценок.
"""
import glob, json, os, sys
import numpy as np

import record_scoring as RS
from .base import G1Model, G1Output

SRC = "h1_eeg/results/tuep_improve_2026-11-15.md"


class TuepPair(G1Model):
    id, title = "tuep_pair", "REVE v4 (TUEP) + эталон с wPLI"
    card = dict(
        task="эпилепсия против других пациентов (TUEP)",
        trained_on="TUEP (TUH, США): 199 пациентов; REVE v4 до этого дообучена на NMT",
        metrics=[dict(data="TUEP, 199 пациентов, 5 фолдов по пациентам", value="ROC AUC 0,661 (0,59–0,74); внутри одной аппаратуры 0,633", source=SRC),
                 dict(data="TUEP, порог Юдена по отложенной части фолда", value="чувствительность 0,52, специфичность 0,69, сбаланс. accuracy 0,60", source=SRC)],
        default_threshold=0.5, default_threshold_source="будет взят из pair_card.json (индекс Юдена по оценкам вне обучения, TUEP)",
        verdict_above="Признаки, характерные для эпилепсии", verdict_below="Признаков эпилепсии не выявлено",
        limits=["Качество умеренное: ROC AUC 0,66 на TUEP, ориентир из литературы 0,76 не достигнут.",
                "Обучена на записях одной сети больниц США; в другой клинике нужна настройка порога."],
        stages=["Чтение и фильтрация сигнала", "Нарезка на окна", "Модель REVE v4 (TUEP)", "Эталон: мощности ритмов и wPLI", "Сведение оценок по записи"],
        eta="Около 1 минуты: связность wPLI считается на процессоре.", demo=False)

    def __init__(self, settings):
        super().__init__(settings)
        self.card = dict(self.card)
        p = self.s.tuep_final_dir / "pair_card.json"
        if p.exists():                                   # пороги и цифры — из паспорта, сохранённого при обучении
            c = json.load(open(p, encoding="utf-8"))
            if "youden" in c.get("thresholds", {}):
                self.card["default_threshold"] = float(c["thresholds"]["youden"])
                self.card["default_threshold_source"] = f"индекс Юдена по оценкам вне обучения, TUEP, {c.get('n', '?')} пациентов (pair_card.json от {c.get('date', '?')})"
            self.card["pair_card"] = c

    def _weights(self):
        return sorted(glob.glob(str(self.s.tuep_final_dir / "reve_tuep_seed*.pt")))

    def available(self):
        miss = []
        if not self._weights():
            miss.append(str(self.s.tuep_final_dir / "reve_tuep_seed*.pt"))
        for p in (self.s.tuep_final_dir / "baseline_bpampw.npz", self.s.reve_dir / "reve_v4_model.py"):
            if not p.exists():
                miss.append(str(p))
        if miss:
            return False, "нет файлов: " + "; ".join(miss) + " — см. h1_eeg/tuep/README.md, «Итоговые модели для приложения»"
        try:
            import torch, transformers  # noqa: F401
        except Exception as e:
            return False, f"нет библиотеки: {e}"
        return True, ""

    def load(self):
        import torch
        self.torch = torch
        self.dev = self.s.device if torch.cuda.is_available() else "cpu"
        if str(self.s.reve_dir) not in sys.path:
            sys.path.insert(0, str(self.s.reve_dir))
        from reve_v4_model import RVNet, rv_input, positions
        self.rv_input, pos, self.models, lows = rv_input, positions(self.dev), [], set()
        for p in self._weights():
            ck = torch.load(p, map_location="cpu", weights_only=False)
            m = RVNet(pos); m.load_state_dict(ck["state_dict"]); self.models.append(m.to(self.dev).eval())
            lows.add(float(ck.get("config", {}).get("lowpass", 45.0)))
        assert len(lows) == 1, f"у запусков разный ФНЧ: {lows}"
        self.sos_reve, self.sos_base = RS.make_sos(lows.pop()), RS.make_sos(0)
        self.lr, self.bsos = RS.LogReg(self.s.tuep_final_dir / "baseline_bpampw.npz"), RS.band_sos()
        self.loaded = True

    def _probs(self, model, W, bs=64):
        torch, out = self.torch, []
        cuda = str(self.dev).startswith("cuda")
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16, enabled=cuda):
            for b in range(0, len(W), bs):
                out.append(torch.sigmoid(model(self.rv_input(W[b:b + bs]).to(self.dev)).float()).cpu().numpy())
        return np.concatenate(out)

    def predict(self, rec, progress=lambda stage, frac: None):
        t = self.timer()
        progress(0, 0.05)
        W, starts = RS.eval_windows_array(rec.x, self.sos_reve)
        W0, _ = RS.eval_windows_array(rec.x, self.sos_base)
        progress(1, 0.15)
        P = []
        for k, m in enumerate(self.models):
            P.append(self._probs(m, W)); progress(2, 0.15 + 0.35 * (k + 1) / len(self.models))
        MB = RS.band_amp_matrix(W0); progress(3, 0.6)
        MW = RS.wpli_matrix(W0, self.bsos); progress(3, 0.9)
        reve = lambda idx: float(np.mean([RS.top20(p[idx]) for p in P]))
        base = lambda idx: self.lr.predict_proba(RS.baseline_vector(MB[idx], MW[idx]))
        allw = np.arange(len(W))
        r, b = reve(allw), base(allw)
        progress(4, 0.95)
        parts = [dict(t0=a, t1=c, score=(reve(idx) + base(idx)) / 2) for a, c, idx in self.part_index(starts, rec.x.shape[1])]
        return G1Output(score=(r + b) / 2, parts=parts, components={"REVE v4 (TUEP)": r, "эталон с wPLI": b}, n_windows=len(W), seconds=t())
