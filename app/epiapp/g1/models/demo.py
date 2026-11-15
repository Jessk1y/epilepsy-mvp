"""Демо-модель: настоящих весов нет, оценка считается из простого свойства сигнала (доля медленной активности 4–8 Гц
относительно альфа-ритма). Клинического смысла НЕ имеет — нужна, чтобы показать весь путь без видеокарты."""
import time
import numpy as np, scipy.signal as ss

import record_scoring as RS
from .base import G1Model, G1Output

K, B = 2.6, 0.19      # подобраны так, чтобы синтетические демо-записи давали оценки как в макете (0,81 и 0,66)


class DemoModel(G1Model):
    id, title = "demo", "Демо-модель (без нейросети)"
    card = dict(
        task="демонстрация пути врача; оценка не имеет клинического смысла",
        trained_on="не обучалась",
        metrics=[],
        default_threshold=0.5, default_threshold_source="условный порог демо-модели",
        verdict_above="Признаки, характерные для эпилепсии", verdict_below="Признаков эпилепсии не выявлено",
        limits=["ДЕМО: оценка считается по простой формуле и не говорит о диагнозе."],
        stages=["Чтение и фильтрация сигнала", "Нарезка на окна", "Демо-модель: оценка окон", "Сведение оценок по записи"],
        eta="Несколько секунд.", demo=True)

    def predict(self, rec, progress=lambda stage, frac: None):
        t = self.timer()
        progress(0, 0.1)
        W, starts = RS.whole_record_windows(rec.x)
        progress(1, 0.3)
        good = [i for i, c in enumerate(rec.channels) if c not in rec.flat]
        f, P = ss.welch(W[:, good].astype(np.float64), 200, nperseg=400, axis=-1)
        th, al = P[..., (f >= 4) & (f < 8)].sum(-1).mean(-1), P[..., (f >= 8) & (f < 13)].sum(-1).mean(-1)
        p = 1 / (1 + np.exp(-(K * np.log10(th / (al + 1e-9) + 1e-9) + B)))
        for k in range(3):                       # в демо расчёт слишком быстрый — показываем этапы
            progress(2, 0.4 + 0.15 * k); time.sleep(0.35)
        progress(3, 0.95)
        parts = [dict(t0=a, t1=b, score=float(p[idx].mean())) for a, b, idx in self.part_index(starts, rec.x.shape[1])]
        return G1Output(score=float(p.mean()), parts=parts, components={"демо": float(p.mean())}, n_windows=len(W), seconds=t())
