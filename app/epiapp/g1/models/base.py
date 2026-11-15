"""Контракт модели Г1. Интерфейс приложения работает только с этим контрактом: модель можно заменить без переделки экранов.

Чтобы добавить модель: класс-наследник G1Model в этой папке + строка в registry.py.
"""
import time
from dataclasses import dataclass, field
import numpy as np

N_PARTS = 6          # на сколько частей делится запись для показа устойчивости оценки


@dataclass
class G1Output:
    score: float                                  # оценка всей записи, 0..1
    parts: list = field(default_factory=list)     # [{"t0": с, "t1": с, "score": …}] — оценки по частям записи
    components: dict = field(default_factory=dict)  # оценки составных частей модели (для справки)
    n_windows: int = 0
    seconds: float = 0.0


class G1Model:
    id = "base"
    title = "—"
    # Паспорт модели: всё, что интерфейс показывает о модели. Цифры — только из файлов результатов, с указанием файла.
    card = dict(
        task="",                # какую задачу решала модель при обучении
        trained_on="",
        metrics=[],             # [{"data": …, "value": …, "source": файл}]
        default_threshold=0.5, default_threshold_source="",
        verdict_above="", verdict_below="",
        limits=[],              # ограничения — строки
        stages=[],              # этапы расчёта для экрана «Идёт оценка»
        eta="",                 # сколько обычно длится
        demo=False)

    def __init__(self, settings):
        self.s = settings
        self.loaded = False

    def available(self):
        """(можно ли запустить, причина если нет)."""
        return True, ""

    def load(self):
        self.loaded = True

    def predict(self, rec, progress=lambda stage, frac: None) -> G1Output:
        raise NotImplementedError

    # ---- общее для реализаций
    @staticmethod
    def part_index(starts, n_total, n_parts=N_PARTS):
        """Окна -> части записи по времени: список (t0, t1, номера окон). Частей меньше, если окон мало (не меньше 5 окон на часть)."""
        k = len(starts)
        n_parts = max(1, min(n_parts, k // 5))
        edges = np.linspace(0, n_total, n_parts + 1)
        out = []
        for a, b in zip(edges[:-1], edges[1:]):
            idx = np.where((starts >= a) & (starts < b))[0]
            if len(idx):
                out.append((float(a) / 200, float(b) / 200, idx))
        return out

    @staticmethod
    def timer():
        t0 = time.time()
        return lambda: round(time.time() - t0, 1)
