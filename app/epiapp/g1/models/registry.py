"""Список моделей Г1. Новая модель = класс по контракту base.G1Model + строка здесь."""
from .demo import DemoModel
from .reve_dino import ReveDino
from .tuep_pair import TuepPair

MODELS = {m.id: m for m in (DemoModel, ReveDino, TuepPair)}
_CACHE = {}


def get_model(settings, model_id=None):
    mid = model_id or settings.g1_model
    assert mid in MODELS, f"неизвестная модель Г1: {mid} (есть: {', '.join(MODELS)})"
    if mid not in _CACHE:
        _CACHE[mid] = MODELS[mid](settings)
    return _CACHE[mid]


def list_models(settings):
    out = []
    for mid in MODELS:
        m = get_model(settings, mid)
        ok, why = m.available()
        out.append(dict(id=mid, title=m.title, available=ok, reason=why, demo=bool(m.card.get("demo")), active=mid == settings.g1_model))
    return out
