"""Контракт моделей Г1: любую модель можно подключить без изменения интерфейса."""
import numpy as np

from epiapp.g1 import prepare as P
from epiapp.g1.models import registry
from epiapp.g1.models.base import G1Model, G1Output
from conftest import wait_job

CARD_KEYS = {"task", "trained_on", "metrics", "default_threshold", "default_threshold_source", "verdict_above", "verdict_below", "limits", "stages", "eta", "demo"}


def test_cards_complete(settings):
    for mid in registry.MODELS:
        m = registry.MODELS[mid](settings)
        assert CARD_KEYS <= set(m.card), (mid, CARD_KEYS - set(m.card))
        assert 0 <= m.card["default_threshold"] <= 1 and m.card["stages"]
        for x in m.card["metrics"]:
            assert x["source"], "у каждой цифры должен быть файл-источник"
        ok, why = m.available()
        assert ok or why


def test_real_models_report_missing_weights(settings):
    for mid in ("reve_dino", "tuep_pair"):
        ok, why = registry.MODELS[mid](settings).available()
        assert not ok and "нет файлов" in why


def test_demo_model_output(settings, edf):
    rec, _ = P.prepare_edf(str(edf(seconds=300)))
    seen = []
    out = registry.MODELS["demo"](settings).predict(rec, lambda s, f: seen.append((s, f)))
    assert isinstance(out, G1Output) and 0 <= out.score <= 1 and out.n_windows == 30
    assert out.parts and all(0 <= p["score"] <= 1 and p["t1"] > p["t0"] for p in out.parts) and out.parts[-1]["t1"] <= rec.dur_s + 1e-6
    assert seen and all(0 <= f <= 1 for _, f in seen)


def test_new_model_plugs_in(svc, monkeypatch):
    """Новая модель = класс по контракту + строка в реестре. Сервис и интерфейс не меняются."""
    class Const(G1Model):
        id, title = "const", "Постоянная"
        card = dict(registry.MODELS["demo"].card, default_threshold=0.3, verdict_above="выше", verdict_below="ниже", stages=["один этап"])
        def predict(self, rec, progress=lambda s, f: None):
            assert rec.x.shape[0] == 19 and rec.x.dtype == np.float32
            return G1Output(score=0.4, parts=[dict(t0=0, t1=rec.dur_s, score=0.4)], n_windows=1)
    monkeypatch.setitem(registry.MODELS, "const", Const)
    monkeypatch.setattr(svc.s, "g1_model", "const")
    v = svc.g1_demo("bad"); assert wait_job(svc, svc.g1_run(v["id"]))["status"] == "done"
    r = svc.g1_view(v["id"])["result"]
    assert r["score"] == 0.4 and r["above"] and r["verdict"] == "выше" and r["model_title"] == "Постоянная"
    assert svc.meta()["g1"]["model"]["id"] == "const"
