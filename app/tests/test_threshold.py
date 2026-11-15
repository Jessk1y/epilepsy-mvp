"""Порог клиники: совпадение с правилами экспериментов (tuep_protocol), хранение и применение."""
import numpy as np, pytest

import tuep_protocol as TP
from epiapp.g1 import threshold as T
from conftest import wait_job


@pytest.mark.parametrize("seed", range(8))
def test_same_as_experiments(seed):
    r = np.random.default_rng(seed)
    y = np.r_[np.ones(10, int), np.zeros(10, int)]
    s = np.round(np.r_[r.normal(0.65, 0.18, 10), r.normal(0.4, 0.18, 10)].clip(0, 1), 2 if seed % 2 else 6)   # с совпадающими оценками и без
    assert T.thr_youden(y, s) == pytest.approx(TP.thr_youden(y, s))
    assert T.thr_sens80(y, s) == pytest.approx(TP.thr_recall80(y, s))
    c = T.confusion(y, s, T.thr_sens80(y, s))
    assert c["sensitivity"] >= 0.8 and c["tp"] + c["fn"] == 10 and c["tn"] + c["fp"] == 10
    m = TP.metr(y, (s >= 0.5).astype(int)); c = T.confusion(y, s, 0.5)
    assert c["sensitivity"] == pytest.approx(m["rec"]) and c["specificity"] == pytest.approx(m["spec"]) and c["balanced_accuracy"] == pytest.approx(m["bacc"])


def test_auc_and_median():
    from sklearn.metrics import roc_auc_score
    r = np.random.default_rng(0); y = r.integers(0, 2, 60); s = np.round(r.random(60), 1)
    assert T.auc(y, s) == pytest.approx(roc_auc_score(y, s))
    assert T.thr_median([0.1, 0.5, 0.9, 0.7]) == pytest.approx(0.6)


def test_propose_warnings():
    p = T.propose([0.2, 0.8, 0.6], [0, 1, None])
    assert p["labeled"]["n"] == 2 and p["unlabeled"]["n"] == 3 and len(p["warnings"]) == 3
    assert T.propose([0.2, 0.3], [1, 1])["labeled"] is None


def test_threshold_is_stored_and_applied(svc):
    v = svc.g1_demo("ok"); assert wait_job(svc, svc.g1_run(v["id"]))["status"] == "done"
    r = svc.g1_view(v["id"])["result"]
    assert not r["threshold"]["clinic"] and r["threshold"]["value"] == svc.model().card["default_threshold"]
    svc.clinic_demo_fill()
    t = svc.clinic_set_threshold("youden")
    assert t["clinic"] and t["method"] == "youden" and "20 размеченным" in t["source"]
    r2 = svc.g1_view(v["id"])["result"]                      # порог применяется к уже оценённой записи
    assert r2["threshold"]["value"] == t["value"] and r2["above"] == (r2["score"] >= t["value"])
    m = svc.clinic_set_threshold("median")
    assert m["method"] == "median" and svc.threshold_info()["value"] == m["value"] and "без разметки" in m["source"]
    svc.clinic_set_threshold("manual", 0.99)
    assert svc.g1_view(v["id"])["result"]["above"] is False
    from epiapp.service import Service                       # порог переживает перезапуск
    assert Service(svc.s).threshold_info()["value"] == 0.99
    svc.clinic_reset()
    assert not svc.threshold_info()["clinic"]


def test_threshold_is_per_model(svc):
    svc.clinic_demo_fill(); svc.clinic_set_threshold("youden")
    assert svc.store.threshold("demo", True) is not None and svc.store.threshold("reve_dino", True) is None


def test_scored_study_becomes_clinic_record(svc):
    v = svc.g1_demo("bad"); wait_job(svc, svc.g1_run(v["id"]))
    recs = svc.clinic()["records"]
    assert len(recs) == 1 and recs[0]["source"] == "study" and recs[0]["label"] is None and recs[0]["low_quality"] == 1
    svc.clinic_label(recs[0]["id"], 1)
    assert svc.clinic()["records"][0]["label"] == 1
