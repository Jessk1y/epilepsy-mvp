"""Общее для тестов: приложение в демо-режиме с рабочим каталогом во временной папке. Запуск: py -m pytest app/tests -q"""
import os, sys, time
import numpy as np, pytest

APP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, APP)
import epiapp.g1  # noqa: E402,F401  (подключает h1_eeg/tuep и h1_eeg/src)
from epiapp.config import load_settings  # noqa: E402
from epiapp.g1 import synth  # noqa: E402


@pytest.fixture()
def settings(tmp_path, monkeypatch):
    monkeypatch.setenv("EPI_VAR_DIR", str(tmp_path / "var"))
    for k in ("EPI_G1_MODEL", "EPI_G2_CMD", "EPI_G2_ADAPTER", "EPI_MODE"):
        monkeypatch.delenv(k, raising=False)
    return load_settings("demo")


@pytest.fixture()
def svc(settings):
    from epiapp.service import Service
    return Service(settings)


@pytest.fixture()
def edf(tmp_path):
    """Короткая синтетическая запись: 150 с, 250 Гц, имена каналов как у клинических приборов."""
    def make(seconds=150, fs=250, flat=(), drop=(), art_frac=0.05, seed=1, name="rec.edf", theta=0.3):
        X, names, arts = synth.make_eeg(seconds, fs, seed, theta, flat, art_frac)
        keep = [i for i, n in enumerate(names) if n not in drop]
        labels = [("EEG " + names[i] + "-Ref") if names[i] != "EKG1" else "EKG1" for i in keep]
        p = tmp_path / name
        synth.write_edf(str(p), labels, fs, X[keep])
        return p
    return make


def wait_job(svc, job, timeout=120):
    t0 = time.time()
    while time.time() - t0 < timeout:
        j = svc.job(job["id"])
        if j["status"] in ("done", "error"):
            return j
        time.sleep(0.1)
    raise TimeoutError(job)
