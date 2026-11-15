"""Г2: контракт «команда + файлы», проверка входа, кандидаты, набор для просмотра."""
import sys, textwrap
import numpy as np, pytest

from epiapp.g2 import candidates as CAND, contract as G2C, volume as VOL
from conftest import wait_job

nib = pytest.importorskip("nibabel")


def test_check_input_flair_5mm():
    ok = VOL.check_input(dict(zooms=[1, 1, 1], ndim=3), dict(zooms=[1, 1, 1], ndim=3))
    bad = VOL.check_input(dict(zooms=[1, 1, 1], ndim=3), dict(zooms=[0.9, 0.9, 5.0], ndim=3))
    assert ok["can_run"] and not ok["low_quality"] and not ok["flair5"]
    assert bad["can_run"] and bad["flair5"] and bad["low_quality"] and bad["items"][1]["status"] == "bad"


def test_cluster_table():
    prob = np.zeros((40, 40, 40), np.float32); prob[5:11, 5:11, 5:11] = 0.95; prob[30:34, 30:34, 30:34] = 0.6; prob[20, 20, 20] = 0.99
    aff = np.diag([1.0, 1.0, 1.0, 1.0]); aff[:3, 3] = [-20, -20, -20]
    c = CAND.cluster_table(prob, 0.5, aff)
    assert len(c) == 1 and round(c[0]["volume_ml"], 3) == 0.216                             # одиночный воксель и малая область отброшены
    c = CAND.cluster_table(prob, 0.5, aff, min_ml=0.05)
    assert len(c) == 2 and c[0]["peak_prob"] > c[1]["peak_prob"] and c[0]["hemi"] == "левое" and c[1]["hemi"] == "правое" and [x["n"] for x in c] == [1, 2]
    assert CAND.cluster_table(prob, 0.9, aff, min_ml=0.05)[0]["xyz"][0] < 0 and CAND.cluster_table(prob, 0.999, aff) == []


CLI = '''
import argparse, json, numpy as np, nibabel as nib
ap = argparse.ArgumentParser(); ap.add_argument("--t1"); ap.add_argument("--flair"); ap.add_argument("--out"); a = ap.parse_args()
t1 = nib.load(a.t1); prob = np.zeros(t1.shape, np.float32); prob[10:18, 20:28, 12:20] = 0.93
nib.save(nib.Nifti1Image(prob, t1.affine), a.out + "/prob.nii.gz")
x, y, z = (t1.affine @ [13, 23, 15, 1])[:3]
open(a.out + "/candidates.csv", "w").write("peak_prob,volume_ml,x,y,z\\n0.93,0.9,%f,%f,%f\\n" % (x, y, z))
json.dump(dict(t1_mm=1.0, flair_mm=3.0, mni_corr=0.6, flags=["проверка"], ok=True), open(a.out + "/qc.json", "w"))
'''


def test_external_command_contract(tmp_path, monkeypatch):
    """Любая команда, выполняющая контракт, подключается настройкой EPI_G2_CMD. Снимки — в ориентации LAS и с другой сеткой FLAIR."""
    r = np.random.default_rng(0)
    aff = np.diag([-1.0, 1.0, 1.0, 1.0]); aff[:3, 3] = [20, -25, -15]                    # LAS: приложение приведёт к RAS
    nib.save(nib.Nifti1Image(r.random((40, 50, 30)).astype(np.float32), aff), tmp_path / "t1.nii.gz")
    aff_f = np.diag([-1.0, 1.0, 3.0, 1.0]); aff_f[:3, 3] = [20, -25, -15]
    nib.save(nib.Nifti1Image(r.random((40, 50, 10)).astype(np.float32), aff_f), tmp_path / "flair.nii.gz")
    cli = tmp_path / "cli.py"; cli.write_text(textwrap.dedent(CLI), encoding="utf-8")
    monkeypatch.setenv("EPI_VAR_DIR", str(tmp_path / "var")); monkeypatch.setenv("EPI_G2_CMD", f'"{sys.executable}" "{cli}"')
    from epiapp.config import load_settings
    from epiapp.service import Service
    s = load_settings("real"); assert s.g2_adapter == "external"
    svc = Service(s)
    st = svc.g2_create()
    svc.g2_file(st["id"], "t1", tmp_path / "t1.nii.gz", "t1.nii.gz")
    v = svc.g2_file(st["id"], "flair", tmp_path / "flair.nii.gz", "flair.nii.gz")
    assert v["status"] == "checked" and v["check"]["summary"]["flair_mm"] == 3.0 and not v["check"]["flair5"]
    assert svc.g2_thumb(st["id"], "t1")["w"] == 40
    j = wait_job(svc, svc.g2_run(st["id"])); assert j["status"] == "done", j["error"]
    res = svc.g2_view(st["id"])["result"]
    assert res["info"]["shape"] == [40, 50, 30] and res["qc"]["flags"] == ["проверка"] and res["candidates_from"] == "model"
    c = res["candidates"][0]
    assert c["peak_prob"] == 0.93 and c["ijk"] == [39 - 13, 23, 15]                       # координаты модели попали в сетку просмотра (RAS)
    pack = VOL.load_pack(svc.s.var_dir / "g2" / st["id"] / "pack")
    assert pack["prob"][tuple(c["ijk"])] == round(0.93 * 255 - 0.5) and pack["flair"].shape == (40, 50, 30)
    own = svc.g2_candidates(st["id"], 0.5)                                                # другая рабочая точка — пересчёт приложением
    assert own["source"] == "app" and len(own["candidates"]) == 1 and own["candidates"][0]["volume_ml"] == pytest.approx(0.512)
    assert svc.g2_volume_path(st["id"], "prob").stat().st_size == 40 * 50 * 30


def test_external_command_failure_is_reported(tmp_path, monkeypatch):
    nib.save(nib.Nifti1Image(np.zeros((8, 8, 8), np.float32), np.eye(4)), tmp_path / "a.nii.gz")
    cli = tmp_path / "bad.py"; cli.write_text("import sys; sys.stderr.write('нет FSL\\n'); sys.exit(2)", encoding="utf-8")
    monkeypatch.setenv("EPI_VAR_DIR", str(tmp_path / "var")); monkeypatch.setenv("EPI_G2_CMD", f'"{sys.executable}" "{cli}"')
    from epiapp.config import load_settings
    from epiapp.service import Service
    svc = Service(load_settings("real"))
    st = svc.g2_create()
    for k in ("t1", "flair"):
        svc.g2_file(st["id"], k, tmp_path / "a.nii.gz", "a.nii.gz")
    j = wait_job(svc, svc.g2_run(st["id"]))
    assert j["status"] == "error" and "нет FSL" in j["error"]


def test_demo_adapter(svc):
    v = svc.g2_demo("flair5")
    assert v["check"]["flair5"] and svc.g2_thumb(v["id"], "flair")["demo"]
    assert wait_job(svc, svc.g2_run(v["id"]))["status"] == "done"
    r = svc.g2_view(v["id"])["result"]
    assert len(r["candidates"]) == 1 and r["default_thr"] == 0.9
    n = [len(svc.g2_candidates(v["id"], t)["candidates"]) for t in (0.3, 0.7, 0.9)]
    assert n == [3, 2, 1]
    assert svc.g2.card["table"] and svc.g2.card["table_source"].startswith("h2_mri/notebooks/")
