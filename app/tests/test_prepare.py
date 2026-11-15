"""Подготовка ЭЭГ и проверка входа на синтетическом EDF."""
import numpy as np

import tuep_common as C
from epiapp.g1 import prepare as P


def test_shape_and_same_as_shared_prep(edf):
    """Без каналов-пустышек подготовка совпадает с tuep_common.load_record (общий набор Г1), если режекция та же."""
    p = edf()
    rec, chk = P.prepare_edf(str(p), notch_hz=60.0)
    ref, info = C.load_record(str(p))
    assert chk["can_run"] and rec.x.shape == (19, 150 * 200) == ref.shape
    assert np.allclose(rec.x, ref.astype(np.float32), rtol=2e-3, atol=0.02), float(np.abs(rec.x - ref).max())   # разница — только округление float16
    assert abs(rec.x.mean(0)).max() < 1.0                       # средний референт
    assert chk["summary"]["sfreq"] == 250 and chk["summary"]["n_good"] == 19


def test_mains_is_removed(edf):
    rec, _ = P.prepare_edf(str(edf()), notch_hz=50.0)
    f = np.fft.rfftfreq(rec.x.shape[1], 1 / 200); S = np.abs(np.fft.rfft(rec.x[0]))
    assert S[(f > 49.8) & (f < 50.2)].max() < 3 * np.median(S[(f > 40) & (f < 45)])


def test_short_record_warns_and_too_short_fails(edf):
    _, chk = P.prepare_edf(str(edf(seconds=150)))
    assert next(i for i in chk["items"] if i["key"] == "duration")["status"] == "warn" and chk["low_quality"]
    rec, chk = P.prepare_edf(str(edf(seconds=40, name="s.edf")))
    assert rec is None and not chk["can_run"]


def test_flat_channels_are_zeroed_and_reported(edf):
    rec, chk = P.prepare_edf(str(edf(flat=("T4", "O2"))))
    assert sorted(chk["flat"]) == ["O2", "T4"] and chk["can_run"] and chk["low_quality"]
    assert next(i for i in chk["items"] if i["key"] == "channels")["value"] == "17 из 19"
    assert np.all(rec.x[C.TARGET.index("T4")] == 0) and np.all(rec.x[C.TARGET.index("O2")] == 0)
    good = [i for i in range(19) if C.TARGET[i] not in ("T4", "O2")]
    assert abs(rec.x[good].mean(0)).max() < 1.0                 # референт — по каналам с сигналом


def test_missing_channel_cannot_run(edf):
    rec, chk = P.prepare_edf(str(edf(drop=("Pz",))))
    assert rec is None and not chk["can_run"] and chk["missing"] == ["Pz"]


def test_not_edf(tmp_path):
    p = tmp_path / "x.edf"; p.write_bytes(b"\xffBIOSEMI" + b"0" * 600)
    rec, chk = P.prepare_edf(str(p))
    assert rec is None and not chk["can_run"] and "BDF" in chk["items"][0]["note"]


def test_artifacts_fraction(edf):
    _, clean = P.prepare_edf(str(edf(seconds=300, art_frac=0.0, name="a.edf")))
    _, dirty = P.prepare_edf(str(edf(seconds=300, art_frac=0.4, name="b.edf")))
    assert clean["summary"]["art_pct"] <= 3, clean["summary"]
    assert 25 <= dirty["summary"]["art_pct"] <= 60 and dirty["artifacts"], dirty["summary"]


def test_channel_names():
    assert P.norm_name("EEG Fp1-A1") == ("FP1", "A1") and P.norm_name("T7") == ("T3", "") and P.norm_name("EEG FP1-REF")[0] == "FP1"
    assert P.norm_name("Fp1-F3") == (None, None) and P.norm_name("EKG1") == (None, None) and P.norm_name("EEG P8-M2")[0] == "T6"


def test_view_chunk(edf):
    rec, _ = P.prepare_edf(str(edf()))
    names, data, fs, bad = P.view_chunk(rec.x, 10, 10, "avg")
    assert len(names) == 19 and data.shape == (19, 2000) and fs == 200
    names, data, _, _ = P.view_chunk(rec.x, 145, 10, "bip")       # у конца записи кусок короче
    assert len(names) == 18 and data.shape == (18, 1000)
