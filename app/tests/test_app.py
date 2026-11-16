"""Сервер целиком в демо-режиме: путь врача по обеим гипотезам и отчёт."""
import time
import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def client(settings):
    from epiapp.server import create_app
    with TestClient(create_app(settings)) as c:
        yield c


def wait(c, job):
    for _ in range(600):
        j = c.get("/api/jobs/" + job["id"]).json()
        if j["status"] in ("done", "error"):
            return j
        time.sleep(0.1)
    raise TimeoutError


def test_pages_and_meta(client):
    assert "ЭпиАссистент" in client.get("/").text and client.get("/static/js/main.js").status_code == 200
    m = client.get("/api/meta").json()
    assert m["demo"] and m["g1"]["model"]["id"] == "demo" and m["g2"]["adapter"] == "demo" and not m["g2"]["upload"]


def test_g1_upload_run_report(client, edf):
    r = client.post("/api/g1/studies?name=" + "запись%201.edf", content=edf(seconds=200).read_bytes())
    assert r.status_code == 200, r.text
    s = r.json()
    assert s["name"] == "запись 1.edf" and s["check"]["can_run"] and s["demo"]
    assert not list((client.app.state.svc.s.var_dir / "uploads").iterdir())             # исходный EDF не хранится
    sig = client.get(f"/api/g1/studies/{s['id']}/signal?t0=5&sec=10&montage=bip").json()
    assert len(sig["names"]) == 18 and len(sig["data"][0]) == 2000
    assert wait(client, client.post(f"/api/g1/studies/{s['id']}/run").json())["status"] == "done"
    res = client.get(f"/api/g1/studies/{s['id']}").json()["result"]
    assert 0 <= res["score"] <= 1 and res["verdict"] and res["model_card"]["demo"]
    # отчёт: без заключения врача не выдаётся
    bad = client.post("/api/reports", json=dict(study_id=s["id"], doctor="Иванов И. И.", conclusion="  "))
    assert bad.status_code == 422 and "Без заключения врача" in bad.json()["error"]
    ok = client.post("/api/reports", json=dict(study_id=s["id"], doctor="Иванов И. И.", conclusion="Эпилептиформной активности нет <b>"))
    assert ok.status_code == 200
    page = client.get(ok.json()["url"]).text
    assert "Эпилептиформной активности нет &lt;b&gt;" in page and "второе мнение" in page and "Иванов И. И." in page and "ДЕМО" in page
    assert "не локализует" in page and "пилот" not in page.lower()
    assert client.post(f"/api/g1/studies/{s['id']}/decision", json=dict(agree=True)).json()["agree"] is True


def test_report_requires_result(client):
    s = client.post("/api/g1/studies/demo?kind=ok").json()
    assert client.post("/api/reports", json=dict(study_id=s["id"], conclusion="текст заключения")).status_code == 409


def test_rejected_upload(client, edf):
    assert client.post("/api/g1/studies?name=x.txt", content=b"123").status_code == 422
    s = client.post("/api/g1/studies?name=x.edf", content=edf(drop=("O1",)).read_bytes()).json()
    assert s["status"] == "rejected" and client.post(f"/api/g1/studies/{s['id']}/run").status_code == 409


def test_clinic_flow_with_real_uploads(client, edf):
    """Настройка под клинику по загруженным записям: размеченные -> Юден, все -> медиана."""
    for i in range(4):
        r = client.post(f"/api/g1/clinic/records?label={i % 2}&name=c{i}.edf", content=edf(seconds=120, seed=10 + i, theta=0.1 + 0.5 * (i % 2), name=f"c{i}.edf").read_bytes())
        assert wait(client, r.json())["status"] == "done"
    c = client.get("/api/g1/clinic").json()
    assert len(c["records"]) == 4 and c["proposal"]["labeled"]["n_pos"] == 2 and c["proposal"]["warnings"]
    assert c["proposal"]["labeled"]["auc"] == 1.0                                          # демо-модель реагирует на медленную активность
    t = client.post("/api/g1/clinic/threshold", json=dict(method="youden")).json()
    assert t["clinic"] and client.get("/api/meta").json()["g1"]["threshold"]["value"] == t["value"]
    assert client.post("/api/g1/clinic/threshold", json=dict(method="manual", value=7)).status_code == 422
    rid = c["records"][0]["id"]
    assert len(client.delete(f"/api/g1/clinic/records/{rid}").json()["records"]) == 3


def test_g2_demo_flow(client):
    assert client.post("/api/g2/studies").status_code == 409                               # в демо свои снимки не принимаются
    s = client.post("/api/g2/studies/demo?kind=ok").json()
    assert wait(client, client.post(f"/api/g2/studies/{s['id']}/run").json())["status"] == "done"
    r = client.get(f"/api/g2/studies/{s['id']}").json()["result"]
    nx, ny, nz = r["info"]["shape"]
    assert len(client.get(f"/api/g2/studies/{s['id']}/volume/prob").content) == nx * ny * nz
    assert len(client.get(f"/api/g2/studies/{s['id']}/candidates?thr=0.5").json()["candidates"]) == 2
    rep = client.post("/api/reports", json=dict(study_id=s["id"], conclusion="Очаг в левой лобной доле.", thr=0.5)).json()
    page = client.get(rep["url"]).text
    assert "№2" in page and "порог 0,50" in page and "левое полушарие" in page


def test_doctor_decision_goes_to_list_and_report(client):
    """Решение врача видно в списке исследований и попадает в отчёт: согласие по ЭЭГ, отметки кандидатов по МРТ."""
    # --- ЭЭГ: согласие с оценкой
    s = client.post("/api/g1/studies/demo?kind=ok").json()
    assert s["report"] is None
    wait(client, client.post(f"/api/g1/studies/{s['id']}/run").json())
    client.post(f"/api/g1/studies/{s['id']}/decision", json=dict(agree=False))
    rep = client.post("/api/reports", json=dict(study_id=s["id"], conclusion="Признаков эпилепсии нет.")).json()
    assert "не согласен с оценкой программы" in client.get(rep["url"]).text
    row = next(x for x in client.get("/api/g1/studies").json() if x["id"] == s["id"])
    assert row["report"] == rep["id"] and row["decision"]["agree"] is False
    assert client.get(f"/api/g1/studies/{s['id']}").json()["report"] == rep["id"]
    # --- МРТ: отметка у кандидата; ключ — координаты, округлённые как в браузере (Math.round)
    from epiapp.report import cand_key
    assert cand_key(dict(xyz=[-36.25, 13.0, 40.5])) == "-36;13;41" and cand_key(dict(xyz=[-0.4, 2.5, -2.5])) == "0;3;-2"
    g = client.post("/api/g2/studies/demo?kind=ok").json()
    wait(client, client.post(f"/api/g2/studies/{g['id']}/run").json())
    g = client.get(f"/api/g2/studies/{g['id']}").json()
    cand = g["result"]["candidates"][0]
    client.post(f"/api/g2/studies/{g['id']}/decision", json=dict(thr=g["result"]["default_thr"], marks={cand_key(cand): "not"}))
    row = next(x for x in client.get("/api/g2/studies").json() if x["id"] == g["id"])
    assert row["decision"]["marks"] and row["created"] and row["report"] is None
    rep = client.post("/api/reports", json=dict(study_id=g["id"], conclusion="Очага нет.", thr=g["result"]["default_thr"])).json()
    assert "врач: не очаг" in client.get(rep["url"]).text
