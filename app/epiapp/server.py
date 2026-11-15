"""Сервер приложения (FastAPI). Маршруты тонкие: вся логика — в service.py и report.py."""
import urllib.parse
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from . import __version__, report as REPORT
from .config import WEB_DIR, load_settings
from .service import AppError, Service
from .store import new_id

MAX_UPLOAD = 3 * 2 ** 30


def create_app(settings=None):
    s = settings or load_settings()
    svc = Service(s)
    app = FastAPI(title="ЭпиАссистент", version=__version__, docs_url="/api/docs", openapi_url="/api/openapi.json")
    app.add_middleware(GZipMiddleware, minimum_size=2000)
    app.state.svc = svc
    if s.demo:
        svc.warm_demo()

    @app.exception_handler(AppError)
    async def app_error(_, e: AppError):
        return JSONResponse({"error": str(e)}, status_code=e.code)

    @app.middleware("http")
    async def no_cache(request, call_next):
        r = await call_next(request)
        if request.url.path.startswith(("/static", "/api")) or request.url.path == "/":
            r.headers["Cache-Control"] = "no-store"
        return r

    async def save_body(request: Request, suffix):
        """Тело запроса — содержимое файла — потоком на диск рабочего каталога."""
        p = s.var_dir / "uploads" / (new_id("up-") + suffix)
        n = 0
        with open(p, "wb") as f:
            async for chunk in request.stream():
                n += len(chunk)
                if n > MAX_UPLOAD:
                    f.close(); p.unlink(); raise AppError("файл больше 3 ГБ", 413)
                f.write(chunk)
        if n == 0:
            p.unlink(); raise AppError("пустой файл", 422)
        return p

    def fname(name, default):
        return Path(urllib.parse.unquote(name or default)).name[:120] or default

    # ---------------------------------------------------------------- общее
    @app.get("/", response_class=HTMLResponse)
    def index():
        return (WEB_DIR / "index.html").read_text(encoding="utf-8")

    @app.get("/api/meta")
    def meta():
        return svc.meta()

    @app.get("/api/jobs/{jid}")
    def job(jid: str):
        return svc.job(jid)

    # ---------------------------------------------------------------- Г1
    @app.get("/api/g1/studies")
    def g1_list():
        return svc.g1_list()

    @app.post("/api/g1/studies")
    async def g1_upload(request: Request, name: str = "", code: str = ""):
        name = fname(name, "запись.edf")
        if not name.lower().endswith(".edf"):
            raise AppError("нужен файл .edf", 422)
        p = await save_body(request, ".edf")
        try:
            return svc.g1_ingest(p, name, code=code[:60])
        finally:
            svc.cleanup_upload(p)                           # исходный файл не храним

    @app.post("/api/g1/studies/demo")
    def g1_demo(kind: str = "ok"):
        return svc.g1_demo(kind)

    @app.get("/api/g1/studies/{sid}")
    def g1_get(sid: str):
        return svc.g1_view(sid)

    @app.get("/api/g1/studies/{sid}/signal")
    def g1_signal(sid: str, t0: float = 0, sec: float = 10, montage: str = "avg", hp: float = 0.5, lp: float = 70):
        return svc.g1_signal(sid, t0, sec, montage, hp, lp)

    @app.post("/api/g1/studies/{sid}/run")
    def g1_run(sid: str):
        return svc.g1_run(sid)

    @app.post("/api/g1/studies/{sid}/decision")
    async def g1_decision(sid: str, request: Request):
        return svc.decision(sid, "g1", await request.json())

    @app.get("/api/g1/clinic")
    def clinic():
        return svc.clinic()

    @app.post("/api/g1/clinic/demo")
    def clinic_demo():
        return svc.clinic_demo_fill()

    @app.post("/api/g1/clinic/records")
    async def clinic_add(request: Request, name: str = "", label: str = ""):
        name = fname(name, "запись.edf")
        if not name.lower().endswith(".edf"):
            raise AppError("нужен файл .edf", 422)
        p = await save_body(request, ".edf")
        try:
            return svc.clinic_add_record(p, name, {"1": 1, "0": 0}.get(label))
        finally:
            svc.cleanup_upload(p)

    @app.patch("/api/g1/clinic/records/{rid}")
    async def clinic_label(rid: str, request: Request):
        svc.clinic_label(rid, (await request.json()).get("label"))
        return svc.clinic()

    @app.delete("/api/g1/clinic/records/{rid}")
    def clinic_del(rid: str):
        svc.store.delete_clinic_record(rid)
        return svc.clinic()

    @app.delete("/api/g1/clinic")
    def clinic_clear(records: int = 0):
        return svc.clinic_reset(bool(records))

    @app.post("/api/g1/clinic/threshold")
    async def clinic_thr(request: Request):
        b = await request.json()
        return svc.clinic_set_threshold(b.get("method"), b.get("value"))

    # ---------------------------------------------------------------- Г2
    @app.get("/api/g2/studies")
    def g2_list():
        return svc.g2_list()

    @app.post("/api/g2/studies/demo")
    def g2_demo(kind: str = "ok"):
        return svc.g2_demo(kind)

    @app.post("/api/g2/studies")
    def g2_create(code: str = ""):
        return svc.g2_create(code[:60])

    @app.put("/api/g2/studies/{sid}/file")
    async def g2_file(sid: str, request: Request, kind: str, name: str = ""):
        name = fname(name, kind + ".nii.gz")
        low = name.lower()
        if not (low.endswith(".nii") or low.endswith(".nii.gz")):
            raise AppError("нужен файл NIfTI (.nii или .nii.gz). Папку DICOM сначала преобразуйте в NIfTI (dcm2niix).", 422)
        p = await save_body(request, ".nii.gz" if low.endswith(".gz") else ".nii")
        return svc.g2_file(sid, kind, p, name)

    @app.get("/api/g2/studies/{sid}")
    def g2_get(sid: str):
        return svc.g2_view(sid)

    @app.post("/api/g2/studies/{sid}/run")
    def g2_run(sid: str):
        return svc.g2_run(sid)

    @app.get("/api/g2/studies/{sid}/volume/{name}")
    def g2_volume(sid: str, name: str):
        return FileResponse(svc.g2_volume_path(sid, name), media_type="application/octet-stream")

    @app.get("/api/g2/studies/{sid}/thumb/{kind}")
    def g2_thumb(sid: str, kind: str):
        return svc.g2_thumb(sid, kind)

    @app.get("/api/g2/studies/{sid}/candidates")
    def g2_candidates(sid: str, thr: float):
        return svc.g2_candidates(sid, thr)

    @app.post("/api/g2/studies/{sid}/decision")
    async def g2_decision(sid: str, request: Request):
        return svc.decision(sid, "g2", await request.json())

    # ---------------------------------------------------------------- отчёт
    @app.post("/api/reports")
    async def report_issue(request: Request):
        b = await request.json()
        r = REPORT.issue(svc, b.get("study_id"), b.get("doctor"), b.get("conclusion"), b.get("thr"))
        return dict(id=r["id"], url=f"/reports/{r['id']}")

    @app.get("/reports/{rid}", response_class=HTMLResponse)
    def report_page(rid: str):
        r = svc.store.report(rid)
        if not r:
            raise AppError("отчёт не найден", 404)
        return REPORT.render(r)

    app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")
    return app
