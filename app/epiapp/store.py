"""Хранилище приложения — SQLite в рабочем каталоге (EPI_VAR_DIR). Сигнал и снимки в базу не пишутся: только сведения,
оценки, пороги, решения врача и отчёты. Демо-записи помечаются demo = 1 и с настоящими не смешиваются."""
import json, sqlite3, threading, time, uuid

SCHEMA = """
CREATE TABLE IF NOT EXISTS studies(id TEXT PRIMARY KEY, kind TEXT, name TEXT, code TEXT, created REAL, demo INT, status TEXT,
  model_id TEXT, check_json TEXT, result_json TEXT, decision_json TEXT, files_json TEXT, error TEXT);
CREATE TABLE IF NOT EXISTS clinic_records(id TEXT PRIMARY KEY, model_id TEXT, name TEXT, score REAL, label INT, source TEXT,
  study_id TEXT, created REAL, demo INT, low_quality INT);
CREATE TABLE IF NOT EXISTS thresholds(id INTEGER PRIMARY KEY AUTOINCREMENT, model_id TEXT, method TEXT, value REAL, created REAL,
  demo INT, details_json TEXT);
CREATE TABLE IF NOT EXISTS reports(id TEXT PRIMARY KEY, study_id TEXT, kind TEXT, created REAL, doctor TEXT, conclusion TEXT, payload_json TEXT);
CREATE TABLE IF NOT EXISTS kv(k TEXT PRIMARY KEY, v TEXT);
"""
JSON_COLS = ("check_json", "result_json", "decision_json", "files_json", "details_json", "payload_json")


def new_id(prefix=""):
    return prefix + uuid.uuid4().hex[:10]


class Store:
    def __init__(self, path):
        self.db = sqlite3.connect(str(path), check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.lock = threading.RLock()
        with self.lock:
            self.db.executescript(SCHEMA); self.db.commit()

    def _row(self, r):
        if r is None:
            return None
        d = dict(r)
        for c in JSON_COLS:
            if c in d:
                d[c[:-5]] = json.loads(d.pop(c)) if d[c] else None
        return d

    def q(self, sql, args=()):
        with self.lock:
            return [self._row(r) for r in self.db.execute(sql, args).fetchall()]

    def one(self, sql, args=()):
        r = self.q(sql, args)
        return r[0] if r else None

    def run(self, sql, args=()):
        with self.lock:
            cur = self.db.execute(sql, args); self.db.commit()
            return cur.lastrowid

    # ---- исследования
    def add_study(self, kind, name, demo, check=None, files=None, status="checked", code=""):
        sid = new_id(kind + "-")
        self.run("INSERT INTO studies(id, kind, name, code, created, demo, status, check_json, files_json) VALUES(?,?,?,?,?,?,?,?,?)",
                 (sid, kind, name, code, time.time(), int(demo), status, json.dumps(check, ensure_ascii=False), json.dumps(files or {}, ensure_ascii=False)))
        return sid

    def study(self, sid):
        return self.one("SELECT * FROM studies WHERE id = ?", (sid,))

    def update_study(self, sid, **kw):
        sets, args = [], []
        for k, v in kw.items():
            if k in ("check", "result", "decision", "files"):
                k, v = k + "_json", json.dumps(v, ensure_ascii=False)
            sets.append(f"{k} = ?"); args.append(v)
        self.run(f"UPDATE studies SET {', '.join(sets)} WHERE id = ?", (*args, sid))

    def studies(self, kind, limit=30):
        return self.q("SELECT id, kind, name, code, created, demo, status, model_id, check_json, result_json, decision_json FROM studies "
                      "WHERE kind = ? ORDER BY created DESC LIMIT ?", (kind, limit))

    # ---- записи клиники и пороги (Г1)
    def add_clinic_record(self, model_id, name, score, label=None, source="calib", study_id=None, demo=False, low_quality=False):
        rid = new_id("r-")
        self.run("INSERT INTO clinic_records VALUES(?,?,?,?,?,?,?,?,?,?)",
                 (rid, model_id, name, float(score), label, source, study_id, time.time(), int(demo), int(low_quality)))
        return rid

    def clinic_records(self, model_id, demo):
        return self.q("SELECT * FROM clinic_records WHERE model_id = ? AND demo = ? ORDER BY created", (model_id, int(demo)))

    def set_label(self, rid, label):
        self.run("UPDATE clinic_records SET label = ? WHERE id = ?", (label, rid))

    def delete_clinic_record(self, rid):
        self.run("DELETE FROM clinic_records WHERE id = ?", (rid,))

    def clear_clinic(self, model_id, demo):
        self.run("DELETE FROM clinic_records WHERE model_id = ? AND demo = ?", (model_id, int(demo)))
        self.run("DELETE FROM thresholds WHERE model_id = ? AND demo = ?", (model_id, int(demo)))

    def save_threshold(self, model_id, method, value, demo, details):
        return self.run("INSERT INTO thresholds(model_id, method, value, created, demo, details_json) VALUES(?,?,?,?,?,?)",
                        (model_id, method, float(value), time.time(), int(demo), json.dumps(details, ensure_ascii=False)))

    def threshold(self, model_id, demo):
        """Действующий порог клиники для модели — последний сохранённый; None, если не настраивался."""
        return self.one("SELECT * FROM thresholds WHERE model_id = ? AND demo = ? ORDER BY id DESC LIMIT 1", (model_id, int(demo)))

    # ---- отчёты
    def add_report(self, study_id, kind, doctor, conclusion, payload):
        rid = new_id("rep-")
        self.run("INSERT INTO reports VALUES(?,?,?,?,?,?,?)", (rid, study_id, kind, time.time(), doctor, conclusion, json.dumps(payload, ensure_ascii=False)))
        return rid

    def report(self, rid):
        return self.one("SELECT * FROM reports WHERE id = ?", (rid,))

    # ---- прочее
    def get(self, k, default=None):
        r = self.one("SELECT v FROM kv WHERE k = ?", (k,))
        return json.loads(r["v"]) if r else default

    def put(self, k, v):
        self.run("INSERT INTO kv(k, v) VALUES(?, ?) ON CONFLICT(k) DO UPDATE SET v = excluded.v", (k, json.dumps(v, ensure_ascii=False)))
