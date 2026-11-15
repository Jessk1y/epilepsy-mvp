"""Очередь расчётов. Один рабочий поток: на рабочей машине одна видеокарта, расчёты идут по одному.
Интерфейс опрашивает состояние задачи (этап, доля выполнения) и рисует ход, как в макете."""
import queue, threading, time, traceback, uuid


class Job:
    def __init__(self, title, stages):
        self.id, self.title, self.stages = uuid.uuid4().hex[:10], title, list(stages)
        self.stage, self.frac, self.status, self.error, self.result = 0, 0.0, "queued", "", None
        self.created = time.time(); self.started = self.finished = None

    def progress(self, stage, frac):
        self.stage, self.frac = int(stage), float(frac)

    def as_dict(self):
        return dict(id=self.id, title=self.title, stages=self.stages, stage=self.stage, frac=round(self.frac, 3), status=self.status,
                    error=self.error, result=self.result,
                    seconds=round((self.finished or time.time()) - self.started, 1) if self.started else 0)


class JobQueue:
    def __init__(self):
        self.jobs, self.q = {}, queue.Queue()
        threading.Thread(target=self._work, daemon=True).start()

    def submit(self, title, stages, fn):
        """fn(job) -> результат (сериализуемый). Ошибка попадает в job.error, приложение не падает."""
        job = Job(title, stages)
        self.jobs[job.id] = job
        self.q.put((job, fn))
        return job

    def get(self, jid):
        return self.jobs.get(jid)

    def position(self, jid):
        return sum(1 for j in self.jobs.values() if j.status == "queued" and j.created <= self.jobs[jid].created)

    def _work(self):
        while True:
            job, fn = self.q.get()
            job.status, job.started = "running", time.time()
            try:
                job.result = fn(job); job.status, job.frac = "done", 1.0
            except Exception as e:  # noqa: BLE001
                job.status, job.error = "error", f"{type(e).__name__}: {e}"
                traceback.print_exc()
            job.finished = time.time()
