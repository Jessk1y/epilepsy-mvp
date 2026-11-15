// Обращения к серверу. Ошибки сервера приходят как {error: "…"} — текст показывается врачу.
async function parse(r) {
  let body = null;
  try { body = await r.json(); } catch (e) { /* не JSON */ }
  if (!r.ok) throw new Error((body && (body.error || body.detail)) || `ошибка сервера (${r.status})`);
  return body;
}
export const api = {
  get: (url) => fetch(url).then(parse),
  post: (url, data) => fetch(url, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: data === undefined ? null : JSON.stringify(data) }).then(parse),
  patch: (url, data) => fetch(url, { method: 'PATCH', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(data) }).then(parse),
  del: (url) => fetch(url, { method: 'DELETE' }).then(parse),
  // файл уходит телом запроса на сервер клиники (без multipart), имя — в адресе
  upload: (url, file, method = 'POST') => fetch(url + (url.includes('?') ? '&' : '?') + 'name=' + encodeURIComponent(file.name),
    { method, headers: { 'Content-Type': 'application/octet-stream' }, body: file }).then(parse),
  bytes: (url) => fetch(url).then(r => { if (!r.ok) throw new Error('объём не получен'); return r.arrayBuffer(); }).then(b => new Uint8Array(b)),
};
// Ждать задачу расчёта; onTick(job) — для отрисовки хода. Возвращает задачу в состоянии done, иначе бросает ошибку.
export async function waitJob(id, onTick, alive = () => true) {
  for (;;) {
    const j = await api.get('/api/jobs/' + id);
    if (!alive()) return null;
    onTick && onTick(j);
    if (j.status === 'done') return j;
    if (j.status === 'error') throw new Error(j.error || 'расчёт завершился с ошибкой');
    await new Promise(r => setTimeout(r, 350));
  }
}
