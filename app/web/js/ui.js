// Общие функции интерфейса: разметка, формат чисел, рисование сигнала ЭЭГ, шкала оценки, ход расчёта.
import { api } from './api.js';

export const $ = (s, r = document) => r.querySelector(s);
export const $$ = (s, r = document) => [...r.querySelectorAll(s)];
export const fmt = (x, d = 2) => Number(x).toFixed(d).replace('.', ',');
export const esc = (s) => String(s ?? '').replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
export const mmss = (t) => `${Math.floor(t / 60)}:${String(Math.floor(t % 60)).padStart(2, '0')}`;
export const state = { meta: null, leave: null, shell: null };
// перечитать настройки сервера (порог клиники, модели) и обновить шапку и строку состояния
export async function refreshMeta() { state.meta = await api.get('/api/meta'); if (state.shell) state.shell(); return state.meta; }
export const C = { ai: '#f3bd4b', warn: '#ff8f66', bad: '#ff6b6b', ink: '#dfe5ea', ink2: '#a9b4be', ink3: '#95a1ac', line: '#272e36', line2: '#3a444e' };

export function go(h) { if (location.hash.slice(1) === h) window.dispatchEvent(new HashChangeEvent('hashchange')); else location.hash = h; }
window.go = go;

export function toast(text, bad = false, ms = 3200) {
  $$('.toast').forEach(t => t.remove());
  const t = document.createElement('div'); t.className = 'toast' + (bad ? ' bad' : ''); t.setAttribute('role', 'status'); t.textContent = text; document.body.appendChild(t);
  setTimeout(() => t.remove(), ms);
}
// слово с числом: plural(2, 'кандидат', 'кандидата', 'кандидатов')
export function plural(n, one, few, many) {
  const a = Math.abs(n) % 100, b = a % 10;
  return a > 10 && a < 20 ? many : b === 1 ? one : b >= 2 && b <= 4 ? few : many;
}
export function when(created) {                       // «15.11 20:40»
  if (!created) return '';
  const d = new Date(created * 1000), p = n => String(n).padStart(2, '0');
  return `${p(d.getDate())}.${p(d.getMonth() + 1)} ${p(d.getHours())}:${p(d.getMinutes())}`;
}
export const loading = (text = 'Загрузка…') => `<div class="loading">${esc(text)}</div>`;
export const errorBox = (title, text, back = "go('list')", backText = 'К списку исследований') =>
  `<div class="empty err"><h2>${esc(title)}</h2><p>${esc(text)}</p><div class="row"><button class="btn" onclick="${back}">${backText}</button></div></div>`;

// Открытые исследования — вкладки в шапке (живут, пока открыто окно браузера)
const TABS_KEY = 'epi.tabs';
export const tabs = {
  all() { try { return JSON.parse(sessionStorage.getItem(TABS_KEY)) || []; } catch (e) { return []; } },
  save(a) { sessionStorage.setItem(TABS_KEY, JSON.stringify(a.slice(-6))); },
  open(kind, id, name) { const a = tabs.all(), i = a.findIndex(t => t.id === id); if (i >= 0) a[i] = { kind, id, name }; else a.push({ kind, id, name }); tabs.save(a); },
  close(id) { tabs.save(tabs.all().filter(t => t.id !== id)); },
};

// Клавиши экрана: keys({KeyC: fn, Enter: fn, ArrowLeft: fn}) — по физической клавише, раскладка не важна. Возвращает функцию снятия.
export function keys(map) {
  const h = e => {
    if (e.ctrlKey || e.metaKey || e.altKey) return;
    const t = e.target.tagName; if (t === 'INPUT' || t === 'TEXTAREA' || t === 'SELECT') return;
    if ((e.key === 'Enter' || e.key === ' ') && (t === 'BUTTON' || t === 'A')) return;      // на кнопке Enter нажимает саму кнопку
    const f = map[e.code] || map[e.key]; if (f) { e.preventDefault(); f(e); }
  };
  addEventListener('keydown', h); return () => removeEventListener('keydown', h);
}

// Зона загрузки файла: кнопка выбора и перетаскивание. onFiles(File[])
export function bindDrop(el, onFiles) {
  const inp = $('input[type=file]', el);
  $$('.pick', el).forEach(b => b.onclick = () => inp.click());
  inp.onchange = () => { if (inp.files.length) onFiles([...inp.files]); inp.value = ''; };
  el.ondragover = e => { e.preventDefault(); e.stopPropagation(); el.classList.add('over'); };
  el.ondragleave = () => el.classList.remove('over');
  el.ondrop = e => { e.preventDefault(); e.stopPropagation(); el.classList.remove('over'); if (e.dataTransfer.files.length) onFiles([...e.dataTransfer.files]); };
}

// Ход расчёта: полоса и этапы
export const progressHTML = (stages) => `<div class="prog" role="progressbar" aria-label="Ход расчёта"><i></i></div><ul class="stages">${stages.map(s => `<li>${esc(s)}</li>`).join('')}</ul>`;
export function progressTick(root, job) {
  const bar = $('.prog i', root); if (bar) bar.style.width = Math.round(100 * (job.status === 'done' ? 1 : job.frac)) + '%';
  $$('.stages li', root).forEach((li, j) => li.className = (job.status === 'done' || j < job.stage) ? 'done' : j === job.stage ? 'now' : '');
}

// Пункты проверки входа (ЭЭГ и МРТ): название, значение со значком состояния, пояснение
export function checkItems(items) {
  const cls = { ok: 'o', warn: 'w', bad: 'x' };
  return `<ul class="items">${items.map(i => `<li><span>${esc(i.title)}</span><b class="${cls[i.status] || ''}">${esc(i.value)}</b>${i.note ? `<small>${esc(i.note)}</small>` : ''}</li>`).join('')}</ul>`;
}

// Схема электродов 10–20; каналы без сигнала — красным квадратом
const POS1020 = { Fp1: [.40, .12], Fp2: [.60, .12], F7: [.20, .30], F3: [.35, .32], Fz: [.50, .32], F4: [.65, .32], F8: [.80, .30], T3: [.12, .50], C3: [.31, .50], Cz: [.50, .50], C4: [.69, .50], T4: [.88, .50], T5: [.20, .70], P3: [.35, .68], Pz: [.50, .68], P4: [.65, .68], T6: [.80, .70], O1: [.40, .88], O2: [.60, .88] };
export function headSVG(bad = []) {
  let s = `<svg viewBox="0 0 100 100" class="head" role="img" aria-label="Схема электродов${bad.length ? ': без сигнала ' + bad.join(', ') : ', все каналы с сигналом'}"><circle cx="50" cy="52" r="43" fill="none" stroke="${C.line2}" stroke-width="1"/><path d="M45 10 L50 4 L55 10" fill="none" stroke="${C.line2}" stroke-width="1"/>`;
  for (const [n, [x, y]] of Object.entries(POS1020)) {
    const cx = x * 100, cy = y * 100 + 2;
    s += bad.includes(n) ? `<rect x="${cx - 3.2}" y="${cy - 3.2}" width="6.4" height="6.4" fill="${C.bad}"/><text x="${cx}" y="${cy - 5.5}" font-size="5" fill="${C.bad}" text-anchor="middle" font-family="Consolas,monospace">${n}</text>` : `<circle cx="${cx}" cy="${cy}" r="2.4" fill="${C.ink3}"/>`;
  }
  return s + '</svg>';
}

// Шкала оценки 0–1: порог, оценка записи (ромб) и разброс по частям; точки — записи клиники с известным диагнозом.
// records: [{score, label}] (label 1 — эпилепсия, 0 — без, null — без разметки)
export function scaleSVG({ score, thr, spread, records = [], width = 306 }) {
  const W = width, X = v => 5 + Math.max(0, Math.min(1, v)) * (W - 10), lab = records.filter(r => r.label === 0 || r.label === 1), un = records.filter(r => r.label == null);
  const H = lab.length ? 58 : un.length ? 46 : 32, yT = H - 12;
  let s = `<svg viewBox="0 0 ${W} ${H}" style="display:block;width:100%" role="img" aria-label="Оценка ${fmt(score)}, порог ${fmt(thr)}">`;
  s += `<rect x="${X(0)}" y="8" width="${W - 10}" height="6" fill="${C.line}"/><rect x="${X(thr)}" y="8" width="${X(1) - X(thr)}" height="6" fill="#414b56"/>`;
  s += `<line x1="${X(thr)}" x2="${X(thr)}" y1="2" y2="${yT}" stroke="${C.ink}" stroke-dasharray="3 2"/>`;
  if (spread) s += `<line x1="${X(spread[0])}" x2="${X(spread[1])}" y1="11" y2="11" stroke="${C.ai}" stroke-width="2"/>`;
  s += `<path d="M${X(score)} 3 l6 8 l-6 8 l-6 -8z" fill="${C.ai}" stroke="${'#14181c'}" stroke-width="1"/>`;
  lab.forEach(r => { s += r.label ? `<circle cx="${X(r.score)}" cy="28" r="3" fill="${C.ink2}"/>` : `<circle cx="${X(r.score)}" cy="40" r="2.6" fill="none" stroke="${C.ink2}" stroke-width="1.2"/>`; });
  if (!lab.length) un.forEach(r => { s += `<line x1="${X(r.score)}" x2="${X(r.score)}" y1="22" y2="30" stroke="${C.ink3}" stroke-width="1"/>`; });
  const f = 'font-size="10" font-family="Consolas,monospace"';
  s += `<text x="${X(0)}" y="${H - 1}" ${f} fill="${C.ink3}">0</text><text x="${Math.min(W - 30, Math.max(22, X(thr)))}" y="${H - 1}" ${f} fill="${C.ink}" text-anchor="middle">${fmt(thr)}</text><text x="${X(1)}" y="${H - 1}" ${f} fill="${C.ink3}" text-anchor="end">1</text>`;
  return s + '</svg>';
}

// Сигнал ЭЭГ на canvas. sig — ответ сервера /signal: {names, data[канал][отсчёт] в единицах unit мкВ, fs, sec, artifacts[[a,b]], bad[]}.
// o.uvmm — чувствительность, мкВ/мм (1 мм экрана ≈ 3,78 точки при 96 точках на дюйм); o.auto — подбор усиления под высоту строки.
export function drawEEG(cv, sig, o = {}) {
  const dpr = window.devicePixelRatio || 1, W = cv.clientWidth, H = cv.clientHeight; if (!W || !H || !sig) return;
  cv.width = W * dpr; cv.height = H * dpr;
  const g = cv.getContext('2d'); g.setTransform(dpr, 0, 0, dpr, 0, 0); g.clearRect(0, 0, W, H);
  const ch = sig.names, sec = sig.sec, L = o.labelW ?? 46, rowH = H / ch.length, pw = W - L, bad = sig.bad || [];
  let pxPerUv = 3.78 / (o.uvmm || 7) * sig.unit;
  if (o.auto) {                                            // по типичному размаху (медиана модулей): артефакты на усиление не влияют
    const a = []; sig.data.forEach((d, i) => { if (!bad.includes(ch[i])) for (let k = 0; k < d.length; k += 7) a.push(Math.abs(d[k])); });
    a.sort((x, y) => x - y); pxPerUv = rowH * 0.14 / ((a[a.length >> 1] || 1) * 1.48 + 1e-6);
  }
  (sig.artifacts || []).forEach(([a, b]) => { g.fillStyle = 'rgba(255,143,102,.10)'; g.fillRect(L + pw * a / sec, 0, pw * (b - a) / sec, H); g.fillStyle = 'rgba(255,143,102,.8)'; g.fillRect(L + pw * a / sec, 0, pw * (b - a) / sec, 2); });
  g.strokeStyle = 'rgba(255,255,255,.07)'; g.lineWidth = 1;
  for (let s = 0; s <= sec; s++) { const x = Math.round(L + pw * s / sec) + .5; g.beginPath(); g.moveTo(x, 0); g.lineTo(x, H); g.stroke(); }
  g.save(); g.beginPath(); g.rect(L, 0, pw, H); g.clip(); g.lineJoin = 'round';
  ch.forEach((name, i) => {
    const y0 = rowH * (i + .5), d = sig.data[i], n = d.length, lim = rowH * 1.8;
    g.strokeStyle = bad.includes(name) ? C.bad : '#c9d3dc'; g.lineWidth = 1; g.beginPath();
    const step = Math.max(1, Math.floor(n / (pw * 2)));
    for (let k = 0; k < n; k += step) { const x = L + pw * k / (n - 1), y = y0 - Math.max(-lim, Math.min(lim, d[k] * pxPerUv)); k ? g.lineTo(x, y) : g.moveTo(x, y); }
    g.stroke();
  });
  g.restore();
  g.font = '11px Consolas,"DejaVu Sans Mono",monospace'; g.textBaseline = 'middle';
  ch.forEach((name, i) => { g.fillStyle = bad.includes(name) ? C.bad : C.ink3; g.fillText(name, 6, rowH * (i + .5)); });
  // масштабная метка: 100 мкВ и 1 с
  const bx = W - 14, by = H - 10, bh = 100 / sig.unit * pxPerUv, uvPerMm = 3.78 * sig.unit / pxPerUv;
  g.strokeStyle = C.ink3; g.fillStyle = C.ink3; g.lineWidth = 1.5; g.beginPath(); g.moveTo(bx, by - bh); g.lineTo(bx, by); g.lineTo(bx - pw / sec, by); g.stroke();
  const cap = `100 мкВ · 1 с${o.auto ? ' · ≈' + Math.round(uvPerMm) + ' мкВ/мм' : ''}`, cw = g.measureText(cap).width, cy = by - Math.min(bh, 40) / 2 - 6;
  g.fillStyle = 'rgba(10,12,14,.85)'; g.fillRect(bx - cw - 12, cy - 9, cw + 10, 18);
  g.fillStyle = C.ink2; g.textAlign = 'right'; g.fillText(cap, bx - 6, cy); g.textAlign = 'left';
}
