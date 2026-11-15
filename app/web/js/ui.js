// Общие функции интерфейса. Рисование и разметка — из макета Дарьи; данные теперь приходят с сервера.
export const $ = (s, r = document) => r.querySelector(s);
export const $$ = (s, r = document) => [...r.querySelectorAll(s)];
export const fmt = (x, d = 2) => Number(x).toFixed(d).replace('.', ',');
export const esc = (s) => String(s ?? '').replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
export const mmss = (t) => `${Math.floor(t / 60)}:${String(Math.floor(t % 60)).padStart(2, '0')}`;
export const state = { meta: null };

export function go(h) { if (location.hash.slice(1) === h) window.dispatchEvent(new HashChangeEvent('hashchange')); else location.hash = h; }
window.go = go;

export function toast(text, ms = 3500) {
  const t = document.createElement('div'); t.className = 'toast'; t.textContent = text; document.body.appendChild(t);
  setTimeout(() => t.remove(), ms);
}

export const STEPS = {
  g1: [['upload', 'Загрузка'], ['check', 'Проверка записи'], ['run', 'Расчёт'], ['result', 'Результат'], ['report', 'Отчёт']],
  g2: [['upload', 'Загрузка'], ['check', 'Проверка снимков'], ['run', 'Расчёт'], ['result', 'Результат'], ['report', 'Отчёт']],
};
export function stepper(tab, cur) {
  const st = STEPS[tab], i = st.findIndex(s => s[0] === cur);
  return `<ol class="stepper">${st.map((s, j) => `<li class="st ${j < i ? 'done' : j === i ? 'on' : ''}"><b>${j < i ? '✓' : j + 1}</b>${s[1]}</li>`).join('')}</ol>`;
}

// Карточки проверки входа (одинаковы для ЭЭГ и МРТ)
export function checkCards(items) {
  return `<div class="checks">${items.map(i => `<div class="chk ${i.status}"><div class="k">${esc(i.title)}</div><div class="v">${esc(i.value)}</div>${i.note ? `<div class="small">${esc(i.note)}</div>` : ''}</div>`).join('')}</div>`;
}

// Зона загрузки файла: клик или перетаскивание. onFiles(FileList)
export function bindDrop(el, onFiles) {
  const inp = $('input[type=file]', el);
  $$('.pick', el).forEach(b => b.onclick = () => inp.click());
  inp.onchange = () => { if (inp.files.length) onFiles(inp.files); inp.value = ''; };
  el.ondragover = e => { e.preventDefault(); el.classList.add('over'); };
  el.ondragleave = () => el.classList.remove('over');
  el.ondrop = e => { e.preventDefault(); el.classList.remove('over'); if (e.dataTransfer.files.length) onFiles(e.dataTransfer.files); };
}

// Ход расчёта: полоса и список этапов (как в макете)
export function progressHTML(stages) {
  return `<div class="prog"><i></i></div><ul class="stages">${stages.map(s => `<li>${esc(s)}</li>`).join('')}</ul>`;
}
export function progressTick(root, job) {
  const bar = $('.prog i', root); if (bar) bar.style.width = Math.round(100 * (job.status === 'done' ? 1 : job.frac)) + '%';
  $$('.stages li', root).forEach((li, j) => li.className = (job.status === 'done' || j < job.stage) ? 'done' : j === job.stage ? 'now' : '');
}

// Положения электродов 10–20 на схеме головы (0..1) — из макета
const POS1020 = { Fp1: [.40, .12], Fp2: [.60, .12], F7: [.20, .30], F3: [.35, .32], Fz: [.50, .32], F4: [.65, .32], F8: [.80, .30], T3: [.12, .50], C3: [.31, .50], Cz: [.50, .50], C4: [.69, .50], T4: [.88, .50], T5: [.20, .70], P3: [.35, .68], Pz: [.50, .68], P4: [.65, .68], T6: [.80, .70], O1: [.40, .88], O2: [.60, .88] };
export function headSVG(bad = []) {
  const c = '#2e7d32', b = '#c62828', line = '#99a';
  let s = `<svg viewBox="0 0 100 100" class="head"><circle cx="50" cy="52" r="42" fill="none" stroke="${line}" stroke-width="1.2"/><path d="M45 11 L50 4 L55 11" fill="none" stroke="${line}" stroke-width="1.2"/>`;
  for (const [n, [x, y]] of Object.entries(POS1020)) {
    const isB = bad.includes(n);
    s += `<circle cx="${x * 100}" cy="${y * 100 + 2}" r="4.2" fill="${isB ? b : c}"/><text x="${x * 100}" y="${y * 100 + 2}" font-size="3" fill="#fff" text-anchor="middle" dominant-baseline="central" font-family="system-ui">${n}</text>`;
  }
  return s + '</svg>';
}

// Шкала оценки с порогом клиники — из макета
export function gauge(v, t) {
  const a = x => Math.PI * (1 - x), P = (x, r) => [150 + r * Math.cos(a(x)), 150 - r * Math.sin(a(x))];
  const arc = (x0, x1, c) => { const [p, q] = [P(x0, 110), P(x1, 110)]; return `<path d="M${p} A110 110 0 0 1 ${q}" stroke="${c}" stroke-width="22" fill="none"/>`; };
  const [tx, ty] = P(t, 126), [t2x, t2y] = P(t, 94), [lx, ly] = P(t, 146), [nx, ny] = P(v, 100);
  return `<svg class="gauge" viewBox="0 -4 300 186">${arc(0, t, '#cfe3d6')}${arc(t, 1, '#f6dcb6')}<line x1="${tx}" y1="${ty}" x2="${t2x}" y2="${t2y}" stroke="#1d2733" stroke-width="3"/>
  <text x="${Math.min(262, Math.max(38, lx))}" y="${ly}" font-size="12" text-anchor="middle">порог ${fmt(t)}</text><line x1="150" y1="150" x2="${nx}" y2="${ny}" stroke="#1d2733" stroke-width="4" stroke-linecap="round"/><circle cx="150" cy="150" r="8"/>
  <text x="150" y="178" font-size="26" font-weight="700" text-anchor="middle">${fmt(v)}</text><text x="34" y="170" font-size="12" fill="#789">0</text><text x="266" y="170" font-size="12" fill="#789">1</text></svg>`;
}

// Сигнал ЭЭГ на canvas. sig — ответ сервера /signal: {names, data[канал][отсчёт] в единицах unit мкВ, fs, sec, artifacts[[a,b]], bad[]}.
// o.uvmm — чувствительность, мкВ/мм (1 мм экрана ≈ 3,78 точки при 96 точках на дюйм).
export function drawEEG(cv, sig, o = {}) {
  const dpr = window.devicePixelRatio || 1, W = cv.clientWidth, H = cv.clientHeight; if (!W || !H || !sig) return;
  cv.width = W * dpr; cv.height = H * dpr;
  const g = cv.getContext('2d'); g.setTransform(dpr, 0, 0, dpr, 0, 0); g.clearRect(0, 0, W, H);
  const ch = sig.names, sec = sig.sec, L = o.labelW ?? 44, rowH = H / ch.length, pw = W - L;
  let pxPerUv = 3.78 / (o.uvmm || 7) * sig.unit;
  if (o.auto) {                                            // подбор усиления под высоту строки: по типичному размаху (медиана модулей), артефакты не влияют
    const a = []; sig.data.forEach((d, i) => { if (!(sig.bad || []).includes(ch[i])) for (let k = 0; k < d.length; k += 7) a.push(Math.abs(d[k])); });
    a.sort((x, y) => x - y); pxPerUv = rowH * 0.2 / ((a[a.length >> 1] || 1) * 1.48 + 1e-6);
  }
  const uvPerMm = 3.78 * sig.unit / pxPerUv;
  g.font = '10px system-ui,sans-serif'; g.textBaseline = 'middle';
  g.strokeStyle = o.grid || 'rgba(128,128,128,.22)'; g.lineWidth = 1;
  for (let s = 0; s <= sec; s++) { const x = L + pw * s / sec + .5; g.beginPath(); g.moveTo(x, 0); g.lineTo(x, H); g.stroke(); }
  (sig.artifacts || []).forEach(([a, b]) => { g.fillStyle = o.artColor || 'rgba(240,160,40,.18)'; g.fillRect(L + pw * a / sec, 0, pw * (b - a) / sec, H); });
  g.save(); g.beginPath(); g.rect(L, 0, pw, H); g.clip();
  ch.forEach((name, i) => {
    const y0 = rowH * (i + .5), bad = (sig.bad || []).includes(name), d = sig.data[i], n = d.length;
    g.strokeStyle = bad ? (o.badColor || '#d33') : (o.color || '#23374d'); g.lineWidth = o.lw || 1; g.beginPath();
    const step = Math.max(1, Math.floor(n / (pw * 2)));
    for (let k = 0; k < n; k += step) { const x = L + pw * k / (n - 1), y = y0 - d[k] * pxPerUv; k ? g.lineTo(x, y) : g.moveTo(x, y); }
    g.stroke();
  });
  g.restore();
  ch.forEach((name, i) => { const bad = (sig.bad || []).includes(name); g.fillStyle = bad ? (o.badColor || '#d33') : (o.labelColor || '#667'); g.fillText(name, 4, rowH * (i + .5)); });
  // масштабная метка: 100 мкВ и 1 с
  const bx = W - 16, by = H - 10, bh = 100 / sig.unit * pxPerUv;
  g.strokeStyle = '#8a6fc9'; g.fillStyle = '#8a6fc9'; g.lineWidth = 1.5; g.beginPath(); g.moveTo(bx, by); g.lineTo(bx, by - bh); g.moveTo(bx, by); g.lineTo(bx - pw / sec, by); g.stroke();
  g.textAlign = 'right'; g.fillText(`100 мкВ · 1 с${o.auto ? ' · ≈' + Math.round(uvPerMm) + ' мкВ/мм' : ''}`, bx - 4, by - bh - 6); g.textAlign = 'left';
}

// Справка о модели: цифры только из паспорта модели (на сервере — со ссылками на файлы результатов)
export function modelHelp(card, title) {
  const m = (card.metrics || []).map(x => `<li>${esc(x.data)}: <b>${esc(x.value)}</b> <span class="small">(${esc(x.source)})</span></li>`).join('');
  const l = (card.limits || []).map(x => `<li>${esc(x)}</li>`).join('');
  return `<details class="card-help"><summary class="small">Справка о модели</summary><p class="small"><b>${esc(title)}</b>. Задача при обучении: ${esc(card.task)}. Обучение: ${esc(card.trained_on)}.</p>
    ${m ? `<p class="small" style="margin-bottom:0">Качество на открытых данных:</p><ul class="small">${m}</ul>` : ''}${l ? `<p class="small" style="margin-bottom:0">Ограничения:</p><ul class="small">${l}</ul>` : ''}</details>`;
}
