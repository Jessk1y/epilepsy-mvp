// Список исследований (главная) и экран «Новое исследование».
// В строке — три вещи: вход, оценка программы, решение врача. Справа — предпросмотр выбранного исследования.
import { $, $$, go, fmt, esc, state, keys, when, plural, bindDrop, drawEEG, toast } from './ui.js';
import { api } from './api.js';

let selId = null, filter = 'all', pending = null;      // pending — файлы, брошенные в окно списка

const isNii = n => /\.nii(\.gz)?$/i.test(n), isEdf = n => /\.edf$/i.test(n);

// что показать в строке и в предпросмотре
function describe(x) {
  const thr = state.meta.g1.threshold.value, rep = x.report ? ' · отчёт выдан' : '';
  const d = { input: ['', ''], score: null, lo: false, detail: '', doctor: '', todo: false };
  if (x.kind === 'eeg') {
    if (x.status === 'rejected') return { ...d, input: ['x', 'не подходит'], detail: 'оценка невозможна' };
    d.input = x.low_quality ? ['w', 'с замечаниями'] : ['o', 'без замечаний'];
    if (x.status === 'running') return { ...d, detail: 'идёт расчёт…' };
    if (x.score == null) return { ...d, detail: 'оценка не рассчитана' };
    d.score = x.score; d.lo = x.score < thr; d.detail = `${d.lo ? 'ниже' : 'выше'} порога ${fmt(thr)}`;
    const a = x.decision && x.decision.agree;
    if (a === true) d.doctor = 'согласен' + rep; else if (a === false) d.doctor = 'не согласен' + rep; else if (x.report) d.doctor = 'отчёт выдан'; else { d.doctor = 'ждёт вас'; d.todo = true; }
    return d;
  }
  if (x.status === 'upload') return { ...d, input: ['x', 'снимки не загружены'], detail: 'нужны T1 и FLAIR' };
  d.input = x.flair5 ? ['w', 'FLAIR 5 мм'] : ['o', 'без замечаний'];
  if (x.status === 'running') return { ...d, detail: 'идёт поиск…' };
  if (x.n_cand == null) return { ...d, detail: 'поиск не запускался' };
  d.score = x.top; d.detail = x.n_cand ? `${x.n_cand} ${plural(x.n_cand, 'кандидат', 'кандидата', 'кандидатов')}` : 'кандидатов нет';
  const marked = Object.keys((x.decision && x.decision.marks) || {}).length;
  if (x.report) d.doctor = 'отчёт выдан'; else if (x.n_cand && marked >= x.n_cand) d.doctor = 'кандидаты отмечены'; else { d.doctor = 'ждёт вас'; d.todo = true; }
  return d;
}

export async function listScreen(root) {
  const alive = () => document.body.contains(wrap);
  let rows = [], wrap = document.createElement('div'); wrap.className = 'center'; wrap.style.flexDirection = 'row';
  root.innerHTML = ''; root.appendChild(wrap);
  const load = async () => {
    const [g1, g2] = await Promise.all([api.get('/api/g1/studies'), api.get('/api/g2/studies')]);
    rows = [...g1.map(x => ({ ...x, kind: 'eeg' })), ...g2.map(x => ({ ...x, kind: 'mri' }))].sort((a, b) => (b.created || 0) - (a.created || 0));
    rows.forEach(x => x.d = describe(x));
  };
  await load();
  const shown = () => rows.filter(x => filter === 'all' || (filter === 'todo' ? x.d.todo : x.kind === filter));
  const count = f => rows.filter(x => f === 'all' || (f === 'todo' ? x.d.todo : x.kind === f)).length;
  const open = x => go(`${x.kind}/${x.id}`);

  const render = () => {
    const demo = state.meta.demo;
    if (!rows.length) {
      wrap.innerHTML = `<div class="empty"><h2>Исследований пока нет</h2>
        <p>Загрузите запись ЭЭГ (.edf) или снимки МРТ — T1 и FLAIR (.nii). Программа проверит вход, рассчитает оценку и будет ждать вашего решения.</p>
        <div class="row"><button class="btn pri lg" onclick="go('new')">Новое исследование <kbd>N</kbd></button>
        ${demo ? '<button class="btn lg" id="dEeg">Взять демо-запись ЭЭГ</button><button class="btn lg" id="dMri">Взять демо-снимки МРТ</button>' : ''}</div></div>`;
      if (demo) {
        $('#dEeg', wrap).onclick = () => api.post('/api/g1/studies/demo?kind=ok').then(s => go('eeg/' + s.id)).catch(e => toast(e.message, true));
        $('#dMri', wrap).onclick = () => api.post('/api/g2/studies/demo?kind=ok').then(s => go('mri/' + s.id)).catch(e => toast(e.message, true));
      }
      return;
    }
    const list = shown(); if (!list.some(x => x.id === selId)) selId = list.length ? list[0].id : null;
    const F = [['all', 'Все'], ['todo', 'Ждут врача'], ['eeg', 'ЭЭГ'], ['mri', 'МРТ']];
    wrap.innerHTML = `<div class="center">
      <div class="lbar"><button class="btn pri" onclick="go('new')">Новое исследование <kbd>N</kbd></button>
        <div class="seg" role="tablist">${F.map(([k, t]) => `<button class="${filter === k ? 'on' : ''}" data-f="${k}" role="tab" aria-selected="${filter === k}">${t} <span class="m">${count(k)}</span></button>`).join('')}</div>
        <span class="hint">Файл .edf или .nii можно перетащить прямо в окно</span></div>
      <div class="lscroll">${list.length ? `<table class="wl"><thead><tr><th>Время</th><th>Вид</th><th>Исследование</th><th>Вход</th><th>Программа</th><th></th><th>Врач</th></tr></thead><tbody>
        ${list.map(x => `<tr class="${x.id === selId ? 'on' : ''}" data-id="${x.id}" tabindex="0" aria-selected="${x.id === selId}">
          <td class="m d">${when(x.created) || esc(x.date)}</td><td><span class="k">${x.kind === 'eeg' ? 'ЭЭГ' : 'МРТ'}</span></td><td title="${esc(x.name)}">${esc(x.name)}</td>
          <td><span class="${x.d.input[0]}">${x.d.input[1]}</span></td><td>${x.d.score == null ? '<span class="d">—</span>' : `<span class="sc ${x.d.lo ? 'lo' : ''}">${fmt(x.d.score)}</span>`}</td>
          <td class="d">${x.d.detail}</td><td class="${x.d.todo ? '' : 'd'}">${x.d.todo ? `<span class="todo">${x.d.doctor}</span>` : x.d.doctor}</td></tr>`).join('')}</tbody></table>`
        : '<div class="empty"><h2>Здесь пусто</h2><p>Под этот отбор не подходит ни одно исследование.</p></div>'}</div></div>
      <aside class="prev" id="prev"></aside>`;
    $$('[data-f]', wrap).forEach(b => b.onclick = () => { filter = b.dataset.f; render(); });
    $$('tr[data-id]', wrap).forEach(tr => {
      tr.onclick = () => { if (selId === tr.dataset.id) return open(rows.find(x => x.id === selId)); selId = tr.dataset.id; render(); };
      tr.ondblclick = () => open(rows.find(x => x.id === tr.dataset.id));
    });
    preview();
  };

  const preview = async () => {
    const box = $('#prev', wrap), x = rows.find(r => r.id === selId); if (!box) return;
    if (!x) { box.innerHTML = ''; return; }
    const d = x.d, eeg = x.kind === 'eeg', canOpen = !(x.kind === 'mri' && x.status === 'upload');
    box.innerHTML = `<div class="img"><canvas></canvas><div class="ph" hidden></div>${x.demo ? `<span class="dm">${eeg ? 'СИНТЕТИКА · ДЕМО' : 'СХЕМА · ДЕМО'}</span>` : ''}</div>
      <div class="b"><div class="lab">${eeg ? 'ЭЭГ · вероятность эпилепсии' : 'МРТ · поиск ФКД'}</div><h2>${esc(x.name)}</h2><p class="sub">${when(x.created) || esc(x.date)}${eeg && x.dur_text ? ' · ' + esc(x.dur_text) : ''}</p>
        <dl class="kv"><dt>Вход</dt><dd><span class="${d.input[0]}">${d.input[1]}</span></dd>
        <dt>Программа</dt><dd>${d.score == null ? d.detail : `<span class="m ai">${fmt(d.score)}</span> · ${d.detail}`}</dd>
        ${d.doctor ? `<dt>Врач</dt><dd>${d.doctor}</dd>` : ''}</dl></div>
      <div class="act">${canOpen ? `<button class="btn pri" id="pOpen">Открыть <kbd>⏎</kbd></button>` : `<button class="btn pri" onclick="go('new')">Загрузить снимки</button>`}
        ${x.status === 'done' ? `<button class="btn" onclick="go('report/${x.kind}/${x.id}')">Заключение</button>` : ''}</div>`;
    const po = $('#pOpen', box); if (po) po.onclick = () => open(x);
    const cv = $('canvas', box), ph = $('.ph', box), none = t => { cv.hidden = true; ph.hidden = false; ph.textContent = t; };
    try {
      if (eeg) {
        if (x.status === 'rejected') return none('Запись не прошла проверку — сигнала для просмотра нет');
        const sig = await api.get(`/api/g1/studies/${x.id}/signal?t0=0&sec=10`); if (selId !== x.id || !alive()) return;
        drawEEG(cv, sig, { auto: true, labelW: 40 });
      } else {
        if (x.status === 'upload') return none('Снимки ещё не загружены');
        const t = await api.get(`/api/g2/studies/${x.id}/thumb/flair`); if (selId !== x.id || !alive()) return;
        const bin = atob(t.data), off = document.createElement('canvas'); off.width = t.w; off.height = t.h;
        const c = off.getContext('2d'), img = c.createImageData(t.w, t.h);
        for (let q = 0; q < bin.length; q++) { const v = bin.charCodeAt(q); img.data[q * 4] = img.data[q * 4 + 1] = img.data[q * 4 + 2] = v; img.data[q * 4 + 3] = 255; }
        c.putImageData(img, 0, 0);
        const dpr = devicePixelRatio || 1, W = cv.clientWidth, H = cv.clientHeight, a = t.aspect || t.w / t.h, h = H - 16, w = h * a;
        cv.width = W * dpr; cv.height = H * dpr; const g = cv.getContext('2d'); g.setTransform(dpr, 0, 0, dpr, 0, 0); g.imageSmoothingQuality = 'high';
        g.drawImage(off, (W - w) / 2, 8, w, h);
      }
    } catch (e) { none('Предпросмотр недоступен'); }
  };

  render();
  // окно принимает файлы: .edf — ЭЭГ, .nii — МРТ
  let veil = null, depth = 0;
  const over = e => { if (![...(e.dataTransfer?.types || [])].includes('Files')) return; e.preventDefault(); };
  const enter = e => { if (![...(e.dataTransfer?.types || [])].includes('Files')) return; depth++; if (!veil) { veil = document.createElement('div'); veil.className = 'dropveil'; veil.textContent = 'Отпустите файл: .edf — запись ЭЭГ, .nii — снимки МРТ'; document.body.appendChild(veil); } };
  const leave = () => { if (--depth <= 0 && veil) { veil.remove(); veil = null; depth = 0; } };
  const drop = e => { e.preventDefault(); if (veil) { veil.remove(); veil = null; } depth = 0; const f = [...(e.dataTransfer?.files || [])]; if (f.length) { pending = f; go('new'); } };
  addEventListener('dragover', over); addEventListener('dragenter', enter); addEventListener('dragleave', leave); addEventListener('drop', drop);
  const move = d => { const l = shown(), i = l.findIndex(x => x.id === selId), n = l[Math.max(0, Math.min(l.length - 1, i + d))]; if (n && n.id !== selId) { selId = n.id; render(); const tr = $('tr.on', wrap); if (tr) tr.scrollIntoView({ block: 'nearest' }); } };
  const off = keys({ ArrowDown: () => move(1), ArrowUp: () => move(-1), Enter: () => { const x = rows.find(r => r.id === selId); if (x && !(x.kind === 'mri' && x.status === 'upload')) open(x); }, KeyN: () => go('new') });
  // пока что-то считается — обновлять список
  const timer = setInterval(async () => { if (!alive()) return; if (rows.some(x => x.status === 'running')) { await load(); if (alive()) render(); } }, 2500);
  state.leave = () => { off(); clearInterval(timer); if (veil) veil.remove(); removeEventListener('dragover', over); removeEventListener('dragenter', enter); removeEventListener('dragleave', leave); removeEventListener('drop', drop); };
}

// ---------------------------------------------------------------- новое исследование
export async function newScreen(root) {
  const m = state.meta, g1 = m.g1.model, g2 = m.g2, files = {};
  root.innerHTML = `<div class="new">
    <section><div class="lab">ЭЭГ</div><h2>Запись ЭЭГ</h2>
      <p class="need">Один файл EDF: рутинная запись от 20 минут, 19 каналов системы 10–20. Программа оценит вероятность эпилепсии по записи целиком.</p>
      <div class="drop" id="dropEeg"><input type="file" accept=".edf,.EDF" hidden><b>Перетащите файл .edf сюда</b><button class="btn pick">Выбрать файл…</button><div class="st" role="status"></div></div>
      <label class="fld" style="max-width:320px"><span>Код пациента — по желанию</span><input type="text" id="code" maxlength="60" autocomplete="off"></label>
      ${g1.available ? '' : `<p class="note bad"><b>Модель «${esc(g1.title)}» недоступна:</b> ${esc(g1.reason)}. Запись можно загрузить и проверить, но оценку получить нельзя.</p>`}
      ${m.demo ? `<div class="demo-row"><span>Демо-режим: своя запись пройдёт настоящую проверку, но оценку даст демо-модель без клинического смысла. Можно взять синтетическую запись:</span>
        <button class="btn" data-demo="g1:ok">Демо-запись</button><button class="btn" data-demo="g1:bad">Демо-запись с замечаниями</button></div>` : ''}
    </section>
    <section><div class="lab">МРТ</div><h2>Снимки МРТ</h2>
      <p class="need">Две серии в формате NIfTI (.nii, .nii.gz): T1 и FLAIR. Программа найдёт кандидатов в очаги ФКД. Папку DICOM сначала преобразуйте в NIfTI (dcm2niix).</p>
      <div class="two"><div class="drop ${g2.upload ? '' : 'dis'}" data-kind="t1"><input type="file" accept=".nii,.gz" hidden><b>T1</b><span class="quiet" style="margin:0">3D, лучше 1 мм</span><button class="btn pick" ${g2.upload ? '' : 'disabled'}>Выбрать…</button><div class="st" role="status"></div></div>
      <div class="drop ${g2.upload ? '' : 'dis'}" data-kind="flair"><input type="file" accept=".nii,.gz" hidden><b>FLAIR</b><span class="quiet" style="margin:0">лучше 3D с тонкими срезами</span><button class="btn pick" ${g2.upload ? '' : 'disabled'}>Выбрать…</button><div class="st" role="status"></div></div></div>
      <div class="st bad" id="mriErr" role="alert" style="margin-top:8px"></div>
      ${g2.upload ? '' : `<p class="note">Модель МРТ не подключена${m.demo ? ' (демо-режим)' : ' — не задана команда модели, настройка EPI_G2_CMD'}: свои снимки загрузить нельзя.</p>
        <div class="demo-row"><span>Можно взять демо-снимки — это схема, созданная формулами, а не снимок человека:</span>
        <button class="btn" data-demo="g2:ok">Демо-снимки</button><button class="btn" data-demo="g2:flair5">Демо-снимки с FLAIR 5 мм</button></div>`}
    </section></div>`;
  const stE = $('#dropEeg .st', root);
  const sendEeg = async (f) => {
    if (!isEdf(f.name)) { stE.className = 'st bad'; stE.textContent = `«${f.name}» — не EDF. Нужен файл с расширением .edf.`; return; }
    stE.className = 'st'; stE.textContent = `Загрузка и проверка: ${f.name}…`;
    try { const s = await api.upload('/api/g1/studies?code=' + encodeURIComponent($('#code', root).value.trim()), f); go('eeg/' + s.id); }
    catch (e) { stE.className = 'st bad'; stE.textContent = `Не получилось: ${e.message}`; }
  };
  bindDrop($('#dropEeg', root), fl => sendEeg(fl[0]));
  const err = t => { $('#mriErr', root).textContent = t; };
  const sendMri = async () => {
    if (!files.t1 || !files.flair) return;
    try {
      const s = await api.post('/api/g2/studies');
      for (const k of ['t1', 'flair']) { const st = $(`.drop[data-kind=${k}] .st`, root); st.className = 'st'; st.textContent = `Загрузка: ${files[k].name}…`; await api.upload(`/api/g2/studies/${s.id}/file?kind=${k}`, files[k], 'PUT'); st.className = 'st done'; st.textContent = files[k].name + ' — загружен'; }
      go('mri/' + s.id);
    } catch (e) { err('Не получилось: ' + e.message); }
  };
  const putMri = (kind, f) => {
    if (!g2.upload) return err('Модель МРТ не подключена: свои снимки загрузить нельзя.');
    if (!isNii(f.name)) return err(`«${f.name}» — не NIfTI. Нужен файл .nii или .nii.gz.`);
    err(''); files[kind] = f; const st = $(`.drop[data-kind=${kind}] .st`, root); st.className = 'st'; st.textContent = f.name + (files.t1 && files.flair ? '' : ' — выбран, нужна вторая серия'); sendMri();
  };
  $$('.drop[data-kind]', root).forEach(d => bindDrop(d, fl => putMri(d.dataset.kind, fl[0])));
  $$('[data-demo]', root).forEach(b => b.onclick = async () => {
    const [g, kind] = b.dataset.demo.split(':'); b.disabled = true;
    try { const s = await api.post(`/api/${g}/studies/demo?kind=${kind}`); go((g === 'g1' ? 'eeg/' : 'mri/') + s.id); } catch (e) { b.disabled = false; toast(e.message, true); }
  });
  state.leave = keys({ Escape: () => go('list') });
  // файлы, брошенные в окно списка: .edf — сразу на проверку; .nii — по сериям (по имени файла)
  if (pending) {
    const fl = pending; pending = null;
    const edf = fl.find(f => isEdf(f.name)), nii = fl.filter(f => isNii(f.name));
    if (edf) sendEeg(edf);
    else if (nii.length) nii.forEach((f, i) => putMri(/flair/i.test(f.name) ? 'flair' : /t1|mprage/i.test(f.name) ? 't1' : (files.t1 || i ? 'flair' : 't1'), f));
    else toast('Нужен файл .edf (ЭЭГ) или .nii / .nii.gz (МРТ)', true);
  }
}
