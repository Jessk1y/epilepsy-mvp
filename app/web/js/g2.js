// Г2 · МРТ: пошаговый режим (загрузка → проверка → расчёт → результат → отчёт). Разметка и тексты — из макета Дарьи.
import { $, $$, go, fmt, esc, state, checkCards, bindDrop, progressHTML, progressTick } from './ui.js';
import { api, waitJob } from './api.js';
import { reportScreen } from './report.js';
import { Mri, loadCands, xyzText, opText } from './mri.js';

const S = { seq: 'T1', ov: true, op: .75, showBelow: false };
const wpOf = (s) => (s.decision && s.decision.thr) || s.result.default_thr;
const loOf = (card) => (card.table && card.table.length ? card.table[0].thr : 0.3);

async function drawThumb(cv, sid, kind) {
  const t = await api.get(`/api/g2/studies/${sid}/thumb/${kind}`), bin = atob(t.data);
  cv.width = t.w; cv.height = t.h; cv.style.aspectRatio = t.aspect;
  const c = cv.getContext('2d'), img = c.createImageData(t.w, t.h);
  for (let q = 0; q < bin.length; q++) { const v = bin.charCodeAt(q); img.data[q * 4] = img.data[q * 4 + 1] = img.data[q * 4 + 2] = v; img.data[q * 4 + 3] = 255; }
  c.putImageData(img, 0, 0);
}

export const G2 = {
  // ---------------------------------------------------------------- загрузка
  async upload(root) {
    const g = state.meta.g2, demo = !g.upload, files = {};      // демо-снимки доступны, пока модель МРТ не подключена
    root.innerHTML = `<div class="card"><h2>Загрузите МРТ пациента</h2><p class="sub">Нужны две серии: T1 и FLAIR в формате NIfTI (.nii, .nii.gz). Папку DICOM сначала преобразуйте в NIfTI (dcm2niix).</p>
      <div class="row"><div class="drop col" data-kind="t1"><input type="file" accept=".nii,.gz"><div class="big">T1</div><div class="small">3D, лучше 1 мм</div><p><button class="btn pick" ${g.upload ? '' : 'disabled'}>Выбрать…</button></p><div class="small st"></div></div>
      <div class="drop col" data-kind="flair"><input type="file" accept=".nii,.gz"><div class="big">FLAIR</div><div class="small">лучше 3D тонкие срезы</div><p><button class="btn pick" ${g.upload ? '' : 'disabled'}>Выбрать…</button></p><div class="small st"></div></div></div>
      ${g.upload ? '' : `<div class="note gray">Модель МРТ не подключена${state.meta.demo ? ' (демо-режим)' : ' (не задана команда модели, настройка EPI_G2_CMD)'}: свои снимки загрузить нельзя — возьмите демо-снимки (это схема, а не снимок человека).</div>`}
      <div id="upErr"></div>
      <div class="actions"><span class="small">Обработка на сервере клиники.</span><span>${demo ? '<button class="btn" id="demo5">Демо: FLAIR 5 мм</button> <button class="btn pri" id="demoOk">Взять демо-снимки →</button>' : ''}</span></div></div>`;
    const err = (e) => { $('#upErr', root).innerHTML = `<div class="note bad">${esc(e.message)}</div>`; };
    if (demo) {
      $('#demoOk', root).onclick = () => api.post('/api/g2/studies/demo?kind=ok').then(s => go('g2/check/' + s.id)).catch(err);
      $('#demo5', root).onclick = () => api.post('/api/g2/studies/demo?kind=flair5').then(s => go('g2/check/' + s.id)).catch(err);
    }
    if (!g.upload) return;
    const send = async () => {
      if (!files.t1 || !files.flair) return;
      try {
        const s = await api.post('/api/g2/studies');
        for (const k of ['t1', 'flair']) { $(`.drop[data-kind=${k}] .st`, root).textContent = `Загрузка: ${files[k].name}…`; await api.upload(`/api/g2/studies/${s.id}/file?kind=${k}`, files[k], 'PUT'); $(`.drop[data-kind=${k}] .st`, root).textContent = files[k].name + ' ✓'; }
        go('g2/check/' + s.id);
      } catch (e) { err(e); }
    };
    $$('.drop', root).forEach(d => bindDrop(d, fl => { files[d.dataset.kind] = fl[0]; $('.st', d).textContent = fl[0].name; send(); }));
  },

  // ---------------------------------------------------------------- проверка снимков
  async check(root, id) {
    if (!id) return go('g2/upload');
    const s = await api.get('/api/g2/studies/' + id), c = s.check;
    if (!c) return go('g2/upload');
    const t = c.flair5;
    root.innerHTML = `<div class="card"><h2>Проверка снимков</h2><p class="sub">${esc(s.demo ? 'Демо-пациент' : s.name)} · проверьте качество до запуска</p>
      <div class="row"><div class="col">${checkCards(c.items)}</div>
       <div class="thumbs"><div><canvas id="th1"></canvas><p>T1</p></div><div><canvas id="th2"></canvas><p>FLAIR</p></div></div></div>
      ${(c.notes || []).map(n => `<div class="note bad">${esc(n)}</div>`).join('')}
      ${t ? `<div class="note bad"><b>FLAIR с толстыми срезами (5 мм).</b> Находимость очагов при таком FLAIR резко падает. Если есть 3D FLAIR с тонкими срезами — загрузите его. Иначе результат будет помечен как полученный на снимках низкого качества. <span class="small">${esc(s.card.flair5_note || '')}</span></div>` : ''}
      ${s.demo ? '<p class="small">Демо: «снимки» — схема, созданная формулами, а не снимок человека.</p>' : ''}
      <div class="actions"><button class="btn" onclick="go('g2/upload')">← Назад</button>${c.can_run ? `<button class="btn pri" onclick="go('g2/run/${s.id}')">${t ? 'Всё равно рассчитать' : 'Запустить поиск'} →</button>` : '<span></span>'}</div></div>`;
    drawThumb($('#th1', root), id, 't1').catch(() => {}); drawThumb($('#th2', root), id, 'flair').catch(() => {});
  },

  // ---------------------------------------------------------------- расчёт
  async run(root, id) {
    if (!id) return go('g2/upload');
    const card = state.meta.g2.card;
    const job = await api.post(`/api/g2/studies/${id}/run`);
    root.innerHTML = `<div class="card"><h2>Идёт поиск очагов</h2><p class="sub">${esc(card.eta)}</p>${progressHTML(job.stages)}<div id="runErr"></div></div>`;
    try {
      const j = await waitJob(job.id, x => progressTick(root, x), () => document.body.contains(root));
      if (j) setTimeout(() => go('g2/result/' + id), 300);
    } catch (e) {
      $('#runErr', root).innerHTML = `<div class="note bad"><b>Расчёт не выполнен.</b> ${esc(e.message)}</div><div class="actions"><button class="btn" onclick="go('g2/check/${id}')">← К проверке</button><span></span></div>`;
    }
  },

  // ---------------------------------------------------------------- результат
  async result(root, id) {
    if (!id) return go('g2/upload');
    const s = await api.get('/api/g2/studies/' + id), r = s.result;
    if (!r) return go('g2/check/' + id);
    const card = s.card, wp = wpOf(s);
    root.innerHTML = '<div class="card"><p class="sub">Загрузка снимков…</p></div>';
    const mri = await Mri.load(id, r.info);
    let { above, below } = await loadCands(id, wp, loOf(card));
    let sel = above.length ? 1 : 0, pos = above.length ? { i: above[0].ijk[0], j: above[0].ijk[1], k: above[0].ijk[2] } : mri.center();
    const op = opText(card, wp, fmt), flags = (r.qc && r.qc.flags) || [];
    const render = () => {
      const rows = S.showBelow ? above.concat(below) : above;
      root.innerHTML = `<div class="card"><h2>Результат: кандидаты в очаги ФКД</h2>
        <p class="sub">Найдено выше рабочей точки: <b>${above.length}</b> · рабочая точка: порог ${fmt(wp)}${op ? ' (' + op + ')' : ''}</p>
        ${s.check.flair5 ? '<div class="note warn">FLAIR 5 мм: часть очагов могла быть пропущена.</div>' : ''}
        ${flags.length ? `<div class="note warn">Замечания модели к снимкам: ${esc(flags.join('; '))}</div>` : ''}
        ${card.demo ? '<div class="note warn"><b>ДЕМО.</b> Снимки и карта вероятности — схема, созданная формулами; модель не запускалась.</div>' : ''}
        <div class="row"><div class="col" style="flex:1.1"><div class="viewer"><canvas id="vw"></canvas></div>
         <div class="vctl"><span class="seg"><button class="${S.seq === 'T1' ? 'on' : ''}" data-seq="T1">T1</button><button class="${S.seq === 'FLAIR' ? 'on' : ''}" data-seq="FLAIR">FLAIR</button></span>
         <label><input type="checkbox" id="ov" ${S.ov ? 'checked' : ''}> карта вероятности</label>
         <label>прозрачность <input type="range" id="op" min="0" max="1" step=".05" value="${S.op}"></label></div>
         <label class="small">Срез <input id="sl" type="range" min="0" max="${mri.nz - 1}" step="1" value="${pos.k}" style="width:70%"></label></div>
        <div class="col"><h3>Кандидаты</h3>${rows.length ? `<table><tr><th>№</th><th>Пик. вероятность</th><th>Объём</th><th>Координаты, мм</th></tr>
         ${rows.map(L => `<tr class="click ${L.below ? 'below' : ''} ${sel === L.n ? 'sel' : ''}" data-n="${L.n}"><td><b>${L.n}</b></td><td>${fmt(L.peak_prob)}</td><td>${fmt(L.volume_ml, 1)} см³</td><td>${xyzText(L)}<div class="small">${L.hemi} полушарие</div></td></tr>`).join('')}</table>` : '<p class="small">Выше рабочей точки кандидатов нет.</p>'}
         ${below.length ? `<button class="btn link" id="tgl">${S.showBelow ? 'Скрыть' : 'Показать'} кандидатов ниже рабочей точки (${below.length})</button>` : ''}
         <div class="note info"><b>Второе мнение.</b> Решение за рентгенологом. Ложные находки возможны; часть очагов модель не находит${op ? ' (' + op + ')' : ''}.</div>
         <p class="small">Координаты — в системе снимка (мм); сторона определена относительно середины объёма. Рабочая точка меняется в рабочем месте.</p></div></div>
        <div class="actions"><span><button class="btn" onclick="go('g2/upload')">Новый пациент</button> <button class="btn" onclick="go('ws/g2/${s.id}')">Открыть в рабочем месте</button></span><button class="btn pri" onclick="go('g2/report/${s.id}')">Сформировать отчёт →</button></div></div>`;
      const cv = $('#vw', root), g = mri.geom('ax'); cv.style.aspectRatio = g.pw / g.ph;
      const draw = () => { mri.draw(cv, 'ax', pos, { seq: S.seq, op: S.op, overlay: S.ov, cands: rows, sel, demo: card.demo, caption: `аксиальный срез ${pos.k + 1}/${mri.nz}` }); $('#sl', root).value = pos.k; };
      $$('[data-seq]', root).forEach(b => b.onclick = () => { S.seq = b.dataset.seq; render(); });
      $('#ov', root).onchange = e => { S.ov = e.target.checked; draw(); };
      $('#op', root).oninput = e => { S.op = +e.target.value; draw(); };
      $('#sl', root).oninput = e => { pos = { ...pos, k: +e.target.value }; draw(); };
      $$('tr[data-n]', root).forEach(tr => tr.onclick = () => { const L = rows.find(x => x.n === +tr.dataset.n); sel = L.n; pos = { i: L.ijk[0], j: L.ijk[1], k: L.ijk[2] }; render(); });
      const tg = $('#tgl', root); if (tg) tg.onclick = () => { S.showBelow = !S.showBelow; render(); };
      cv.onwheel = e => { e.preventDefault(); pos = mri.clampPos({ ...pos, k: pos.k + (e.deltaY < 0 ? 1 : -1) }); draw(); };
      requestAnimationFrame(draw);
    };
    const onResize = () => render(); addEventListener('resize', onResize); state.leave = () => removeEventListener('resize', onResize);
    render();
  },

  // ---------------------------------------------------------------- отчёт
  async report(root, id) {
    if (!id) return go('g2/upload');
    const s = await api.get('/api/g2/studies/' + id), r = s.result;
    if (!r) return go('g2/check/' + id);
    const wp = wpOf(s), { above } = await loadCands(id, wp, null), c = s.check, op = opText(s.card, wp, fmt);
    const rows = `<dt>Снимки</dt><dd>T1 ${esc(c.summary.t1)} · FLAIR ${esc(c.summary.flair)}${c.flair5 ? ' (низкое качество)' : ''}</dd>
      <dt>Кандидаты выше рабочей точки</dt><dd>${above.length ? above.map(L => `№${L.n}: вероятность ${fmt(L.peak_prob)}, ${fmt(L.volume_ml, 1)} см³, (${xyzText(L)}) мм, ${L.hemi} полушарие`).join('<br>') : 'нет'}</dd>
      <dt>Рабочая точка</dt><dd>порог ${fmt(wp)}${op && !s.card.demo ? ' — ' + op : ''}</dd>
      <dt>Модель</dt><dd>${esc(s.card.title)}${s.card.demo ? ' — ДЕМО' : ''}</dd>
      <dt>Ограничения</dt><dd>Возможны ложные находки и пропуски очагов.</dd>`;
    reportScreen(root, { tab: 'g2', study: s, kind: 'МРТ', rowsHTML: rows, thr: wp });
  },
};
