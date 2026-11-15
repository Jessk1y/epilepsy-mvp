// Г1 · ЭЭГ: пошаговый режим (загрузка → проверка → расчёт → результат → отчёт) и «Настройка под клинику».
// Разметка и тексты — из макета Дарьи; вместо выдуманных значений — ответы сервера.
import { $, $$, go, fmt, esc, state, toast, checkCards, bindDrop, progressHTML, progressTick, headSVG, gauge, drawEEG, modelHelp, mmss } from './ui.js';
import { api, waitJob } from './api.js';
import { reportScreen } from './report.js';

const actual = (root) => () => document.body.contains(root);      // экран ещё открыт?

// ---------- фрагмент сигнала с листанием (экран проверки)
async function fragment(root, study) {
  const c = study.check, dur = c.summary.dur_s, first = (c.artifacts || [])[0];
  let t0 = first ? Math.max(0, Math.floor(first[0]) - 3) : Math.min(60, Math.max(0, dur - 10)), sig = null;
  const cv = $('#eeg', root);
  const load = async () => {
    sig = await api.get(`/api/g1/studies/${study.id}/signal?t0=${t0}&sec=10`);
    if (!document.body.contains(cv)) return;
    t0 = sig.t0; drawEEG(cv, sig, { auto: true }); $('#eegT', root).textContent = `${mmss(t0)}–${mmss(t0 + 10)} из ${mmss(dur)}`;
  };
  $('#eegPrev', root).onclick = () => { t0 = Math.max(0, t0 - 10); load(); };
  $('#eegNext', root).onclick = () => { t0 = Math.min(Math.max(0, dur - 10), t0 + 10); load(); };
  const nextArt = $('#eegArt', root);
  if (nextArt) nextArt.onclick = () => { const a = (c.artifacts || []).find(x => x[0] > t0 + 10) || c.artifacts[0]; t0 = Math.max(0, Math.floor(a[0]) - 3); load(); };
  const onResize = () => drawEEG(cv, sig, { auto: true }); addEventListener('resize', onResize);
  state.leave = () => removeEventListener('resize', onResize);
  await load();
}

export const G1 = {
  // ---------------------------------------------------------------- загрузка
  async upload(root) {
    const m = state.meta.g1.model, demo = state.meta.demo;
    root.innerHTML = `<div class="card"><h2>Загрузите рутинную запись ЭЭГ</h2><p class="sub">Одна запись 20–30 минут, 19 каналов системы 10–20, формат EDF.</p>
      <div class="drop" id="drop"><input type="file" accept=".edf,.EDF"><div class="big">Перетащите файл .edf сюда</div><div class="small">или</div><p><button class="btn pick">Выбрать файл…</button></p><div class="small" id="upst"></div></div>
      ${m.available ? '' : `<div class="note bad"><b>Модель «${esc(m.title)}» недоступна.</b> ${esc(m.reason)}. Запись можно загрузить и проверить, но оценку получить нельзя.</div>`}
      ${demo ? '<div class="note gray">Демо-режим: загруженная запись проходит настоящую подготовку и проверку, но оценку даёт <b>демо-модель</b> — клинического смысла у неё нет.</div>' : ''}
      <div class="actions"><span class="small">Файл обрабатывается на сервере клиники и никуда не передаётся.</span>
      <span>${demo ? '<button class="btn" id="demoBad">Демо: запись с замечаниями</button> <button class="btn pri" id="demoOk">Взять демо-запись →</button>' : ''}</span></div></div>`;
    const st = $('#upst', root);
    const done = (p) => p.then(s => go('g1/check/' + s.id)).catch(e => { st.innerHTML = `<span style="color:var(--bad)">${esc(e.message)}</span>`; });
    bindDrop($('#drop', root), files => { st.textContent = `Загрузка и проверка: ${files[0].name}…`; done(api.upload('/api/g1/studies', files[0])); });
    if (demo) {
      $('#demoOk', root).onclick = () => { st.textContent = 'Готовится демо-запись (синтетический сигнал)…'; done(api.post('/api/g1/studies/demo?kind=ok')); };
      $('#demoBad', root).onclick = () => { st.textContent = 'Готовится демо-запись (синтетический сигнал)…'; done(api.post('/api/g1/studies/demo?kind=bad')); };
    }
  },

  // ---------------------------------------------------------------- проверка записи
  async check(root, id) {
    if (!id) return go('g1/upload');
    const s = await api.get('/api/g1/studies/' + id), c = s.check;
    const sum = c.summary || {};
    if (!c.can_run) {
      root.innerHTML = `<div class="card"><h2>Проверка записи</h2><p class="sub">${esc(s.name)} · запись не подходит для оценки</p>${checkCards(c.items)}
        <div class="note bad"><b>Оценку получить нельзя.</b> Исправьте запись или загрузите другую.</div>${(c.notes || []).map(n => `<p class="small">${esc(n)}</p>`).join('')}
        <div class="actions"><button class="btn" onclick="go('g1/upload')">← Назад</button><span></span></div></div>`;
      return;
    }
    const low = c.low_quality;
    root.innerHTML = `<div class="card"><h2>Проверка записи</h2><p class="sub">${esc(s.name)} · проверьте, подходит ли запись для оценки</p>
      <div class="row"><div class="col">${checkCards(c.items)}</div><div style="text-align:center">${headSVG(c.flat)}<div class="small">Схема каналов</div></div></div>
      <h3 style="margin-top:18px">Фрагмент сигнала, 10 с</h3>
      <div class="eegnav"><button class="btn" id="eegPrev">◀</button><span id="eegT"></span><button class="btn" id="eegNext">▶</button>
        ${c.artifacts.length ? '<button class="btn link" id="eegArt">к следующему артефакту</button>' : ''}<span>средний референт · ${sum.sfreq} → 200 Гц · 0,5–70 Гц</span></div>
      <canvas class="eeg" id="eeg"></canvas>
      <p class="small">Оранжевым — участок с артефактом (показатель качества записи, определяется приближённо). Модель оценивает запись целиком и <b>не указывает места разрядов</b>.</p>
      ${(c.notes || []).map(n => `<div class="note gray">${esc(n)}</div>`).join('')}
      ${low ? '<div class="note warn"><b>Запись с замечаниями.</b> Оценку можно получить, но в отчёте будет отмечено низкое качество входа. Лучше повторить запись или загрузить другую.</div>' : ''}
      <div class="actions"><button class="btn" onclick="go('g1/upload')">← Назад</button><button class="btn pri" onclick="go('g1/run/${s.id}')">${low ? 'Всё равно рассчитать' : 'Запустить оценку'} →</button></div></div>`;
    await fragment(root, s);
  },

  // ---------------------------------------------------------------- расчёт
  async run(root, id) {
    if (!id) return go('g1/upload');
    const card = state.meta.g1.model.card;
    const job = await api.post(`/api/g1/studies/${id}/run`);
    root.innerHTML = `<div class="card"><h2>Идёт оценка записи</h2><p class="sub">${esc(card.eta)}</p>${progressHTML(job.stages)}<div id="runErr"></div></div>`;
    try {
      const j = await waitJob(job.id, x => progressTick(root, x), actual(root));
      if (j) setTimeout(() => go('g1/result/' + id), 300);
    } catch (e) {
      $('#runErr', root).innerHTML = `<div class="note bad"><b>Расчёт не выполнен.</b> ${esc(e.message)}</div><div class="actions"><button class="btn" onclick="go('g1/check/${id}')">← К проверке</button><span></span></div>`;
    }
  },

  // ---------------------------------------------------------------- результат
  async result(root, id) {
    if (!id) return go('g1/upload');
    const s = await api.get('/api/g1/studies/' + id), r = s.result;
    if (!r) return go('g1/check/' + id);
    const t = r.threshold, above = r.above, c = s.check, card = r.model_card;
    const spread = r.spread ? `Устойчивость: оценки по разным частям записи — от ${fmt(r.spread[0])} до ${fmt(r.spread[1])}. ${r.spread[1] - r.spread[0] < 0.1 ? 'Разброс мал, вывод не зависит от отдельного фрагмента.' : 'Разброс заметный: оценка зависит от того, какая часть записи взята.'}${(r.spread[0] < t.value) !== (r.spread[1] < t.value) ? ' <b>Части записи лежат по разные стороны порога.</b>' : ''}` : '';
    root.innerHTML = `<div class="card"><h2>Результат оценки</h2><p class="sub">${esc(s.name)} · оценка относится ко всей записи</p>
      <div class="row" style="align-items:center"><div style="flex:none">${gauge(r.score, t.value)}</div>
      <div class="col"><span class="pill ${above ? 'warn' : 'ok'}">${above ? 'Выше порога' : 'Ниже порога'}${t.clinic ? ' клиники' : ''}</span>
       <div class="verdict">${esc(r.verdict)}</div>
       <div>Оценка модели: <b>${fmt(r.score)}</b> из 1 · порог${t.clinic ? ' клиники' : ''}: <b>${fmt(t.value)}</b></div>
       <div class="small">${t.clinic ? 'Порог настроен ' + esc(t.source) + ', ' + esc(t.date) + '.' : 'Порог ' + esc(t.source) + '.'} <a href="#g1/clinic">${t.clinic ? 'Изменить' : 'Настроить под клинику'}</a></div>
       ${spread ? `<div class="note gray" style="margin-top:12px">${spread}</div>` : ''}</div></div>
      ${!t.clinic ? '<div class="note warn"><b>Порог не настроен под вашу клинику.</b> Порог с открытых данных в другую клинику переносится плохо: вывод «выше / ниже» может быть неверным. Настройте порог по записям клиники.</div>' : ''}
      ${c.low_quality ? `<div class="note warn">Качество входа ниже нормы (${esc((c.low_reasons || []).join(', '))}) — доверие к оценке снижено.</div>` : ''}
      ${card.demo ? '<div class="note warn"><b>ДЕМО-модель.</b> Оценка посчитана по простой формуле, а не нейросетью, и не говорит о диагнозе.</div>' : ''}
      <div class="note info"><b>Это второе мнение, решение за врачом.</b> Модель не показывает, где в записи разряды: она оценивает запись целиком, в том числе когда явных разрядов нет.</div>
      ${modelHelp(card, r.model_title)}
      <div class="actions"><span><button class="btn" onclick="go('g1/upload')">Новая запись</button> <button class="btn" onclick="go('ws/g1/${s.id}')">Открыть в рабочем месте</button></span><button class="btn pri" onclick="go('g1/report/${s.id}')">Сформировать отчёт →</button></div></div>`;
  },

  // ---------------------------------------------------------------- отчёт
  async report(root, id) {
    if (!id) return go('g1/upload');
    const s = await api.get('/api/g1/studies/' + id), r = s.result;
    if (!r) return go('g1/check/' + id);
    const c = s.check, t = r.threshold;
    const rows = `<dt>Длительность / каналы</dt><dd>${esc(c.summary.dur_text)} · ${c.summary.n_good} из 19</dd><dt>Артефакты</dt><dd>≈ ${c.summary.art_pct} % записи</dd>
      <dt>Оценка модели</dt><dd><b>${fmt(r.score)}</b> (порог${t.clinic ? ' клиники' : ''} ${fmt(t.value)}) — ${r.above ? 'выше' : 'ниже'} порога</dd><dt>Вывод модели</dt><dd>${esc(r.verdict)}</dd>
      ${c.low_quality ? `<dt>Качество входа</dt><dd>ниже нормы: ${esc((c.low_reasons || []).join(', '))}</dd>` : ''}
      <dt>Модель</dt><dd>${esc(r.model_title)}${r.model_card.demo ? ' — ДЕМО' : ''}</dd>
      <dt>Ограничения</dt><dd>Модель не локализует события. Оценка — вспомогательная.</dd>`;
    reportScreen(root, { tab: 'g1', study: s, kind: 'ЭЭГ', rowsHTML: rows });
  },

  // ---------------------------------------------------------------- настройка под клинику
  async clinic(root) {
    let d = await api.get('/api/g1/clinic');
    const act0 = d.threshold;
    const ui = { mode: act0.method === 'median' ? 'B' : 'A', pick: act0.method === 'sens80' ? 'sens80' : 'youden', busy: '' };

    const dots = (lab, thr) => {
      const X = v => 20 + v * 600, recs = d.records.filter(r => r.label === 0 || r.label === 1);
      let s = '<svg viewBox="0 0 640 120" style="width:100%"><line x1="20" y1="60" x2="620" y2="60" stroke="#ccd"/>';
      for (let v = 0; v <= 1; v += .25) s += `<text x="${X(v)}" y="112" font-size="11" fill="#789" text-anchor="middle">${fmt(v, 2)}</text>`;
      recs.forEach((r, i) => { s += `<circle cx="${X(r.score)}" cy="${(r.label ? 40 : 80) + (i % 3 - 1) * 5}" r="7" fill="${r.label ? '#d9822b' : '#2f80c2'}" opacity=".85"><title>${esc(r.name)}: ${fmt(r.score)}</title></circle>`; });
      s += `<line x1="${X(thr)}" y1="15" x2="${X(thr)}" y2="98" stroke="#1d2733" stroke-width="2" stroke-dasharray="5 3"/><text x="${X(thr) + (thr > .8 ? -6 : 6)}" y="18" font-size="12" text-anchor="${thr > .8 ? 'end' : 'start'}">порог ${fmt(thr)}</text>
        <text x="22" y="30" font-size="11" fill="#d9822b">● эпилепсия (${lab.n_pos})</text><text x="22" y="100" font-size="11" fill="#2f80c2">● без эпилепсии (${lab.n_neg})</text></svg>`;
      return s;
    };
    const hist = (u) => {
      const bins = Array(20).fill(0); d.records.forEach(r => bins[Math.min(19, Math.floor(r.score * 20))]++);
      const mx = Math.max(...bins, 1); let s = '<svg viewBox="0 0 640 150" style="width:100%">';
      bins.forEach((c, i) => { s += `<rect x="${20 + i * 30}" y="${120 - c / mx * 100}" width="26" height="${c / mx * 100}" fill="#8fb3e0"/>`; });
      const x = 20 + u.threshold * 600;
      return s + `<line x1="${x}" y1="10" x2="${x}" y2="122" stroke="#1d2733" stroke-width="2" stroke-dasharray="5 3"/><text x="${x + 6}" y="16" font-size="12">медиана ${fmt(u.threshold)} = порог</text><text x="20" y="142" font-size="11" fill="#789">0</text><text x="620" y="142" font-size="11" fill="#789" text-anchor="end">1</text></svg>`;
    };
    const uploader = (kinds) => `<div class="upl">${kinds.map(([label, text]) => `<label class="btn"><input type="file" accept=".edf,.EDF" multiple hidden data-label="${label}">${text}</label>`).join('')}
      ${d.demo ? '<button class="btn" id="demoFill">Загрузить демо-набор клиники</button>' : ''}</div><div class="small" id="busy">${esc(ui.busy)}</div>`;
    const reclist = () => !d.records.length ? '' : `<details><summary class="small">Записи клиники: ${d.records.length} (оценки модели «${esc(d.model.title)}»)</summary><div class="reclist"><table><tr><th>Запись</th><th>Оценка</th><th>Диагноз</th><th></th></tr>
      ${d.records.map(r => `<tr><td>${esc(r.name)}${r.low_quality ? ' ⚠' : ''}<div class="small">${{ study: 'из обычной работы', calib: 'для настройки', demo: 'демо' }[r.source] || ''}</div></td><td>${fmt(r.score)}</td>
        <td><select data-rid="${r.id}"><option value="">не указан</option><option value="1" ${r.label === 1 ? 'selected' : ''}>эпилепсия</option><option value="0" ${r.label === 0 ? 'selected' : ''}>без эпилепсии</option></select></td>
        <td><button class="x" data-del="${r.id}" title="Убрать запись">✕</button></td></tr>`).join('')}</table></div></details>`;
    const ci = (c) => c ? ` <span class="small">(95 % ДИ ${fmt(c[0])}–${fmt(c[1])})</span>` : '';
    const partA = () => {
      const lab = d.proposal.labeled;
      if (!lab) return `<h3>Размеченные записи клиники</h3><p class="small">Нужно около ${d.need.labeled} записей с известным диагнозом: примерно поровну с эпилепсией и без. Каждая запись проходит ту же подготовку и ту же модель, что и в обычной работе.</p>
        ${uploader([['1', 'Добавить записи с эпилепсией…'], ['0', 'Добавить записи без эпилепсии…']])}${reclist()}`;
      const row = (k, name) => { const x = lab[k]; return `<tr class="click ${ui.pick === k ? 'sel' : ''}" data-pick="${k}"><td>${ui.pick === k ? '◉' : '○'} ${name}</td><td>${fmt(x.threshold)}</td><td>${x.tp} из ${x.n_pos}${ci(x.ci.sensitivity)}</td><td>${x.tn} из ${x.n_neg}${ci(x.ci.specificity)}</td></tr>`; };
      return `<h3>Загружено ${lab.n} размеченных записей${d.demo ? ' · демо' : ''}</h3>${dots(lab, lab[ui.pick].threshold)}
        <table><tr><th>Точка</th><th>Порог</th><th>Чувствительность</th><th>Специфичность</th></tr>${row('youden', 'Индекс Юдена')}${row('sens80', 'Чувствительность 80 %')}</table>
        <p class="small">На ${lab.n} записях оценки грубые: доверительные интервалы широкие, а порог выбран по этим же записям — цифры оптимистичны. После накопления подтверждённых случаев порог можно уточнить.${lab.auc != null ? ` ROC AUC на этих записях: ${fmt(lab.auc)}.` : ''}</p>
        ${uploader([['1', 'Добавить записи с эпилепсией…'], ['0', 'Добавить записи без эпилепсии…']])}${reclist()}`;
    };
    const partB = () => {
      const u = d.proposal.unlabeled;
      if (!u) return `<h3>Записи клиники без разметки</h3><p class="small">Выгрузите записи клиники без диагнозов (лучше не меньше ${d.need.unlabeled}). Сюда же попадают записи, оценённые в обычной работе.</p>${uploader([['', 'Добавить записи клиники…']])}`;
      return `<h3>Оценки ${u.n} записей клиники без разметки${d.demo ? ' · демо' : ''}</h3>${hist(u)}
        <p class="small">Порог по медиане отмечает примерно половину записей как «выше порога». Подходит, когда доля эпилепсии в потоке клиники близка к половине; иначе лучше вариант с разметкой.</p>
        ${uploader([['', 'Добавить записи клиники…']])}${reclist()}`;
    };
    const value = () => ui.mode === 'A' ? d.proposal.labeled?.[ui.pick].threshold : d.proposal.unlabeled?.threshold;

    const render = () => {
      const a = ui.mode === 'A', v = value(), act = d.threshold;
      root.innerHTML = `<div class="card"><h2>Настройка под клинику</h2><p class="sub">Порог переносится не «как есть» с открытых данных, а подбирается по записям вашей клиники. Делается один раз при подключении.</p>
        <div class="note ${act.clinic ? 'info' : 'warn'}">Сейчас действует порог <b>${fmt(act.value)}</b> — ${esc(act.source)}${act.date ? ', ' + esc(act.date) : ''}. Модель: «${esc(d.model.title)}»; для другой модели порог настраивается отдельно.
          ${act.clinic ? ' <button class="btn link" id="resetThr">Сбросить</button>' : ''}</div>
        <div class="row"><div class="opt ${a ? 'on' : ''}" data-mode="A"><h3>◉ По 20 размеченным записям</h3><p class="small">10 с эпилепсией и 10 без, диагнозы известны. Порог — по индексу Юдена; дополнительно точка с чувствительностью 80 %.</p></div>
        <div class="opt ${a ? '' : 'on'}" data-mode="B"><h3>◉ Без разметки</h3><p class="small">Достаточно выгрузить записи клиники без диагнозов. Порог — медиана оценок этих записей.</p></div></div>
        <div class="card" style="margin-top:16px;background:#fbfcfe">${a ? partA() : partB()}${d.proposal.warnings.map(w => `<div class="note warn">${esc(w)}</div>`).join('')}</div>
        <div class="actions"><span><button class="btn" onclick="history.back()">← Назад</button> ${d.records.length ? '<button class="btn link" id="clearAll">Удалить записи клиники</button>' : ''}</span>
        <button class="btn pri" id="saveThr" ${v == null ? 'disabled' : ''}>${v == null ? 'Сохранить порог' : 'Сохранить порог ' + fmt(v)}</button></div></div>`;
      $$('.opt', root).forEach(o => o.onclick = () => { ui.mode = o.dataset.mode; render(); });
      $$('[data-pick]', root).forEach(o => o.onclick = () => { ui.pick = o.dataset.pick; render(); });
      $$('select[data-rid]', root).forEach(sel => sel.onchange = async () => { d = await api.patch('/api/g1/clinic/records/' + sel.dataset.rid, { label: sel.value === '' ? null : +sel.value }); render(); });
      $$('[data-del]', root).forEach(b => b.onclick = async () => { d = await api.del('/api/g1/clinic/records/' + b.dataset.del); render(); });
      $$('input[type=file]', root).forEach(inp => inp.onchange = () => addFiles([...inp.files], inp.dataset.label));
      const df = $('#demoFill', root); if (df) df.onclick = async () => { d = await api.post('/api/g1/clinic/demo'); render(); };
      const rs = $('#resetThr', root); if (rs) rs.onclick = async () => { d = await api.del('/api/g1/clinic'); state.meta = await api.get('/api/meta'); render(); };
      const ca = $('#clearAll', root); if (ca) ca.onclick = async () => { if (confirm('Удалить все записи клиники и порог для этой модели?')) { d = await api.del('/api/g1/clinic?records=1'); state.meta = await api.get('/api/meta'); render(); } };
      $('#saveThr', root).onclick = async () => {
        try {
          await api.post('/api/g1/clinic/threshold', { method: ui.mode === 'A' ? ui.pick : 'median' });
          d = await api.get('/api/g1/clinic'); state.meta = await api.get('/api/meta'); render(); toast('Порог сохранён и применяется ко всем записям');
        } catch (e) { toast(e.message); }
      };
    };
    // записи клиники: по одной — загрузка, подготовка, оценка моделью
    const addFiles = async (files, label) => {
      const errs = [];
      for (let i = 0; i < files.length; i++) {
        ui.busy = `Обработка ${i + 1} из ${files.length}: ${files[i].name}…`; const b = $('#busy', root); if (b) b.textContent = ui.busy;
        try { const j = await api.upload('/api/g1/clinic/records?label=' + label, files[i]); await waitJob(j.id); } catch (e) { errs.push(`${files[i].name}: ${e.message}`); }
      }
      ui.busy = errs.length ? 'Не обработаны — ' + errs.join('; ') : '';
      if (!document.body.contains(root)) return;
      d = await api.get('/api/g1/clinic'); render();
    };
    render();
  },
};
