// Рабочее место ЭЭГ: одна страница на все состояния записи — проверка входа, расчёт, оценка, решение врача.
// В центре — подготовленный сигнал (листается кусками с сервера), справа — что говорит программа и что решает врач.
// Модель оценивает запись целиком и мест разрядов не указывает: на сигнале отмечаются только артефакты (показатель качества).
import { $, $$, go, fmt, esc, state, keys, tabs, drawEEG, mmss, progressHTML, progressTick, checkItems, headSVG, scaleSVG, toast, errorBox, C } from './ui.js';
import { api, waitJob } from './api.js';

const S = { mont: 'avg', uvmm: 0, sec: 10, filt: '0.5,70' };   // uvmm = 0 — усиление подбирается автоматически

export async function eegScreen(root, id) {
  if (!id) return go('list');
  let s = await api.get('/api/g1/studies/' + id);
  tabs.open('eeg', id, s.name);
  const c = s.check, sum = c.summary || {}, dur = sum.dur_s || 0, arts = c.artifacts || [];
  const model = state.meta.g1.model;
  let t0 = arts.length && !s.result ? Math.max(0, Math.floor(arts[0][0]) - 3) : 0, sig = null, timer = null, records = null, running = false;
  const alive = () => document.body.contains(root.firstElementChild) && location.hash.includes(id);

  // ---- запись не прошла проверку: сигнала нет, оценки не будет
  if (!c.can_run) {
    const bad = c.items.filter(i => i.status === 'bad');
    root.innerHTML = `<div class="center"><div class="empty err"><h2>Оценку получить нельзя</h2>
      <p>${esc(s.name)} не прошла проверку входа${bad.length ? ': ' + esc(bad.map(i => `${i.title.toLowerCase()} — ${i.note || i.value}`).join('; ')) : ''}.</p>
      <div class="row"><button class="btn pri" onclick="go('new')">Загрузить другую запись</button><button class="btn" onclick="go('list')">К списку</button></div></div></div>
      <aside class="right"><div class="scroll"><div class="blk"><div class="lab">Проверка записи <span class="x">не пройдена</span></div>${checkItems(c.items)}
      ${(c.notes || []).map(n => `<p class="quiet">${esc(n)}</p>`).join('')}</div></div></aside>`;
    return;
  }

  // ---- правая панель
  const demoNote = (card) => card.demo ? '<div class="alert demo-a"><b>ДЕМО.</b> Оценку даёт формула, а не нейросеть: клинического смысла у числа нет.</div>' : '';
  const checkBlock = (title) => `<div class="blk"><div class="lab">${title} <span class="${c.low_quality ? 'w' : 'o'}">${c.low_quality ? 'есть замечания' : 'без замечаний'}</span></div>
    ${checkItems(c.items)}${(c.notes || []).map(n => `<p class="quiet">${esc(n)}</p>`).join('')}</div>`;
  const right = () => {
    const r = s.result;
    if (!r) {                                               // проверка пройдена, оценки ещё нет
      return `<aside class="right">${demoNote(model.card)}<div class="scroll">
        <div class="blk"><h3>${c.low_quality ? 'Запись с замечаниями' : 'Запись подходит для оценки'}</h3>
          <p>${c.low_quality ? `Оценку получить можно, но доверие к ней будет ниже: ${esc((c.low_reasons || []).join(', '))}. Если есть возможность — лучше повторить запись.` : 'Проверка входа пройдена. Просмотрите сигнал и запустите оценку.'}</p></div>
        ${checkBlock('Проверка записи')}
        <div class="blk"><div class="lab">Каналы 10–20 <span>${sum.n_good} из 19 с сигналом</span></div>${headSVG(c.flat)}</div>
        </div><div class="decide" id="runBox">${running ? '' : (model.available
          ? `<div class="lab">Следующий шаг</div><p class="hint">${esc(model.card.eta || '')}</p><button class="btn pri lg wide" id="runBtn">${c.low_quality ? 'Всё равно рассчитать оценку' : 'Рассчитать оценку'} <kbd>⏎</kbd></button>`
          : `<div class="lab">Оценка недоступна</div><p class="hint">Модель «${esc(model.title)}» недоступна: ${esc(model.reason)}.</p><button class="btn lg wide" disabled>Рассчитать оценку</button>`)}</div></aside>`;
    }
    const t = r.threshold, d = s.decision || {}, det = t.details || {}, lab = (records || []).filter(x => x.label === 0 || x.label === 1);
    const item = k => c.items.find(i => i.key === k) || { status: 'ok' }, cls = st => ({ warn: 'w', bad: 'x' }[st] || '');
    const tight = r.spread && r.spread[1] - r.spread[0] < 0.1, split = r.spread && (r.spread[0] < t.value) !== (r.spread[1] < t.value);
    let about = '';
    if (!t.clinic) about = `<p class="quiet w">Порог не настроен под клинику: действует порог модели. В другую клинику он переносится плохо, поэтому вывод «выше / ниже» может быть неверным. <a href="#clinic">Настроить порог</a></p>`;
    else if (lab.length && det.tp != null) about = `<p class="quiet">Точки — записи клиники с известным диагнозом. С эпилепсией (закрашенные): ${det.tp} из ${det.tp + det.fn} выше порога. Без эпилепсии (пустые): ${det.tn} из ${det.tn + det.fp} ниже. <a href="#clinic">Порог клиники</a></p>`;
    else about = `<p class="quiet">Порог клиники: ${esc(t.source)}${t.date ? ', ' + esc(t.date) : ''}. Штрихи — записи клиники без разметки. <a href="#clinic">Порог клиники</a></p>`;
    return `<aside class="right">${demoNote(r.model_card)}<div class="scroll">
      <div class="blk"><div class="lab">Оценка программы <span>вся запись</span></div>
        <div class="big"><b>${fmt(r.score)}</b><div>${r.above ? 'выше' : 'ниже'} порога${t.clinic ? ' клиники' : ''}<small>порог ${fmt(t.value)} · ${esc(r.verdict.toLowerCase())}</small></div></div>
        <div style="margin-top:12px">${scaleSVG({ score: r.score, thr: t.value, spread: r.spread, records: t.clinic ? (records || []) : [] })}</div>${about}</div>
      <div class="blk"><div class="lab">Доверие к оценке <span class="${c.low_quality || split ? 'w' : 'o'}">${c.low_quality || split ? 'снижено' : 'обычное'}</span></div>
        <div class="rows"><span>Длительность</span><span class="${cls(item('duration').status)}">${mmss(dur)}</span>
          <span>Каналы 10–20</span><span class="${cls(item('channels').status)}">${sum.n_good}/19${c.flat.length ? ' · ' + esc(c.flat.join(', ')) : ''}</span>
          <span>Артефакты</span><span class="${cls(item('artifacts').status)}">≈${sum.art_pct} %</span>
          ${r.spread ? `<span>Оценки частей записи</span><span class="${split ? 'w' : ''}">${fmt(r.spread[0])}–${fmt(r.spread[1])}</span>` : ''}
          <span>Частота</span><span>${sum.sfreq}→200 Гц</span></div>
        ${r.spread ? `<p class="quiet">${split ? 'Части записи лежат по разные стороны порога: вывод зависит от того, какой фрагмент взят.' : tight ? 'Части записи согласуются: вывод не зависит от отдельного фрагмента.' : 'Разброс между частями записи заметный.'}</p>` : ''}</div>
      <div class="blk"><div class="lab">Модель <span><a href="#models">подробнее</a></span></div><div>${esc(r.model_title)}</div>
        <p class="quiet">Оценивает запись целиком и не указывает места разрядов. Диагноз не ставит.</p></div>
      </div><div class="decide"><div class="lab">Решение врача</div>
        <div class="pair"><button class="btn ${d.agree === true ? 'on' : ''}" data-agree="1" aria-pressed="${d.agree === true}">Согласен <kbd>С</kbd></button><button class="btn ${d.agree === false ? 'on' : ''}" data-agree="0" aria-pressed="${d.agree === false}">Не согласен <kbd>Н</kbd></button></div>
        <button class="btn pri lg wide" id="toRep">${s.report ? 'Заключение (отчёт уже выдан)' : 'Заключение'} <kbd>⏎</kbd></button></div></aside>`;
  };

  // полоса под шкалой времени: оценка программы по частям записи (ступеньками) и порог
  const parts = () => {
    const r = s.result; if (!r || !r.parts.length) return '';
    const y = v => (30 - v * 30).toFixed(1); let d = '';
    r.parts.forEach((p, i) => { d += `${i ? 'L' : 'M'}${(p.t0 / dur * 1000).toFixed(1)},${y(p.score)} L${(p.t1 / dur * 1000).toFixed(1)},${y(p.score)} `; });
    return `<div class="parts" title="Оценка программы по частям записи"><svg viewBox="0 0 1000 30" preserveAspectRatio="none" aria-hidden="true"><line x1="0" x2="1000" y1="${y(r.threshold.value)}" y2="${y(r.threshold.value)}" stroke="${C.ink2}" stroke-dasharray="5 4" vector-effect="non-scaling-stroke"/><path d="${d}" stroke="${C.ai}" fill="none" stroke-width="2" vector-effect="non-scaling-stroke"/></svg></div>`;
  };
  const sel = (id2, label, opts, cur) => `<select class="sel" id="${id2}" aria-label="${label}">${opts.map(([v, t]) => `<option value="${v}" ${String(v) === String(cur) ? 'selected' : ''}>${t}</option>`).join('')}</select>`;
  const nz = state.meta.g1.notch_hz;

  const render = () => {
    const r = s.result, ticks = [0, 1, 2, 3, 4, 5, 6].map(i => `<span>${mmss(dur * i / 6)}</span>`).join('');
    root.innerHTML = `<div class="center"><div class="tool">
        ${sel('mont', 'Монтаж', [['avg', 'Средний референт'], ['bip', 'Продольный биполярный']], S.mont)}
        ${sel('uvmm', 'Усиление', [[0, 'Усиление авто'], [5, '5 мкВ/мм'], [7, '7 мкВ/мм'], [10, '10 мкВ/мм'], [15, '15 мкВ/мм'], [20, '20 мкВ/мм']], S.uvmm)}
        ${sel('secs', 'Развёртка', [[10, '10 с на экран'], [20, '20 с на экран']], S.sec)}
        ${sel('filt', 'Фильтры', [['0.5,70', `0,5–70 Гц · режекция ${nz}`], ['1,35', `1–35 Гц · режекция ${nz}`], ['0,0', 'Как у модели: 0,1–75 Гц']], S.filt)}
        <span class="gap"></span><button class="btn ib" id="prev" aria-label="Назад по записи">◀</button><span class="m t" id="tm"></span><button class="btn ib" id="next" aria-label="Вперёд по записи">▶</button>
        ${arts.length ? '<button class="btn" id="toArt" style="height:26px">К артефакту</button>' : ''}
        <span class="note2"><kbd>←</kbd><kbd>→</kbd> листать</span></div>
      <div class="sigwrap"><canvas id="eeg" role="img" aria-label="Сигнал ЭЭГ, ${esc(s.name)}"></canvas></div>
      <div class="time"><div class="hd"><span>Вся запись · ${mmss(dur)}</span><span><i style="background:rgba(255,143,102,.55)"></i>артефакты, ≈${sum.art_pct} % — определяются приближённо</span>
          ${r ? `<span><i style="background:${C.ai};height:2px;vertical-align:3px"></i>оценка по частям записи</span><span><i style="border-top:1px dashed ${C.ink2};height:0;vertical-align:3px"></i>порог ${fmt(r.threshold.value)}</span>` : ''}
          <span class="say">Мест разрядов программа не указывает</span></div>
        <div class="tl" id="tl" title="Щелчок — перейти к этому месту записи">${arts.map(([a, b]) => `<i style="left:${a / dur * 100}%;width:${(b - a) / dur * 100}%"></i>`).join('')}<b id="pg"></b></div>
        ${parts()}<div class="tick">${ticks}</div></div></div>` + right();
    $('#mont', root).onchange = e => { S.mont = e.target.value; load(); };
    $('#uvmm', root).onchange = e => { S.uvmm = +e.target.value; draw(); };
    $('#secs', root).onchange = e => { S.sec = +e.target.value; load(); };
    $('#filt', root).onchange = e => { S.filt = e.target.value; load(); };
    $('#prev', root).onclick = () => page(-1); $('#next', root).onclick = () => page(1);
    const ta = $('#toArt', root); if (ta) ta.onclick = nextArt;
    $('#tl', root).onclick = e => { const b = e.currentTarget.getBoundingClientRect(); t0 = Math.round((e.clientX - b.left) / b.width * dur - S.sec / 2); load(); };
    $$('[data-agree]', root).forEach(b => b.onclick = () => decide(b.dataset.agree === '1'));
    const rb = $('#runBtn', root); if (rb) rb.onclick = run;
    const tr = $('#toRep', root); if (tr) tr.onclick = () => go('report/eeg/' + id);
    if (sig) draw(); else load();
  };
  const draw = () => {
    const cv = $('#eeg', root); if (!cv || !sig) return;
    drawEEG(cv, sig, { uvmm: S.uvmm, auto: !S.uvmm, labelW: S.mont === 'bip' ? 58 : 46 });
    $('#tm', root).textContent = `${mmss(t0)}–${mmss(Math.min(dur, t0 + S.sec))}`;
    const pg = $('#pg', root); pg.style.left = (t0 / dur * 100) + '%'; pg.style.width = Math.max(.6, S.sec / dur * 100) + '%';
  };
  const load = () => {                                   // запрос куска сигнала; частые нажатия объединяются
    t0 = Math.max(0, Math.min(Math.max(0, dur - S.sec), t0));
    clearTimeout(timer);
    timer = setTimeout(async () => {
      const [hp, lp] = S.filt.split(','), want = t0;
      try {
        const x = await api.get(`/api/g1/studies/${id}/signal?t0=${t0}&sec=${S.sec}&montage=${S.mont}&hp=${hp}&lp=${lp}`);
        if (want !== t0 || !alive()) return;
        sig = x; draw();
      } catch (e) { if (alive()) toast('Сигнал не загрузился: ' + e.message, true); }
    }, 30);
  };
  const page = (d) => { t0 += d * S.sec; load(); };
  const nextArt = () => { const a = arts.find(x => x[0] > t0 + S.sec) || arts[0]; if (a) { t0 = Math.max(0, Math.floor(a[0]) - 3); load(); } };
  const decide = async (agree) => {
    if (!s.result) return;
    try { s.decision = await api.post(`/api/g1/studies/${id}/decision`, { agree }); render(); toast(agree ? 'Отмечено: согласен с оценкой' : 'Отмечено: не согласен с оценкой'); }
    catch (e) { toast('Решение не сохранилось: ' + e.message, true); }
  };
  const run = async () => {
    if (running || s.result || !model.available) return;
    running = true;
    const box = $('#runBox', root);
    try {
      const job = await api.post(`/api/g1/studies/${id}/run`);
      box.innerHTML = `<div class="lab">Идёт расчёт</div>${progressHTML(job.stages)}`;
      const j = await waitJob(job.id, x => { const b = $('#runBox', root); if (b) progressTick(b, x); }, alive);
      running = false; if (!j) return;
      s = await api.get('/api/g1/studies/' + id); await loadRecords(); t0 = 0; sig = null; render();
    } catch (e) {
      running = false;
      if ($('#runBox', root)) $('#runBox', root).innerHTML = `<div class="lab x">Расчёт не выполнен</div><p class="hint">${esc(e.message)}</p><button class="btn lg wide" id="runBtn">Повторить расчёт</button>`;
      const rb = $('#runBtn', root); if (rb) rb.onclick = run;
    }
  };
  // записи клиники для шкалы: как эта запись выглядит среди записей клиники с известным диагнозом
  const loadRecords = async () => { try { records = (await api.get('/api/g1/clinic')).records.filter(x => x.study_id !== id); } catch (e) { records = []; } };

  if (s.result) await loadRecords();
  render();
  const off = keys({
    ArrowRight: () => page(1), ArrowLeft: () => page(-1),
    KeyC: () => decide(true), KeyY: () => decide(false),
    Enter: () => s.result ? go('report/eeg/' + id) : run(),
  });
  const onResize = () => draw(); addEventListener('resize', onResize);
  state.leave = () => { off(); removeEventListener('resize', onResize); clearTimeout(timer); };
}
