// Настройка под клинику: порог ЭЭГ подбирается по записям самой клиники, а не переносится с открытых данных.
// Два способа: по размеченным записям (индекс Юдена или чувствительность 80 %) и без разметки (медиана оценок).
// Порог хранится на сервере отдельно для каждой модели и применяется ко всем записям, в том числе уже оценённым.
import { $, $$, go, fmt, esc, state, refreshMeta, toast, C } from './ui.js';
import { api, waitJob } from './api.js';

export async function clinicScreen(root) {
  let d = await api.get('/api/g1/clinic');
  const act0 = d.threshold;
  const ui = { mode: act0.method === 'median' ? 'B' : 'A', pick: act0.method === 'sens80' ? 'sens80' : 'youden', busy: '' };
  const alive = () => location.hash.slice(1).startsWith('clinic');

  const dots = (lab, thr) => {
    const W = 760, X = v => 16 + v * (W - 32), recs = d.records.filter(r => r.label === 0 || r.label === 1);
    let s = `<svg viewBox="0 0 ${W} 132" role="img" aria-label="Оценки размеченных записей клиники и порог ${fmt(thr)}">`;
    s += `<rect x="${X(thr)}" y="20" width="${X(1) - X(thr)}" height="84" fill="rgba(255,255,255,.04)"/>`;
    s += `<text x="${X(0)}" y="42" font-size="11.5" fill="${C.ink2}">с эпилепсией · ${lab.n_pos}</text><text x="${X(0)}" y="88" font-size="11.5" fill="${C.ink2}">без эпилепсии · ${lab.n_neg}</text>`;
    recs.forEach((r, i) => { const y = (r.label ? 54 : 72) + (i % 3 - 1) * 4; s += (r.label ? `<circle cx="${X(r.score)}" cy="${y}" r="5" fill="${C.ink}"/>` : `<circle cx="${X(r.score)}" cy="${y + 26}" r="4.5" fill="none" stroke="${C.ink}" stroke-width="1.5"/>`).replace('/>', `><title>${esc(r.name)}: ${fmt(r.score)}</title></circle>`); });
    s += `<line x1="${X(thr)}" x2="${X(thr)}" y1="8" y2="104" stroke="${C.ai}" stroke-width="1.5"/><text x="${X(thr) + (thr > .8 ? -7 : 7)}" y="16" font-size="12" fill="${C.ai}" font-family="Consolas,monospace" text-anchor="${thr > .8 ? 'end' : 'start'}">порог ${fmt(thr)}</text>`;
    s += `<line x1="${X(0)}" x2="${X(1)}" y1="104" y2="104" stroke="${C.line2}"/>`;
    for (let v = 0; v <= 1.001; v += .25) s += `<line x1="${X(v)}" x2="${X(v)}" y1="104" y2="109" stroke="${C.line2}"/><text x="${X(v)}" y="124" font-size="11" fill="${C.ink3}" font-family="Consolas,monospace" text-anchor="middle">${fmt(v, 2)}</text>`;
    return s + '</svg>';
  };
  const hist = (u) => {
    const W = 760, X = v => 16 + v * (W - 32), bins = Array(20).fill(0); d.records.forEach(r => bins[Math.min(19, Math.floor(r.score * 20))]++);
    const mx = Math.max(...bins, 1), bw = (W - 32) / 20; let s = `<svg viewBox="0 0 ${W} 150" role="img" aria-label="Распределение оценок записей клиники, медиана ${fmt(u.threshold)}">`;
    bins.forEach((n, i) => { s += `<rect x="${16 + i * bw + 1}" y="${118 - n / mx * 96}" width="${bw - 2}" height="${n / mx * 96}" fill="#4a5560"/>`; });
    const x = X(u.threshold);
    s += `<line x1="${x}" x2="${x}" y1="8" y2="118" stroke="${C.ai}" stroke-width="1.5"/><text x="${x + 7}" y="16" font-size="12" fill="${C.ai}" font-family="Consolas,monospace">медиана ${fmt(u.threshold)} — порог</text><line x1="16" x2="${W - 16}" y1="118" y2="118" stroke="${C.line2}"/>`;
    for (let v = 0; v <= 1.001; v += .25) s += `<text x="${X(v)}" y="138" font-size="11" fill="${C.ink3}" font-family="Consolas,monospace" text-anchor="middle">${fmt(v, 2)}</text>`;
    return s + '</svg>';
  };
  const uploader = (kinds) => `<div class="upl">${kinds.map(([label, text]) => `<label class="btn"><input type="file" accept=".edf,.EDF" multiple hidden data-label="${label}">${text}</label>`).join('')}
    ${d.demo ? '<button class="btn" id="demoFill">Взять демо-набор клиники</button>' : ''}</div><div class="quiet" id="busy" role="status" style="margin:0;min-height:18px">${esc(ui.busy)}</div>`;
  const SRC = { study: 'из обычной работы', calib: 'для настройки', demo: 'демо' };
  const reclist = () => !d.records.length ? '' : `<h2>Записи клиники · ${d.records.length}</h2><p class="quiet" style="margin:0 0 4px">Оценки модели «${esc(d.model.title)}». Диагноз можно проставить позже — запись перейдёт в размеченные.</p>
    <div class="reclist"><table class="t"><thead><tr><th>Запись</th><th>Откуда</th><th>Оценка</th><th>Диагноз</th><th></th></tr></thead><tbody>
    ${d.records.map(r => `<tr><td>${esc(r.name)}${r.low_quality ? ' <span class="w">запись с замечаниями</span>' : ''}</td><td style="color:var(--ink3)">${SRC[r.source] || ''}</td><td class="num">${fmt(r.score)}</td>
      <td><select data-rid="${r.id}" aria-label="Диагноз для ${esc(r.name)}"><option value="">не указан</option><option value="1" ${r.label === 1 ? 'selected' : ''}>эпилепсия</option><option value="0" ${r.label === 0 ? 'selected' : ''}>без эпилепсии</option></select></td>
      <td><button class="btn ghost" style="height:24px;padding:0 6px" data-del="${r.id}" aria-label="Убрать запись ${esc(r.name)}">×</button></td></tr>`).join('')}</tbody></table></div>`;
  const ci = (c) => c ? `<div style="color:var(--ink3);font-size:11.5px">95 % ДИ ${fmt(c[0])}–${fmt(c[1])}</div>` : '';
  const partA = () => {
    const lab = d.proposal.labeled, up = uploader([['1', 'Добавить записи с эпилепсией…'], ['0', 'Добавить записи без эпилепсии…']]);
    if (!lab) return `<div class="panel"><h2 style="margin-top:0">Нужны размеченные записи</h2><p style="color:var(--ink2);margin:0;max-width:680px">Около ${d.need.labeled} записей с известным диагнозом, примерно поровну с эпилепсией и без. Каждая проходит ту же подготовку и ту же модель, что и в обычной работе.</p>${up}</div>`;
    const row = (k, name, why) => { const x = lab[k], on = ui.pick === k; return `<tr class="pick ${on ? 'on' : ''}" data-pick="${k}" tabindex="0" role="radio" aria-checked="${on}"><td><span class="radio"></span>${name}<div style="color:var(--ink3);font-size:11.5px;margin-left:20px">${why}</div></td><td class="num">${fmt(x.threshold)}</td>
      <td class="num">${x.tp} из ${x.n_pos}${ci(x.ci.sensitivity)}</td><td class="num">${x.tn} из ${x.n_neg}${ci(x.ci.specificity)}</td><td class="num">${fmt(x.precision)}</td><td class="num">${fmt(x.accuracy)}</td><td class="num">${fmt(x.balanced_accuracy)}</td></tr>`; };
    return `<div class="panel chart">${dots(lab, lab[ui.pick].threshold)}</div>
      <table class="t" style="margin-top:12px" role="radiogroup" aria-label="Точка на шкале"><thead><tr><th>Как выбрать порог</th><th>Порог</th><th>Чувствительность</th><th>Специфичность</th><th>Precision</th><th>Accuracy</th><th>Сбаланс. accuracy</th></tr></thead><tbody>
      ${row('youden', 'Индекс Юдена', 'лучшее равновесие пропусков и ложных тревог')}${row('sens80', 'Чувствительность 80 %', 'пропускать не больше одной записи с эпилепсией из пяти')}</tbody></table>
      <p class="quiet" style="max-width:760px">На ${lab.n} записях цифры грубые: доверительные интервалы широкие, а порог выбран по этим же записям, поэтому цифры оптимистичны.${lab.auc != null ? ` ROC AUC на этих записях: ${fmt(lab.auc)}.` : ''} Когда накопятся подтверждённые случаи, порог стоит уточнить.</p>${up}`;
  };
  const partB = () => {
    const u = d.proposal.unlabeled, up = uploader([['', 'Добавить записи клиники…']]);
    if (!u) return `<div class="panel"><h2 style="margin-top:0">Нужны записи клиники</h2><p style="color:var(--ink2);margin:0;max-width:680px">Выгрузите записи без диагнозов — лучше не меньше ${d.need.unlabeled}. Сюда же попадают записи, оценённые в обычной работе.</p>${up}</div>`;
    return `<div class="panel chart">${hist(u)}</div>
      <p class="quiet" style="max-width:760px">Порог по медиане отмечает примерно половину записей как «выше порога» (сейчас ${Math.round(u.share_above * 100)} % из ${u.n}). Годится, когда доля эпилепсии в потоке клиники близка к половине; иначе надёжнее способ с разметкой.</p>${up}`;
  };
  const value = () => ui.mode === 'A' ? d.proposal.labeled?.[ui.pick].threshold : d.proposal.unlabeled?.threshold;

  const render = () => {
    const a = ui.mode === 'A', v = value(), act = d.threshold;
    root.innerHTML = `<div class="pagecol"><div class="lab">ЭЭГ · порог оценки</div><h1>Настройка под клинику</h1>
        <p class="lead">Оценки модели в разных клиниках сдвинуты: порог с открытых данных переносится плохо. Поэтому порог подбирается по записям вашей клиники. Делается один раз при подключении; у каждой модели порог свой.</p>
        ${a ? partA() : partB()}${reclist()}</div>
      <aside class="right"><div class="scroll">
        <div class="blk"><div class="lab">Сейчас действует <span class="${act.clinic ? 'o' : 'w'}">${act.clinic ? 'порог клиники' : 'не настроен'}</span></div>
          <div class="big"><b style="color:var(--ink)">${fmt(act.value)}</b><div>${act.clinic ? 'подобран по записям клиники' : 'порог модели'}<small>${esc(act.source)}${act.date ? ', ' + esc(act.date) : ''}</small></div></div>
          <p class="quiet">Модель: ${esc(d.model.title)}.${act.clinic ? '' : ' Пока порог не настроен, вывод «выше / ниже» может быть неверным.'}</p>
          ${act.clinic ? '<button class="btn" id="resetThr" style="margin-top:8px">Вернуть порог модели</button>' : ''}</div>
        <div class="blk"><div class="lab">Способ</div>
          <button class="method ${a ? 'on' : ''}" data-mode="A" aria-pressed="${a}"><b>По размеченным записям</b><span>Около ${d.need.labeled} записей с известным диагнозом, поровну с эпилепсией и без. Точнее.</span></button>
          <button class="method ${a ? '' : 'on'}" data-mode="B" aria-pressed="${!a}"><b>Без разметки</b><span>Записи клиники без диагнозов; порог — медиана их оценок. Проще, но грубее.</span></button></div>
        ${d.proposal.warnings.length ? `<div class="blk"><div class="lab w">Обратите внимание</div>${d.proposal.warnings.map(w => `<p style="margin:0 0 6px">${esc(w)}</p>`).join('')}</div>` : ''}
        </div><div class="decide"><p class="hint">${v == null ? 'Добавьте записи — программа предложит порог.' : 'Порог применится ко всем записям, в том числе уже оценённым.'}</p>
          <button class="btn pri lg wide" id="saveThr" ${v == null ? 'disabled' : ''}>${v == null ? 'Сохранить порог' : 'Сохранить порог ' + fmt(v)}</button>
          ${d.records.length ? '<button class="btn ghost wide" id="clearAll" style="margin-top:6px">Удалить записи клиники и порог</button>' : ''}</div></aside>`;
    $$('[data-mode]', root).forEach(o => o.onclick = () => { ui.mode = o.dataset.mode; render(); });
    $$('[data-pick]', root).forEach(o => { o.onclick = () => { ui.pick = o.dataset.pick; render(); }; o.onkeydown = e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); o.click(); } }; });
    $$('select[data-rid]', root).forEach(sel => sel.onchange = async () => { d = await api.patch('/api/g1/clinic/records/' + sel.dataset.rid, { label: sel.value === '' ? null : +sel.value }); render(); });
    $$('[data-del]', root).forEach(b => b.onclick = async () => { d = await api.del('/api/g1/clinic/records/' + b.dataset.del); render(); });
    $$('input[type=file]', root).forEach(inp => inp.onchange = () => addFiles([...inp.files], inp.dataset.label));
    const df = $('#demoFill', root); if (df) df.onclick = async () => { df.disabled = true; try { d = await api.post('/api/g1/clinic/demo'); render(); } catch (e) { df.disabled = false; toast(e.message, true); } };
    const rs = $('#resetThr', root); if (rs) rs.onclick = async () => { d = await api.del('/api/g1/clinic'); await refreshMeta(); render(); toast('Действует порог модели'); };
    const ca = $('#clearAll', root); if (ca) ca.onclick = async () => { if (confirm(`Удалить ${d.records.length} записей клиники и порог для модели «${d.model.title}»? Отменить это нельзя.`)) { d = await api.del('/api/g1/clinic?records=1'); await refreshMeta(); render(); toast('Записи клиники и порог удалены'); } };
    $('#saveThr', root).onclick = async () => {
      try {
        await api.post('/api/g1/clinic/threshold', { method: ui.mode === 'A' ? ui.pick : 'median' });
        d = await api.get('/api/g1/clinic'); await refreshMeta(); render(); toast('Порог сохранён и применён ко всем записям');
      } catch (e) { toast(e.message, true); }
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
    if (!alive()) return;
    d = await api.get('/api/g1/clinic'); render();
  };
  render();
}
