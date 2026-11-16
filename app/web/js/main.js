// Каркас: шапка с вкладками открытых исследований, строка состояния, маршруты.
// Адрес после # хранит экран и номер исследования — страницу можно обновлять и пересылать ссылку внутри клиники.
import { $, $$, go, state, esc, fmt, tabs, loading, errorBox } from './ui.js';
import { api } from './api.js';
import { listScreen, newScreen } from './list.js';
import { eegScreen } from './eeg.js';
import { mriScreen } from './mriws.js';
import { reportScreen } from './report.js';
import { clinicScreen } from './clinic.js';
import { modelsScreen } from './models.js';

const ROUTES = { list: listScreen, new: newScreen, eeg: eegScreen, mri: mriScreen, report: reportScreen, clinic: clinicScreen, models: modelsScreen };


function shell() {
  const m = state.meta, h = decodeURIComponent(location.hash.slice(1)) || 'list', [a, b, c] = h.split('/');
  const curId = a === 'report' ? c : b;
  $('#tabs').innerHTML = `<button class="tab home ${a === 'list' || a === 'new' ? 'on' : ''}" data-h="list">Список</button>` + tabs.all().map(t =>
    `<button class="tab ${t.id === curId ? 'on' : ''}" data-h="${t.kind}/${t.id}" title="${esc(t.name)}"><small>${t.kind === 'eeg' ? 'ЭЭГ' : 'МРТ'}</small><span>${esc(t.name)}</span><i class="cl" data-close="${t.id}" role="button" aria-label="Закрыть вкладку">×</i></button>`).join('');
  $('#topRight').innerHTML = `<a href="#clinic" class="${a === 'clinic' ? 'on' : ''}">Настройка под клинику</a><a href="#models" class="${a === 'models' ? 'on' : ''}">О моделях</a>
    <span class="cl">${esc(m.clinic)}</span>${m.demo ? '<span class="demo" title="Демо-режим: записи и снимки синтетические, оценки не имеют клинического смысла">ДЕМО</span>' : ''}`;
  const t = m.g1.threshold;
  $('#status').innerHTML = `<span>Модель ЭЭГ: <b>${esc(m.g1.model.title)}</b></span>
    <span>Порог ЭЭГ: <b class="m">${fmt(t.value)}</b> ${t.clinic ? '· настроен под клинику' : '· <a href="#clinic">не настроен под клинику</a>'}</span>
    <span>Модель МРТ: <b>${esc(m.g2.card.title)}</b></span>
    <span class="sp">${m.demo ? 'Демо: данные синтетические' : 'Данные остаются на сервере клиники'} · второе мнение, решение за врачом</span>`;
}

async function route() {
  if (state.leave) { try { state.leave(); } catch (e) { /* экран уже снят */ } state.leave = null; }
  let h = decodeURIComponent(location.hash.slice(1)) || 'list';
  let [a, b, c] = h.split('/');
  if (!ROUTES[a]) { history.replaceState(null, '', '#list'); a = 'list'; b = c = undefined; }      // старые адреса (#home, #g1/…, #ws/…) ведут в список
  shell();
  const app = $('#app'); app.innerHTML = loading();
  try { await ROUTES[a](app, b, c); } catch (e) { app.innerHTML = errorBox('Не получилось открыть экран', e.message); }
  shell();                                              // экран мог открыть вкладку исследования
}

async function start() {
  state.meta = await api.get('/api/meta'); state.shell = shell;
  $('#brand').onclick = () => go('list');
  $('#tabs').onclick = e => {
    const x = e.target.closest('[data-close]');
    if (x) { e.stopPropagation(); const on = x.closest('.tab').classList.contains('on'); tabs.close(x.dataset.close); on ? go('list') : shell(); return; }
    const t = e.target.closest('[data-h]'); if (t) go(t.dataset.h);
  };
  addEventListener('hashchange', route);
  route();
}
start().catch(e => { $('#app').innerHTML = errorBox('Сервер не отвечает', e.message + '. Проверьте, что программа запущена, и обновите страницу.', 'location.reload()', 'Обновить страницу'); });
