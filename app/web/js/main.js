// Маршруты и главная страница — по макету Дарьи. Адрес после # хранит экран и номер исследования: страницу можно обновлять.
import { $, $$, go, state, esc, stepper, STEPS } from './ui.js';
import { api } from './api.js';
import { BRAIN_SVG } from './brain_svg.js';
import { G1 } from './g1.js';
import { G2 } from './g2.js';
import { wsG1 } from './g1ws.js';
import { wsG2 } from './g2ws.js';

function todayRu() { return new Date().toLocaleDateString('ru-RU', { day: 'numeric', month: 'long', year: 'numeric' }); }
function homePage() {
  const demo = state.meta.demo;
  return `<div class="home"><section class="hero">
  <div class="scan">${BRAIN_SVG}<span class="scanlbl">МРТ · поиск ФКД</span></div>
  <div class="txt"><h1>ЭпиАссистент</h1>
   <p class="lead">Второе мнение для врача: поиск очагов фокальной кортикальной дисплазии на МРТ и оценка ЭЭГ</p>
   <span class="date"><i></i>${todayRu()}</span></div></section>
  <p class="choose">Выберите исследование — откроется рабочее место врача</p>
  <div class="mods">
   <button class="mod mri" onclick="go('ws/g2')"><div class="ic"><svg viewBox="0 0 48 48" width="52" height="52"><rect x="6" y="6" width="36" height="36" rx="10" fill="#e3ddfb"/><ellipse cx="24" cy="25" rx="11" ry="13" fill="#b9acee"/><path d="M24 13 V37" stroke="#fff" stroke-width="2"/><circle cx="19" cy="21" r="3.2" fill="#ffb38a"/></svg></div><div><h3>МРТ</h3><p>T1 и FLAIR · поиск очагов ФКД, карта вероятности и список кандидатов</p><div class="go">Открыть рабочее место →</div><span class="alt" onclick="event.stopPropagation();go('g2/upload')">или пошаговый режим</span></div></button>
   <button class="mod eeg" onclick="go('ws/g1')"><div class="ic"><svg viewBox="0 0 48 48" width="52" height="52"><rect x="6" y="6" width="36" height="36" rx="10" fill="#d6f1e5"/><path d="M10 26 H16 L19 17 L23 33 L27 21 L30 28 L33 24 H38" fill="none" stroke="#3d9471" stroke-width="2.6" stroke-linecap="round" stroke-linejoin="round"/></svg></div><div><h3>ЭЭГ</h3><p>Рутинная запись от 20 минут, 19 каналов · вероятность эпилепсии по записи</p><div class="go">Открыть рабочее место →</div><span class="alt" onclick="event.stopPropagation();go('g1/upload')">или пошаговый режим</span></div></button>
  </div>
  <p class="foot">${demo ? 'Демонстрационная версия · данные выдуманы · ' : 'Файлы обрабатываются на сервере клиники и никуда не передаются · '}результат программы — второе мнение, решение принимает врач</p></div>`;
}

async function route() {
  if (state.leave) { try { state.leave(); } catch (e) { /* экран уже снят */ } state.leave = null; }
  const h = decodeURIComponent(location.hash.slice(1)), app = $('#app'), main = $('main');
  const show = (tabs) => { for (const id of ['#tabs', '#clinicLink', '#modelChip']) $(id).style.display = tabs ? '' : 'none'; };
  if (!h || h === 'home') { main.classList.remove('wide'); show(false); app.innerHTML = homePage(); scrollTo(0, 0); return; }
  const [a, b, c] = h.split('/');
  main.classList.toggle('wide', a === 'ws');
  if (a === 'ws') {
    show(false);
    const mod = b === 'g1' ? 'g1' : 'g2';
    app.innerHTML = `<div class="wsbar noprint"><button class="btn link" onclick="go('home')">← На главную</button>
      <span><button class="btn" onclick="go('${mod}/upload')">Пошаговый режим</button></span></div><div class="ws" id="ws"></div>
      <p class="small" style="margin-top:8px">Рабочее место просмотра: просмотрщик как в программах ЭЭГ и PACS, оценка модели — на панели справа. ${mod === 'g2' ? 'Три плоскости, клик по кандидату переводит к очагу, <kbd>↑</kbd><kbd>↓</kbd> листают срезы.' : '<kbd>←</kbd><kbd>→</kbd> листают запись.'}</p>`;
    scrollTo(0, 0);
    try { await (mod === 'g1' ? wsG1 : wsG2)($('#ws'), c); } catch (e) { $('#ws').innerHTML = `<div class="empty"><b>Не получилось.</b> ${esc(e.message)}</div>`; }
    return;
  }
  show(true);
  const tab = STEPS[a] ? a : 'g1', scr = b || 'upload', mod = tab === 'g1' ? G1 : G2;
  $$('.tab').forEach(x => x.classList.toggle('on', x.dataset.tab === tab));
  const m = state.meta[tab];
  $('#modelChip').textContent = 'модель: ' + (tab === 'g1' ? m.model.title : m.card.title);
  const screen = mod[scr] || mod.upload;
  app.innerHTML = '<div class="back noprint"><button class="btn link" onclick="go(\'home\')">← На главную</button></div>' + (scr === 'clinic' ? '' : stepper(tab, mod[scr] ? scr : 'upload')) + '<div id="screen"><div class="card"><p class="sub">Загрузка…</p></div></div>';
  try { await screen($('#screen'), c); } catch (e) {
    $('#screen').innerHTML = `<div class="card"><div class="note bad"><b>Не получилось.</b> ${esc(e.message)}</div><div class="actions"><button class="btn" onclick="go('${tab}/upload')">← К загрузке</button></div></div>`;
  }
}

async function start() {
  state.meta = await api.get('/api/meta');
  $('#modeTag').outerHTML = state.meta.demo ? '<span class="demo" id="modeTag">ДЕМО · данные выдуманы</span>' : `<span class="mode-real" id="modeTag">${esc(state.meta.clinic)} · данные остаются на сервере</span>`;
  $('#tabs').onclick = e => { const t = e.target.dataset.tab; if (t) go(t + '/upload'); };
  $('#clinicLink').onclick = () => go('g1/clinic');
  $('#logoHome').onclick = () => go('home');
  addEventListener('hashchange', route);
  route();
}
start().catch(e => { $('#app').innerHTML = `<div class="card"><div class="note bad">Сервер не отвечает: ${esc(e.message)}</div></div>`; });
