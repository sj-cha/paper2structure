/* Paper -> Structure web tool: three panels over the JSON API of app/server.py.
   Panel 1 uploads papers (several at once, one job each) and runs the summarizer; panel 2 shows the
   summary and runs the evidence analysis per technique; panel 3 takes the material system(s), the
   DFT/MD choice and the atom budget, runs plan + construction + review, and shows the structures in
   3Dmol.js. Jobs are polled together every 2 s; a poll that fails is retried, not abandoned. */
'use strict';

const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];
const esc = (s) => String(s ?? '').replace(/[&<>"']/g, (c) => ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'}[c]));
const cssId = (s) => String(s).replace(/[^A-Za-z0-9_-]/g, '_');

const S = {
  info: null, projects: [], pid: null, project: null, run: null, runName: null, step: 1, openSeq: 0,
  jobs: {}, watching: new Set(), pollTimer: null, polling: false, pollFailures: {}, viewers: {}, atomsAuto: true,
  settings: defaultSettings(), files: [],
};

function defaultSettings() {
  return {systems: [], techniques: [], max_plans: 1, method: 'DFT', realism: 0.3, max_atoms: 220,
          model_type: '', seed: 0, max_revisions: 0, max_repairs: 1};
}

// mirrors projects.default_max_atoms
function defaultMaxAtoms(method, realism) {
  const [lo, hi] = method === 'DFT' ? [60, 600] : [500, 10000];
  return Math.round((lo + (hi - lo) * realism) / 10) * 10;
}

// ---------------------------------------------------------------- helpers

async function api(path, opts = {}) {
  const init = {method: opts.method || 'GET', headers: {}};
  if (opts.form) init.body = opts.form;
  else if (opts.body !== undefined) { init.body = JSON.stringify(opts.body); init.headers['Content-Type'] = 'application/json'; }
  const res = await fetch(path, init);
  let data = null;
  try { data = await res.json(); } catch (e) { data = null; }
  if (!res.ok) throw new Error((data && data.error) || `${res.status} ${res.statusText}`);
  return data;
}

let toastTimer;
function toast(msg, kind = 'info') {
  const t = $('#toast');
  t.textContent = msg;
  t.className = `toast ${kind}`;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => t.classList.add('hidden'), kind === 'error' ? 9000 : 4000);
}

const modelInfo = (name) => (S.info && S.info.models.find((m) => m.name === name)) || null;
const modelTitle = (name) => { const m = modelInfo(name); return m ? m.title : name; };
const providerOf = (name) => (name === 'dry-run' ? 'none' : name.includes(':') ? name.split(':')[0] : 'openai');
const providerName = (p) => ({openai: 'OpenAI', google: 'Gemini'}[p] || p);

function settingsLabel(s) {
  if (!s || !s.method) return '';
  return `${s.method} · ≤${s.max_atoms} atoms · realism ${Number(s.realism).toFixed(2)}${s.model_type ? ` · ${s.model_type}` : ''}`;
}

function statusText(status) {
  return {accepted: 'accepted by the reviewer', inconclusive: 'built, reviewer not convinced',
          failed: 'failed', unfinished: 'unfinished'}[status] || status;
}

const latticeText = (l) => ['a', 'b', 'c', 'alpha', 'beta', 'gamma'].filter((k) => l[k] != null).map((k) => `${k}=${l[k]}`).join(', ');

function elapsed(job) {
  if (!job.started) return 'queued';
  const s = Math.max(0, Math.round((job.finished || Date.now() / 1000) - job.started));
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, '0')}`;
}

const sameList = (x, y) => JSON.stringify([...x].sort()) === JSON.stringify([...y].sort());

function projectBusy(pid) {
  return Object.values(S.jobs).some((j) => j.project_id === pid && (j.status === 'queued' || j.status === 'running'));
}

// The ticked techniques as the server resolves them: empty means every technique with an agent.
function tickedTechniques() {
  const ready = S.project.available_techniques;
  const sel = S.settings.techniques.filter((t) => ready.includes(t));
  return sel.length ? sel : [...ready];
}

// The final report is the reviewer's markdown, rendered with scripts, handlers and unsafe URLs stripped
// (the model read an untrusted PDF).
const URL_ATTR = /^(href|src|xlink:href|action|formaction|poster|background)$/i;
function renderMd(text) {
  if (!window.marked) return `<pre>${esc(text)}</pre>`;
  const tpl = document.createElement('template');
  tpl.innerHTML = marked.parse(text);
  $$('script, iframe, object, embed, style, form, meta, link', tpl.content).forEach((n) => n.remove());
  $$('*', tpl.content).forEach((n) => {
    [...n.attributes].forEach((a) => {
      const v = a.value.replace(/[\u0000- ]/g, '');  // browsers drop tabs/newlines before parsing a scheme
      if (a.name.startsWith('on') || a.name === 'srcdoc' || /^(javascript|data|vbscript):/i.test(v)
          || (URL_ATTR.test(a.name) && !/^(https?:|mailto:|#|\/(?!\/)|[^:]*$)/i.test(v))) n.removeAttribute(a.name);
    });
  });
  return tpl.innerHTML;
}

function modelSelectHtml(id, selected) {
  const groups = [...new Set(S.info.models.map((m) => m.group))];
  return `<select id="${id}">${groups.map((g) => `<optgroup label="${esc(g)}">${S.info.models.filter((m) => m.group === g)
    .map((m) => `<option value="${esc(m.name)}" ${m.name === selected ? 'selected' : ''}>${esc(m.title)} — ${esc(m.price)}</option>`).join('')}</optgroup>`).join('')}</select>`;
}

function effortSelectHtml(id, selected, model) {
  const off = providerOf(model || '') === 'google';
  return `<select id="${id}" ${off ? 'disabled title="Ignored for Gemini models: the agents pass the effort as an OpenAI setting"' : ''}>${S.info.efforts.map((e) => `<option ${e === selected ? 'selected' : ''}>${e}</option>`).join('')}</select>`;
}

function setTheme(theme) {
  document.documentElement.dataset.theme = theme;
  $('#themeBtn').textContent = theme === 'dark' ? '☀' : '☾';
  try { localStorage.setItem('p2s-theme', theme); } catch (e) { /* private window */ }
}

// ---------------------------------------------------------------- start-up

async function init() {
  let theme = 'light';
  try { theme = localStorage.getItem('p2s-theme') || 'light'; } catch (e) { /* ignore */ }
  setTheme(theme);
  $('#themeBtn').onclick = () => setTheme(document.documentElement.dataset.theme === 'dark' ? 'light' : 'dark');
  try { S.info = await api('/api/info'); } catch (e) { toast(`Cannot reach the server: ${e.message}`, 'error'); return; }
  $('#model').outerHTML = modelSelectHtml('model', (S.info.models.find((m) => m.default) || S.info.models[0]).name);
  $('#effort').outerHTML = effortSelectHtml('effort', 'low');
  $('#projModel3').outerHTML = modelSelectHtml('projModel3', '');
  $('#projEffort3').outerHTML = effortSelectHtml('projEffort3', 'low');
  S.info.model_types.forEach((t) => $('#modelType').add(new Option(t, t)));
  $$('.step-btn').forEach((b) => (b.onclick = () => showStep(+b.dataset.step)));
  $('#paperSwitch').onchange = () => { if ($('#paperSwitch').value) openProject($('#paperSwitch').value, 2); else { S.pid = null; S.project = null; render(); showStep(1); } };
  wireStep1();
  wireStep3();
  try { await loadProjects(); } catch (e) { toast(`Could not list the papers: ${e.message}`, 'error'); }
  await loadJobs();
  const h = new URLSearchParams(location.hash.slice(1));
  if (h.get('p') && S.projects.some((p) => p.id === h.get('p'))) await openProject(h.get('p'), +(h.get('s') || 2));
  else showStep(1);
}

function showStep(n) {
  const p = S.project;
  const can = {1: true, 2: !!(p && p.summary), 3: !!(p && p.summary)};
  if (!can[n]) n = 1;
  S.step = n;
  $$('.step').forEach((s) => s.classList.toggle('hidden', s.id !== `step${n}`));
  $$('.step-btn').forEach((b) => { const k = +b.dataset.step; b.classList.toggle('on', k === n); b.classList.toggle('done', k < n && can[k]); b.disabled = !can[k]; });
  history.replaceState(null, '', S.pid ? `#p=${S.pid}&s=${n}` : '#');
  if (n === 3) setTimeout(resizeViewers, 50);
}

function render() {
  renderSwitch();
  renderProjects();
  renderJobList();
  renderStep2();
  renderStep3();
  updateButtons();
}

function updateButtons() {
  const busy = !!(S.pid && projectBusy(S.pid));
  ['#charBtn', '#generateBtn', '#resummarize'].forEach((sel) => { const b = $(sel); if (b) b.disabled = busy; });
  const g = $('#generateBtn');
  if (g && S.project && S.project.project.model === 'dry-run') { g.disabled = true; g.title = 'The dry-run model cannot plan or build: choose a real model (Advanced, or panel 2).'; }
  else if (g) g.title = '';
}

function renderSwitch() {
  const sel = $('#paperSwitch');
  sel.innerHTML = '<option value="">— open a paper —</option>' + S.projects.map((p) => `<option value="${esc(p.id)}" ${p.id === S.pid ? 'selected' : ''}>${esc(p.title.slice(0, 70))}</option>`).join('');
}

// ---------------------------------------------------------------- projects

async function loadProjects() {
  S.projects = await api('/api/projects');
  S.projects.forEach((p) => { if (p.active_job) S.watching.add(p.active_job); });
  ensurePolling();
  renderSwitch();
  renderProjects();
}

function renderProjects() {
  const el = $('#projectList');
  if (!S.projects.length) { el.innerHTML = '<p class="empty">Nothing yet — drop a paper above to start.</p>'; return; }
  el.innerHTML = S.projects.map((p) => `
    <div class="proj-row ${p.id === S.pid ? 'on' : ''}" data-id="${esc(p.id)}" title="${esc(p.pdf_name)}">
      <div class="t">${esc(p.title)}</div>
      <div class="m">${esc(p.created.slice(0, 16))} · ${esc(modelTitle(p.model))} · ${esc(p.effort)}</div>
      <div class="badges">
        ${p.active_job || projectBusy(p.id) ? '<span class="badge busy">working…</span>' : ''}
        <span class="badge ${p.summary ? 'ok' : ''}">summary</span>
        <span class="badge ${p.evidence ? 'ok' : ''}">evidence</span>
        <span class="badge ${p.runs ? 'ok' : ''}">${p.runs} run${p.runs === 1 ? '' : 's'}</span>
      </div>
      <button class="icon del" title="Delete this paper and everything made from it" data-del="${esc(p.id)}">×</button>
    </div>`).join('');
  $$('.proj-row', el).forEach((row) => (row.onclick = (e) => { if (e.target.dataset.del) return; openProject(row.dataset.id, 2); }));
  $$('[data-del]', el).forEach((b) => (b.onclick = async (e) => {
    e.stopPropagation();
    if (!confirm('Delete this paper and all of its results?')) return;
    try {
      await api(`/api/projects/${b.dataset.del}`, {method: 'DELETE'});
      if (S.pid === b.dataset.del) { S.pid = null; S.project = null; S.run = null; S.runName = null; $$('.progress').forEach((x) => x.classList.add('hidden')); showStep(1); }
      await loadProjects();
      render();
    } catch (err) { toast(err.message, 'error'); }
  }));
}

async function openProject(pid, step = 2) {
  const seq = ++S.openSeq;
  try {
    const state = await api(`/api/projects/${pid}`);
    if (seq !== S.openSeq) return;  // a later open won
    S.pid = pid;
    S.project = state;
    S.run = null;
    S.runName = state.project.current_run;
    $$('.progress').forEach((b) => b.classList.add('hidden'));
    initSettingsFromProject();
    render();
    if (state.active_job) { S.watching.add(state.active_job); ensurePolling(); }
    renderProjectProgress();
    showStep(step);
  } catch (e) { toast(e.message, 'error'); }
}

async function refreshProject(reinit = false) {
  if (!S.pid) return;
  const pid = S.pid;
  const state = await api(`/api/projects/${pid}`);
  if (S.pid !== pid) return;  // the user switched papers meanwhile
  S.project = state;
  if (reinit) initSettingsFromProject();
  render();
}

function initSettingsFromProject() {
  const p = S.project;
  const base = p.project.last_settings || (p.plans[0] && p.plans[0].settings) || null;
  S.settings = Object.assign(defaultSettings(), base || {});
  S.settings.model_type = S.settings.model_type || '';
  if (p.summary) {
    const ids = p.summary.systems.map((s) => s.system_id);
    S.settings.systems = S.settings.systems.filter((i) => ids.includes(i));
    if (!S.settings.systems.length && ids.length) S.settings.systems = [ids[0]];
    const ready = p.available_techniques;
    S.settings.techniques = S.settings.techniques.filter((t) => ready.includes(t));
    if (!S.settings.techniques.length) S.settings.techniques = [...ready];
  }
  S.atomsAuto = !base || base.max_atoms === defaultMaxAtoms(base.method, base.realism);
}

// ---------------------------------------------------------------- jobs (all polled together)

async function loadJobs() {
  try {
    const jobs = await api('/api/jobs');  // summaries: the log holds only the last line
    jobs.forEach((j) => { if (!S.jobs[j.id]) S.jobs[j.id] = j; if (j.status === 'queued' || j.status === 'running') S.watching.add(j.id); });
    ensurePolling();
    renderJobList();
  } catch (e) { /* the list view is best effort */ }
}

function trackJob(id, kind, pid, title) {
  S.jobs[id] = {id, project_id: pid, title, kind, status: 'queued', log: [], created: Date.now() / 1000, full: true};
  S.watching.add(id);
  ensurePolling();
  pollOnce();
}

function ensurePolling() {
  if (S.watching.size && !S.pollTimer) S.pollTimer = setInterval(pollOnce, 2000);
  if (!S.watching.size && S.pollTimer) { clearInterval(S.pollTimer); S.pollTimer = null; }
}

async function pollOnce() {
  if (S.polling) return;  // a slow poll must not pile up requests
  S.polling = true;
  try {
    for (const id of [...S.watching]) {
      let job;
      try { job = await api(`/api/jobs/${id}`); }
      catch (e) {  // a transient error (e.g. a file being written) is retried, up to a limit
        S.pollFailures[id] = (S.pollFailures[id] || 0) + 1;
        const lost = /no such job/.test(e.message);
        if (lost || S.pollFailures[id] > 30) {
          S.watching.delete(id);
          const j = S.jobs[id];
          if (j && (j.status === 'queued' || j.status === 'running')) { j.status = 'failed'; j.error = lost ? 'job lost (server restarted?)' : `unreachable: ${e.message}`; j.finished = Date.now() / 1000; }
        }
        continue;
      }
      S.pollFailures[id] = 0;
      job.full = true;
      S.jobs[id] = job;
      if (job.status === 'done' || job.status === 'failed') {
        try { await onJobFinished(job); S.watching.delete(id); }
        catch (e) {  // keep watching so the finish is processed on the next poll
          job.refreshFailures = (job.refreshFailures || 0) + 1;
          if (job.refreshFailures > 5) { S.watching.delete(id); toast(`Could not refresh after "${job.title}": ${e.message}`, 'error'); }
        }
      }
    }
    renderJobList();
    renderProjectProgress();
    updateButtons();
  } finally {
    S.polling = false;
    ensurePolling();
  }
}

async function onJobFinished(job) {
  await loadProjects();
  if (job.project_id === S.pid) await refreshProject(job.kind === 'summarize');  // systems and techniques come from the new summary
  if (job.notified) return;
  job.notified = true;
  const name = job.title.length > 50 ? job.title.slice(0, 50) + '…' : job.title;
  if (job.status === 'failed') { toast(`${name}: ${job.error}`, 'error'); return; }
  if (job.kind === 'summarize') {
    toast(job.result && job.result.characterization_error ? `Summary ready (evidence analysis skipped): ${name}` : `Summary ready: ${name}`);
    if (job.project_id === S.pid && S.step === 1) showStep(2);
  } else if (job.kind === 'characterize') {
    toast(`Evidence analyzed: ${name}`);
  } else if (job.kind === 'generate' && job.result && job.result.run) {
    toast(`Run finished (${statusText(job.result.status)}): ${name}`);
    if (job.project_id === S.pid) await loadRun(job.result.run);
  }
}

function jobTitle(job) {
  const log = job.log || [];
  return {summarize: 'Reading the paper (Summarizer)' + (log.some((l) => /^Analyzing the evidence|^Evidence analyzed/.test(l)) ? ' and analyzing the evidence (characterization agents)' : ''),
          characterize: 'Analyzing the evidence (characterization agents)',
          generate: log.some((l) => /^A plan for exactly these settings exists/.test(l)) ? 'Building and reviewing (Constructor and Reviewer; saved plan reused)' : 'Planning, building and reviewing (Planner, Constructor and Reviewer)'}[job.kind] || job.kind;
}

// Panel 1: one row per job of any paper, the active ones first; open expanders survive the 2 s re-render.
function renderJobList() {
  const el = $('#jobList');
  const jobs = Object.values(S.jobs).sort((a, b) => b.created - a.created).filter((j) => j.status !== 'done' || Date.now() / 1000 - (j.finished || 0) < 600).slice(0, 12);
  if (!jobs.length) { el.innerHTML = ''; return; }
  const open = new Set($$('details[open]', el).map((d) => d.dataset.job));
  el.innerHTML = `<div class="card"><h2>In progress <span class="muted">${S.info.workers} papers at a time</span></h2>` + jobs.map((j) => `
    <details class="sub" data-job="${esc(j.id)}" ${j.status === 'failed' || open.has(j.id) ? 'open' : ''}><summary><div class="job"><span class="spinner ${j.status}"></span>
      <div><div class="t">${esc(j.title)} <span class="badge kind">${esc(j.kind)}</span> <span class="badge ${j.status}">${esc(j.status)}</span> <span class="muted small">${elapsed(j)}</span></div>
        <div class="l">${esc(j.error || ((j.log || []).length ? j.log[j.log.length - 1] : 'queued'))}</div></div>
      <button class="small" data-open="${esc(j.project_id)}">open</button></div></summary>
      <pre class="log">${esc((j.log || []).slice(-40).join('\n'))}</pre></details>`).join('') + '</div>';
  $$('[data-open]', el).forEach((b) => (b.onclick = (e) => { e.preventDefault(); openProject(b.dataset.open, 2); }));
}

// The progress card of the open project on panels 2 and 3, with a stage strip for the graph run.
const STAGES = [['SummarizePaper', 'Summary'], ['AnalyzeEvidence', 'Evidence'], ['PlanStructures', 'Plan'], ['ConstructStructures', 'Build'],
                ['ValidateStructures', 'Validate'], ['ScientificRevision', 'Review'], ['FinalizeResults', 'Report']];

function renderProjectProgress() {
  if (!S.pid) return;
  const mine = Object.values(S.jobs).filter((j) => j.project_id === S.pid && j.kind !== 'summarize').sort((a, b) => b.created - a.created);
  $$('.progress').forEach((b) => b.classList.add('hidden'));
  const job = mine[0];
  if (!job) return;
  if (!job.full && !job.fetching) {  // a summary from /api/jobs: get the whole log once
    job.fetching = true;
    api(`/api/jobs/${job.id}`).then((j) => { S.jobs[j.id] = Object.assign(j, {full: true}); renderProjectProgress(); }).catch(() => { job.fetching = false; });
  }
  const box = $(job.kind === 'generate' ? '#progress3' : '#progress2');
  const logOpen = !!box.querySelector('details[open]');
  box.classList.remove('hidden');
  const log = job.log || [];
  let strip = '';
  if (job.kind === 'generate' && job.full) {
    const seen = STAGES.map(([node]) => { let k = -1; log.forEach((l, i) => { if (l.startsWith(`[${node}]`)) k = i; }); return k; });  // last occurrence: repairs re-enter Build
    const lastIdx = Math.max(-1, ...seen);
    const replanned = log.some((l) => /^Planning structures/.test(l));
    strip = '<div class="stage-strip">' + STAGES.map(([node, label], i) => {
      let cls = '', text = label, title = '';
      const loaded = node === 'SummarizePaper' || node === 'AnalyzeEvidence';
      if (loaded) { cls = seen[i] >= 0 ? 'reused' : ''; text = label + ' (saved)'; title = 'Loaded from disk: this agent is not run again'; }
      else if (node === 'PlanStructures') { text = replanned ? 'Plan (new)' : 'Plan (saved)'; title = replanned ? 'The settings changed: planned again' : 'The plan for these settings is reused'; }
      if (seen[i] >= 0 && seen[i] === lastIdx && job.status === 'running') cls = 'active';
      else if (seen[i] >= 0 && !loaded) cls = 'done';
      if (job.status === 'failed' && seen[i] === lastIdx) cls = 'failed';
      if (node === 'FinalizeResults' && job.status === 'done') cls = 'done';
      return `<span class="stage ${cls}" title="${esc(title)}">${esc(text)}</span>`;
    }).join('') + '</div>';
  }
  const said = log.filter((l) => !/^\[/.test(l) && !/^ERROR/.test(l));
  box.innerHTML = `<div class="job-head"><span class="spinner ${job.status}"></span><strong>${esc(jobTitle(job))}</strong>
      <span class="badge ${job.status}">${esc(job.status)}</span><span class="muted small">${elapsed(job)}</span></div>
    ${strip}
    ${job.error ? `<div class="error">${esc(job.error)}</div>` : `<div class="current">${esc(said[said.length - 1] || 'starting…')}</div>`}
    <details class="sub" ${logOpen ? 'open' : ''}><summary>Agent log (${log.length} lines)</summary><pre class="log">${esc(log.slice(-150).join('\n'))}</pre></details>`;
}

// ---------------------------------------------------------------- panel 1

function selectedModel() { return $('#model').value; }

function renderModelLine() {
  const m = modelInfo(selectedModel());
  const el = $('#modelLine');
  if (!m) { el.innerHTML = ''; return; }
  const eff = $('#effort');
  eff.disabled = m.provider === 'google';
  eff.title = eff.disabled ? 'Ignored for Gemini models: the agents pass the effort as an OpenAI setting' : '';
  const key = m.provider === 'none' ? '' : (S.info.keys[m.provider]
    ? `<span class="badge ok">${providerName(m.provider)} key ✓</span>`
    : `<span class="badge bad">${providerName(m.provider)} key missing</span> <button class="link" id="addKey">add it</button>`);
  const many = S.files.length > 1 && m.provider === 'google' ? '<span class="badge busy">several papers on a Gemini free-tier key may hit its rate limits</span>' : '';
  const price = m.price.startsWith('$') ? `${esc(m.price)} per 1M tokens (input / output)` : esc(m.price);
  el.innerHTML = `<span class="muted">${price}${m.note ? ' · ' + esc(m.note) : ''}</span> ${key} ${many}`;
  const add = $('#addKey');
  if (add) add.onclick = () => showKeyBox(m.provider);
}

function showKeyBox(provider) {
  const names = {openai: 'OPENAI_API_KEY', google: 'GOOGLE_API_KEY'}[provider];
  $('#keyBox').classList.remove('hidden');
  $('#keyBox').dataset.provider = provider;
  $('#keyText').innerHTML = `No ${providerName(provider)} key found. Put <code>${names}=...</code> in the <code>.env</code> file of the repo (durable), or paste it here for this server session:`;
}

function wireStep1() {
  const input = $('#pdf'), drop = $('#drop');
  const setFiles = (files) => {
    S.files = [...files].filter((f) => /\.pdf$/i.test(f.name));
    $('#fileList').innerHTML = S.files.map((f) => `<span>${esc(f.name)}</span>`).join('');
    $('#dropText').textContent = S.files.length ? `${S.files.length} PDF${S.files.length === 1 ? '' : 's'} chosen — add more or upload` : 'Drop PDFs here or click to choose';
    renderModelLine();
  };
  input.onchange = () => setFiles(input.files);
  ['dragenter', 'dragover'].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add('over'); }));
  ['dragleave', 'drop'].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.remove('over'); }));
  drop.addEventListener('drop', (e) => { if (e.dataTransfer.files.length) { input.files = e.dataTransfer.files; setFiles(input.files); } });
  $('#model').onchange = renderModelLine;
  $('#summarizeBtn').onclick = startSummary;
  $('#keyBtn').onclick = async () => {
    try {
      const r = await api('/api/key', {method: 'POST', body: {provider: $('#keyBox').dataset.provider || 'openai', key: $('#keyInput').value}});
      S.info.keys = r.keys;
      $('#keyBox').classList.add('hidden');
      $('#keyInput').value = '';
      renderModelLine();
      toast('Key set for this server session');
    } catch (e) { toast(e.message, 'error'); }
  };
  renderModelLine();
}

async function startSummary() {
  if (!S.files.length) { toast('Choose one or more PDFs first', 'error'); return; }
  const form = new FormData();
  S.files.forEach((f) => form.append('pdf', f));
  form.append('model', selectedModel());
  form.append('effort', $('#effort').value);
  form.append('analyze_figures', $('#figures').checked ? 'true' : 'false');
  form.append('max_figures', $('#maxFigures').value);
  form.append('then_characterize', $('#thenChar').checked ? 'true' : 'false');
  const btn = $('#summarizeBtn');
  btn.disabled = true;
  try {
    const r = await api('/api/projects', {method: 'POST', form});
    r.projects.forEach((p) => { S.jobs[p.job_id] = {id: p.job_id, project_id: p.project_id, title: p.title, kind: 'summarize', status: 'queued', log: [], created: Date.now() / 1000, full: true}; S.watching.add(p.job_id); });
    S.files = [];
    $('#fileList').innerHTML = '';
    $('#pdf').value = '';
    $('#dropText').textContent = 'Drop PDFs here or click to choose';
    await loadProjects();
    if (r.projects.length === 1) { const state = await api(`/api/projects/${r.projects[0].project_id}`); S.pid = r.projects[0].project_id; S.project = state; S.run = null; S.runName = null; initSettingsFromProject(); }
    render();
    ensurePolling();
    pollOnce();
    toast(r.projects.length === 1 ? 'Reading the paper…' : `Reading ${r.projects.length} papers, ${Math.min(r.projects.length, S.info.workers)} at a time…`);
  } catch (e) { toast(e.message, 'error'); }
  finally { btn.disabled = false; }
}

// ---------------------------------------------------------------- panel 2: summary + evidence analysis

function currentCharacterization() {
  // the newest analysis made with the project's model and effort on the ticked techniques (as the server resolves them)
  const p = S.project, pr = p.project, ticked = tickedTechniques();
  return p.characterizations.find((c) => c.model === pr.model && c.effort === pr.effort && sameList(c.techniques, ticked)) || null;
}

function renderStep2() {
  const head = $('#s2header'), tech = $('#s2techniques'), res = $('#s2results'), p = S.project;
  if (!p || !p.summary) { head.innerHTML = '<div class="card"><p class="empty">Upload and summarize a paper on panel 1 first.</p></div>'; tech.innerHTML = res.innerHTML = ''; return; }
  const m = p.summary.metadata, st = S.settings, sum = p.project.summary || {}, pr = p.project;
  const sysRows = p.summary.systems.map((s) => {
    const lat = (s.structure.lattice_parameters || []).map((x) => `${x.quantity}=${x.value}${x.unit || ''}`).join(', ');
    return `<tr><td><b>${esc(s.system_id)}</b></td><td>${esc(s.name)}</td><td>${esc(s.form || '')}</td><td>${esc(s.structure.space_group || s.structure.crystal_system || '')}</td><td>${esc(lat)}</td><td>${esc(s.synthesis.method || '')}</td></tr>`;
  }).join('');
  head.innerHTML = `
    <div class="card">
      <div class="run-head"><h2 class="grow" style="margin:0">${esc(m.title)}</h2><button class="small" id="resummarize">Read again</button></div>
      <p class="muted small">${esc((m.authors || []).join(', '))}${m.year ? ` · ${m.year}` : ''}${m.doi ? ` · DOI ${esc(m.doi)}` : ''}
        · read by ${esc(modelTitle(pr.model))}${sum.seconds ? ` in ${sum.seconds} s` : ''}${sum.cost != null ? ` · ≈ $${sum.cost.toFixed(3)}` : ''}</p>
      <p class="clamp" id="paperSummary">${esc(m.summary)}</p><button class="link" id="moreSummary">more</button>
      ${sum.errors && sum.errors.length ? `<div class="warn">Some steps had problems:<br>${sum.errors.map(esc).join('<br>')}</div>` : ''}
      <div class="chips" style="margin-top:10px">${p.summary.systems.map((s) => `<span class="chip plain" title="${esc(s.form || '')}"><b>${esc(s.system_id)}</b> ${esc(s.name)}</span>`).join('')}</div>
      <details class="sub"><summary>Systems table (${p.summary.systems.length})</summary>
        <table class="tbl small"><thead><tr><th>ID</th><th>Name</th><th>Form</th><th>Space group</th><th>Lattice parameters</th><th>Synthesis</th></tr></thead><tbody>${sysRows}</tbody></table></details>
    </div>`;
  $('#moreSummary').onclick = () => { $('#paperSummary').classList.toggle('open'); $('#moreSummary').textContent = $('#paperSummary').classList.contains('open') ? 'less' : 'more'; };
  $('#resummarize').onclick = async () => {
    if (!confirm('Read the paper again with the current model? The evidence analysis and plans made from the old summary are discarded (finished runs are kept).')) return;
    try { const r = await api(`/api/projects/${S.pid}/summarize`, {method: 'POST', body: {}}); trackJob(r.job_id, 'summarize', S.pid, pr.title); showStep(1); }
    catch (e) { toast(e.message, 'error'); }
  };

  const chips = p.summary.techniques.map((t) => {
    const ready = t.agent === 'ready', on = st.techniques.includes(t.name);
    const why = 'not in the technique vocabulary';
    return `<label class="chip ${ready ? '' : 'off'}" title="${ready ? `${t.count} evidence items` : why}">
      <input type="checkbox" data-tech="${esc(t.name)}" ${on ? 'checked' : ''} ${ready ? '' : 'disabled'}> ${esc(t.name)} <span class="muted">${t.count}</span></label>`;
  }).join('');
  const cur = currentCharacterization();
  const status = cur ? `analyzed with ${modelTitle(cur.model)}, ${cur.effort} effort · ${cur.seconds} s · ${cur.techniques.join(', ')}` : 'not analyzed yet for these techniques, model and effort';
  tech.innerHTML = `<div class="card">
      <h2>Evidence analysis <span class="muted">one characterization agent per technique reads its figures and text</span></h2>
      <div class="chips" id="techChips">${chips || '<span class="muted">no technique with evidence in this paper</span>'}</div>
      <div class="actions"><button class="primary" id="charBtn">Analyze evidence</button><span class="hint" id="charStatus">${esc(status)}</span></div>
      <details class="adv"><summary>Model for this step</summary><div class="row wrap gap small">
        <label class="inline">Model ${modelSelectHtml('projModel2', pr.model)}</label><label class="inline">Effort ${effortSelectHtml('projEffort2', pr.effort, pr.model)}</label>
        <span class="hint">applies to this paper's remaining steps</span></div></details>
    </div>`;
  $$('#techChips input').forEach((c) => (c.onchange = () => { st.techniques = $$('#techChips input').filter((x) => x.checked).map((x) => x.dataset.tech); renderStep2Results(); renderPlanStatus(); }));
  $('#charBtn').onclick = startCharacterize;
  $('#projModel2').onchange = () => updateProjectSettings({model: $('#projModel2').value});
  $('#projEffort2').onchange = () => updateProjectSettings({effort: $('#projEffort2').value});
  renderStep2Results();
  updateButtons();
}

async function updateProjectSettings(body) {
  try {
    await api(`/api/projects/${S.pid}/settings`, {method: 'POST', body});
    await refreshProject();
    await loadProjects();
  } catch (e) { toast(e.message, 'error'); render(); }
}

async function startCharacterize() {
  if (!S.settings.techniques.length) { toast('Tick at least one technique', 'error'); return; }
  const btn = $('#charBtn');
  btn.disabled = true;
  try {
    const r = await api(`/api/projects/${S.pid}/characterize`, {method: 'POST', body: {techniques: S.settings.techniques}});
    trackJob(r.job_id, 'characterize', S.pid, S.project.project.title);
  } catch (e) { toast(e.message, 'error'); btn.disabled = false; }
}

function evidenceItems(technique) {
  const items = S.project.summary.evidence.filter((e) => e.technique === technique);
  return `<details class="sub"><summary>Source evidence from the paper (${items.length} items)</summary>
    ${items.map((e) => `<div class="ev-item"><div class="ev-text"><span class="eid">${esc(e.evidence_id)}</span>
        <span class="muted small">${esc(e.source_ref)} · ${esc(e.system_ids.join(', '))}</span><div>${esc(e.summary)}</div>
        ${e.measurements.length ? `<div class="small">${e.measurements.map((x) => `${esc(x.quantity)} = ${esc(x.value)} ${esc(x.unit || '')}`).join(' · ')}</div>` : ''}
        ${e.quote ? `<div class="quote">“${esc(e.quote)}”</div>` : ''}</div>
      ${e.image_url ? `<a href="${esc(e.image_url)}" target="_blank"><img class="thumb" src="${esc(e.image_url)}" alt="" loading="lazy"></a>` : ''}</div>`).join('')}</details>`;
}

function renderStep2Results() {
  const res = $('#s2results'), p = S.project;
  const cur = currentCharacterization() || p.characterizations[0];
  if (!cur) {
    res.innerHTML = `<div class="card"><h2>What the paper says <span class="muted">per technique, before any analysis</span></h2>
      ${p.summary.techniques.map((t) => `<details class="ev"><summary><b>${esc(t.name)}</b> · ${t.count} item${t.count === 1 ? '' : 's'}${t.figures ? ` · ${t.figures} from figures` : ''}</summary>${evidenceItems(t.name)}</details>`).join('')}</div>`;
    return;
  }
  const stale = cur !== currentCharacterization();
  res.innerHTML = `<div class="card">
    <h2>What the evidence says <span class="muted">${esc(cur.techniques.join(', '))} · ${esc(modelTitle(cur.model))}, ${esc(cur.effort)} effort · ${cur.seconds} s${stale ? ' · (older analysis: not the ticked techniques or current model)' : ''}</span></h2>
    ${cur.reports_detail.map((r, i) => `<details class="ev" ${i === 0 ? 'open' : ''}><summary><b>${esc(r.technique)}</b> · ${r.observations} observations · ${r.constraints.length} structural constraints</summary>
      ${r.constraints.length ? `<table class="tbl small"><thead><tr><th>Property</th><th>Value</th><th>Systems</th><th>Confidence</th></tr></thead><tbody>
        ${r.constraints.map((c) => `<tr title="${esc(c.derivation)}"><td>${esc(c.property)}</td><td>${esc(c.value)} ${esc(c.unit || '')}</td><td>${esc(c.system_ids.join(', '))}</td><td><span class="badge conf-${esc(c.confidence)}">${esc(c.confidence)}</span></td></tr>`).join('')}</tbody></table>` : '<p class="muted small">No structural constraints from this technique.</p>'}
      ${r.findings.length ? `<ul class="small">${r.findings.map((f) => `<li>${esc(f.claim)} <span class="muted">(${esc(f.confidence)})</span></li>`).join('')}</ul>` : ''}
      ${r.caveats.length ? `<details class="sub"><summary>Caveats (${r.caveats.length})</summary><ul class="small">${r.caveats.map((c) => `<li>${esc(c)}</li>`).join('')}</ul></details>` : ''}
      ${evidenceItems(r.technique)}
    </details>`).join('')}
    ${Object.keys(cur.skipped || {}).length ? `<p class="muted small">Not analyzed: ${esc(Object.entries(cur.skipped).map(([k, v]) => `${k} (${v})`).join('; '))}</p>` : ''}
    ${cur.errors && cur.errors.length ? `<div class="warn">${cur.errors.map(esc).join('<br>')}<br>Click "Analyze evidence" to retry the failed techniques.</div>` : ''}
    <div class="actions"><button class="primary" id="toStep3">Continue to structures →</button></div>
  </div>`;
  $('#toStep3').onclick = () => showStep(3);
}

// ---------------------------------------------------------------- panel 3: what to build, runs

function wireStep3() {
  $$('#method button').forEach((b) => (b.onclick = () => { S.settings.method = b.dataset.v; syncAtoms(); renderStep3Form(); }));
  $('#realism').oninput = () => { S.settings.realism = +$('#realism').value; syncAtoms(); renderStep3Form(); };
  $('#maxAtoms').onchange = () => { S.settings.max_atoms = Math.max(10, Math.min(50000, +$('#maxAtoms').value || 10)); S.atomsAuto = false; renderStep3Form(); };
  $('#atomsAuto').onchange = () => { S.atomsAuto = $('#atomsAuto').checked; syncAtoms(); renderStep3Form(); };
  $('#maxPlans').onchange = () => { S.settings.max_plans = +$('#maxPlans').value; renderPlanStatus(); };
  $('#modelType').onchange = () => { S.settings.model_type = $('#modelType').value; renderPlanStatus(); };
  $('#seed').onchange = () => { S.settings.seed = +$('#seed').value || 0; };
  $('#maxRevisions').onchange = () => { S.settings.max_revisions = Math.max(0, Math.min(2, +$('#maxRevisions').value || 0)); };
  $('#maxRepairs').onchange = () => { S.settings.max_repairs = Math.max(0, Math.min(3, +$('#maxRepairs').value || 0)); };
  $('#projModel3').onchange = () => updateProjectSettings({model: $('#projModel3').value});
  $('#projEffort3').onchange = () => updateProjectSettings({effort: $('#projEffort3').value});
  $('#generateBtn').onclick = startGenerate;
  $('#runSelect').onchange = () => loadRun($('#runSelect').value);
}

function syncAtoms() { if (S.atomsAuto) S.settings.max_atoms = defaultMaxAtoms(S.settings.method, S.settings.realism); }

function renderStep3() {
  if (!S.project) return;
  renderStep3Form();
  renderRuns();
}

function renderStep3Form() {
  const st = S.settings, p = S.project;
  if (p.summary) {
    $('#sysChips').innerHTML = p.summary.systems.map((s) => `<label class="chip" title="${esc(s.form || '')}"><input type="checkbox" data-sys="${esc(s.system_id)}" ${st.systems.includes(s.system_id) ? 'checked' : ''}> <b>${esc(s.system_id)}</b> ${esc(s.name)}</label>`).join('');
    $$('#sysChips input').forEach((c) => (c.onchange = () => { st.systems = $$('#sysChips input').filter((x) => x.checked).map((x) => x.dataset.sys); renderPlanStatus(); }));
  }
  $$('#method button').forEach((b) => b.classList.toggle('on', b.dataset.v === st.method));
  $('#realism').value = st.realism;
  $('#realismVal').textContent = Number(st.realism).toFixed(2);
  $('#maxAtoms').value = st.max_atoms;
  $('#atomsAuto').checked = S.atomsAuto;
  $('#maxPlans').value = st.max_plans;
  $('#modelType').value = st.model_type || '';
  $('#seed').value = st.seed;
  $('#maxRevisions').value = st.max_revisions;
  $('#maxRepairs').value = st.max_repairs;
  $('#projModel3').value = p.project.model;
  const eff = $('#projEffort3');
  eff.value = p.project.effort;
  eff.disabled = providerOf(p.project.model) === 'google';
  renderPlanStatus();
}

function planFor(settings) {
  // a saved plan for exactly these settings, the project's model and effort, and the ticked techniques
  const p = S.project, pr = p.project, b = settings, ticked = tickedTechniques();
  return p.plans.find((r) => {
    const a = r.settings;
    return r.model === pr.model && r.effort === pr.effort && sameList(r.techniques || [], ticked) && sameList(a.systems, b.systems)
      && a.max_plans === b.max_plans && a.method === b.method && Math.abs(a.realism - b.realism) < 1e-9
      && a.max_atoms === b.max_atoms && (a.model_type || '') === (b.model_type || '');
  }) || null;
}

function renderPlanStatus() {
  const p = S.project, el = $('#planStatus');
  if (!p || !p.summary) return;
  const parts = [];
  const cur = currentCharacterization();
  parts.push(cur ? `evidence ✓ (${cur.techniques.join(', ')})` : 'evidence: will be analyzed first');
  if (!S.settings.systems.length) parts.push('tick a system');
  else parts.push(planFor(S.settings) ? 'plan for these settings ✓ → builds directly' : 'will plan first (~1 min), then build');
  el.textContent = parts.join(' · ');
}

async function startGenerate() {
  if (!S.pid) return;
  if (!S.settings.systems.length) { toast('Tick at least one material system', 'error'); return; }
  if (!tickedTechniques().length) { toast('This paper has no technique with an analysis agent', 'error'); return; }
  const btn = $('#generateBtn');
  btn.disabled = true;
  try {
    const r = await api(`/api/projects/${S.pid}/generate`, {method: 'POST', body: {...S.settings, techniques: tickedTechniques(), model_type: S.settings.model_type || null}});
    trackJob(r.job_id, 'generate', S.pid, S.project.project.title);
  } catch (e) { toast(e.message, 'error'); btn.disabled = false; }
}

function renderRuns() {
  const p = S.project, sel = $('#runSelect'), bar = $('#runsBar');
  if (!p || !p.runs.length) { bar.classList.add('hidden'); $('#runDetail').innerHTML = ''; S.run = null; return; }
  bar.classList.remove('hidden');
  sel.innerHTML = p.runs.map((r) => `<option value="${esc(r.name)}">${esc(r.created.slice(0, 16))} · ${esc(statusText(r.status))} · ${r.n_accepted}/${r.n_structures} accepted · ${esc(Object.values(r.systems || {}).join(', ').slice(0, 40))} · ${esc(settingsLabel(r.settings))}</option>`).join('');
  if (!S.runName || !p.runs.some((r) => r.name === S.runName)) S.runName = p.runs[0].name;
  sel.value = S.runName;
  if (!S.run || S.run.name !== S.runName) loadRun(S.runName);
}

async function loadRun(name) {
  S.runName = name;  // the request token: a later choice wins
  try {
    const run = await api(`/api/projects/${S.pid}/runs/${name}`);
    if (S.runName !== name) return;
    S.run = run;
    $('#runSelect').value = name;
    renderRun();
  } catch (e) { toast(e.message, 'error'); }
}

function renderRun() {
  const r = S.run, el = $('#runDetail');
  S.viewers = {};
  if (!r) { el.innerHTML = ''; return; }
  const d = r.decision, st = r.settings.settings || {};
  const verdicts = d ? Object.fromEntries(d.verdicts.map((v) => [v.structure_id, v])) : {};
  el.innerHTML = `
    <div class="card">
      <div class="run-head"><span class="badge status-${esc(r.status)}">${esc(statusText(r.status))}</span>
        <span class="muted small">${esc(settingsLabel(st))} · ${esc(Object.values(r.settings.systems || {}).join(', '))} · ${esc(modelTitle(r.settings.model))}, ${esc(r.settings.effort)} effort · sandbox ${esc(r.settings.sandbox)}</span>
        <span class="grow"></span><a class="btn small" href="${esc(r.zip_url)}">Download run (.zip)</a></div>
      ${r.errors.length ? `<div class="warn">${r.errors.map(esc).join('<br>')}</div>` : ''}
      ${d ? `<p class="small"><b>Reviewer:</b> ${esc(d.reasoning)}</p>${d.unresolved.length ? `<p class="small muted">Unresolved: ${esc(d.unresolved.join('; '))}</p>` : ''}`
          : '<p class="muted small">The run ended before the reviewer saw any structure.</p>'}
      ${r.plans.length ? `<details class="sub"><summary>Plan used (${r.plans.length} hypothes${r.plans.length === 1 ? 'is' : 'es'}${r.plan_record ? `, planned in ${r.plan_record.seconds} s` : ''})</summary>${r.plans.map(planCard).join('')}</details>` : ''}
      <details class="sub"><summary>Usage</summary><p class="small muted">Tokens: ${r.usage.input.toLocaleString()} in / ${r.usage.output.toLocaleString()} out${r.usage.cost != null ? ` · ≈ $${r.usage.cost.toFixed(3)}` : ''} (planner, builder and reviewer of this run; the planner shows 0 when the saved plan was reused)</p></details>
    </div>
    ${r.structures.length ? r.structures.map((s) => structureCard(s, verdicts[s.structure_id])).join('')
      : '<div class="card"><p class="empty">No structure was built in this run — see the errors above and the final report.</p></div>'}
    ${r.report ? `<details class="card"><summary><b>Final report</b> <span class="muted small">(final_report.md)</span></summary><div class="md">${renderMd(r.report)}</div></details>` : ''}`;
  const observer = 'IntersectionObserver' in window ? new IntersectionObserver((entries) => {
    entries.forEach((en) => { if (en.isIntersecting) { observer.unobserve(en.target); mountViewer(r.structures.find((s) => s.structure_id === en.target.dataset.sid), r.name); } });
  }, {rootMargin: '200px'}) : null;
  r.structures.forEach((s, i) => {
    const card = $(`.structure[data-sid="${CSS.escape(s.structure_id)}"]`, el);
    if (i < 2 || !observer) mountViewer(s, r.name); else observer.observe(card);
  });
}

function planCard(pl) {
  return `<div class="plan">
    <div class="plan-head"><b>${esc(pl.plan_id)}</b> <span class="badge kind">${esc(pl.model_type)}</span> <span class="muted small">systems ${esc(pl.system_ids.join(', '))}</span></div>
    <div><b>Hypothesis:</b> ${esc(pl.hypothesis)}</div>
    <details class="sub"><summary>Description, crystals, targets, assumptions</summary>
      <div>${esc(pl.description)}</div>
      ${pl.crystals.length ? `<table class="tbl small"><thead><tr><th>Crystal</th><th>Formula</th><th>Space group</th><th>Lattice (Å)</th><th>Source</th></tr></thead><tbody>
        ${pl.crystals.map((c) => `<tr><td>${esc(c.name)}</td><td>${esc(c.formula)}</td><td>${esc(c.space_group)}</td><td>${esc(latticeText(c.lattice))}</td><td>${esc(c.lattice.source)}${c.lattice.evidence_ids.length ? ` (${esc(c.lattice.evidence_ids.join(', '))})` : ''}</td></tr>`).join('')}</tbody></table>` : ''}
      ${pl.targets.length ? `<div><b>Targets:</b> ${pl.targets.map((t) => `${esc(t.property)} = ${esc(t.value)}${t.tolerance ? ` ± ${esc(t.tolerance)}` : ''}`).join(' · ')}</div>` : ''}
      ${pl.assumptions.length ? `<div><b>Assumptions:</b><ul>${pl.assumptions.map((a) => `<li>${esc(a)}</li>`).join('')}</ul></div>` : ''}
    </details>
  </div>`;
}

function structureCard(s, verdict) {
  const i = s.info, [a, b, c] = i.cell_lengths_A, id = cssId(s.structure_id);
  const bad = s.validation.checks.filter((x) => x.status !== 'pass');
  const verdictBadge = s.accepted ? '<span class="badge ok">accepted</span>'
    : (verdict ? `<span class="badge ${verdict.resembles_paper ? 'ok' : 'bad'}">${verdict.resembles_paper ? 'resembles the paper' : 'does not resemble the paper'}</span>` : '');
  return `<div class="card structure" data-sid="${esc(s.structure_id)}">
    <div class="struct-head"><h3>${esc(s.structure_id)}</h3>${verdictBadge}
      ${s.current === false && !s.accepted ? '<span class="badge" title="rebuilt, repaired or from an earlier plan set">superseded</span>' : ''}
      <span class="badge ${s.validation.passed ? 'ok' : 'bad'}">geometry ${s.validation.passed ? '✓' : '✗'}</span>
      <span class="facts">${esc(i.formula)} · ${i.atoms.toLocaleString()} atoms · ${a} × ${b} × ${c} Å · periodic ${esc(i.periodic)}</span>
      <span class="grow"></span>
      <span class="downloads">${Object.entries(s.downloads).filter(([k]) => k !== 'metadata').map(([k, u]) => `<a class="btn small" href="${esc(u)}" download>${esc({poscar: 'POSCAR', cif: 'CIF', extxyz: 'XYZ', script: 'script'}[k] || k)}</a>`).join('')}</span></div>
    <div class="struct-grid">
      <div><div class="viewer" id="v-${id}"></div>
        <div class="viewer-bar"><button data-style="ballstick" class="on">Ball &amp; stick</button><button data-style="spacefill">Space filling</button>
          <button data-spin>Spin</button><button data-reset>Reset</button><span class="legend" id="lg-${id}"></span></div></div>
      <div>
        ${s.plan ? `<p class="small"><b>Hypothesis (${esc(s.plan.model_type)}):</b> ${esc(s.plan.hypothesis)}</p>` : ''}
        ${s.target_checks.length ? `<table class="tbl small"><thead><tr><th>Target</th><th>Measured</th><th></th></tr></thead><tbody>
          ${s.target_checks.map((t) => `<tr><td>${esc(t.property)}</td><td>${esc(t.measured)}</td><td>${t.satisfied ? '✅' : '❌'}</td></tr>`).join('')}</tbody></table>` : ''}
        ${verdict && verdict.issues.length ? `<p class="small"><b>Reviewer issues:</b> ${esc(verdict.issues.join('; '))}</p>` : ''}
      </div>
    </div>
    <details class="sub"><summary>Details: size, compute estimate, how it was built</summary>
      <div class="metrics"><div><b>${i.valence_electrons.toLocaleString()}</b><span>valence e⁻</span></div>
        <div><b>${i.volume_A3 != null ? i.volume_A3.toLocaleString() : '—'}</b><span>volume (Å³)</span></div>
        <div><b>${i.density_g_cm3 != null ? i.density_g_cm3 : '—'}</b><span>density (g/cm³)</span></div></div>
      <div class="small"><b>Compute estimate</b> <span class="muted">(order of magnitude)</span></div>
      <table class="tbl small"><tbody>${Object.entries(s.estimate).map(([k, v]) => `<tr><td>${esc(k)}</td><td>${esc(v)}</td></tr>`).join('')}</tbody></table>
      ${s.method ? `<p class="small muted">How it was built: ${esc(s.method)}</p>` : ''}
      ${s.deviations.length ? `<p class="small muted">Deviations from the plan: ${esc(s.deviations.join('; '))}</p>` : ''}
      ${bad.length ? `<p class="small muted">Checks: ${esc(bad.map((x) => `${x.name} (${x.status}: ${x.detail})`).join('; '))}</p>` : ''}
    </details>
  </div>`;
}

// ---------------------------------------------------------------- 3Dmol.js viewers

function viewerBg() { return getComputedStyle(document.body).getPropertyValue('--viewer-bg').trim() || 'white'; }

function elementColor(symbol) {
  const palette = window.$3Dmol && $3Dmol.elementColors && $3Dmol.elementColors.Jmol;
  const c = palette ? palette[symbol] : null;
  return typeof c === 'number' ? '#' + c.toString(16).padStart(6, '0') : '#999';
}

function applyStyle(v) {
  const look = v.style === 'spacefill' ? {sphere: {scale: 0.95, colorscheme: 'Jmol'}}
    : v.n > 3000 ? {sphere: {scale: 0.3, colorscheme: 'Jmol'}}  // bonds for this many atoms make the page slow
    : {sphere: {scale: 0.28, colorscheme: 'Jmol'}, stick: {radius: 0.12, colorscheme: 'Jmol'}};
  v.viewer.setStyle({}, look);
}

async function mountViewer(s, runName) {
  if (!s) return;
  const id = cssId(s.structure_id), box = $(`#v-${id}`);
  if (!box || S.viewers[s.structure_id]) return;
  if (!window.$3Dmol) { box.innerHTML = '<p class="empty">The 3D viewer needs 3Dmol.js from the internet; the files still download.</p>'; return; }
  const probe = document.createElement('canvas');
  if (!(probe.getContext('webgl2') || probe.getContext('webgl'))) { box.innerHTML = '<p class="empty">This browser has no WebGL, which the 3D viewer needs; the files still download.</p>'; return; }
  try {
    const data = await api(`/api/projects/${S.pid}/runs/${runName}/structures/${encodeURIComponent(s.structure_id)}/model`);
    if (!document.body.contains(box)) return;  // the run view was replaced while loading
    const viewer = $3Dmol.createViewer(box, {backgroundColor: viewerBg()});
    const model = viewer.addModel(data.text, data.format, {doAssembly: false, duplicateAssemblyAtoms: false});
    const v = {viewer, model, n: data.n_atoms, style: 'ballstick', spinning: false};
    S.viewers[s.structure_id] = v;
    applyStyle(v);
    if (data.periodic) viewer.addUnitCell(model, {box: {color: '#888'}});
    viewer.zoomTo();
    viewer.render();
    $(`#lg-${id}`).innerHTML = data.elements.map((e) => `<span class="chip-el"><i style="background:${elementColor(e)}"></i>${esc(e)}</span>`).join('');
    const card = box.closest('.structure');
    $$('[data-style]', card).forEach((b) => (b.onclick = () => {
      v.style = b.dataset.style;
      $$('[data-style]', card).forEach((x) => x.classList.toggle('on', x === b));
      applyStyle(v);
      viewer.render();
    }));
    $('[data-spin]', card).onclick = () => { v.spinning = !v.spinning; viewer.spin(v.spinning ? 'y' : false); };
    $('[data-reset]', card).onclick = () => { viewer.zoomTo(); viewer.render(); };
  } catch (e) { box.innerHTML = `<p class="empty">Viewer failed: ${esc(e && e.message ? e.message : e)}</p>`; }
}

function resizeViewers() {
  Object.values(S.viewers).forEach((v) => { try { v.viewer.resize(); v.viewer.render(); } catch (e) { /* viewer gone */ } });
}

init();
