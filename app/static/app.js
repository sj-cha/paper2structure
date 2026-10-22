/* Paper -> Structure web tool: three panels over the JSON API of app/server.py.
   Panel 1 uploads a paper and runs the summarizer; panel 2 shows the summary, takes the material
   system(s) to model and runs the evidence analysis and the planner; panel 3 takes the structure
   settings, runs construction and review, and shows the structures in 3Dmol.js. */
'use strict';

const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];
const esc = (s) => String(s ?? '').replace(/[&<>"']/g, (c) => ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'}[c]));
const cssId = (s) => String(s).replace(/[^A-Za-z0-9_-]/g, '_');

const S = {
  info: null, projects: [], pid: null, project: null, run: null, runName: null,
  step: 1, job: null, timer: null, viewers: {}, atomsAuto: true,
  settings: defaultSettings(),
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

function modelTitle(name) {
  const m = S.info && S.info.models.find((x) => x.name === name);
  return m ? m.title : name;
}

function settingsLabel(s) {
  if (!s || !s.method) return '';
  return `${s.method}, ≤${s.max_atoms} atoms, realism ${Number(s.realism).toFixed(2)}${s.model_type ? `, ${s.model_type}` : ''}`;
}

function statusText(status) {
  return {accepted: 'accepted by the reviewer', inconclusive: 'built, but the reviewer was not convinced',
          revision_budget_exhausted: 'built; the reviewer wanted another loop', failed: 'failed', unfinished: 'unfinished'}[status] || status;
}

function latticeText(l) {
  return ['a', 'b', 'c', 'alpha', 'beta', 'gamma'].filter((k) => l[k] != null).map((k) => `${k}=${l[k]}`).join(', ');
}

function elapsed(job) {
  if (!job.started) return '';
  const s = Math.max(0, Math.round((job.finished || Date.now() / 1000) - job.started));
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, '0')}`;
}

function isBusy() {
  return !!(S.job && S.job.project_id === S.pid && (S.job.status === 'queued' || S.job.status === 'running'));
}

// The final report is the reviewer's markdown: rendered, with scripts and event handlers stripped.
function renderMd(text) {
  if (!window.marked) return `<pre>${esc(text)}</pre>`;
  const tpl = document.createElement('template');
  tpl.innerHTML = marked.parse(text);
  $$('script, iframe, object, embed, style', tpl.content).forEach((n) => n.remove());
  $$('*', tpl.content).forEach((n) => {
    [...n.attributes].forEach((a) => {
      if (a.name.startsWith('on') || /^\s*javascript:/i.test(a.value)) n.removeAttribute(a.name);
    });
  });
  return tpl.innerHTML;
}

// ---------------------------------------------------------------- start-up

async function init() {
  try {
    S.info = await api('/api/info');
  } catch (e) {
    toast(`Cannot reach the server: ${e.message}`, 'error');
    return;
  }
  renderModelList();
  $('#keyBox').classList.toggle('hidden', S.info.has_key);
  $$('.step-btn').forEach((b) => (b.onclick = () => showStep(+b.dataset.step)));
  wireStep1();
  wireStep3();
  await loadProjects();
  const h = new URLSearchParams(location.hash.slice(1));
  if (h.get('p')) await openProject(h.get('p'), +(h.get('s') || 2));
  else showStep(1);
}

function showStep(n) {
  const p = S.project;
  const can = {1: true, 2: !!(p && p.summary), 3: !!(p && p.summary)};  // generate plans first when panel 2 was skipped
  if (!can[n]) n = 1;
  S.step = n;
  $$('.step').forEach((s) => s.classList.toggle('hidden', s.id !== `step${n}`));
  $$('.step-btn').forEach((b) => { const k = +b.dataset.step; b.classList.toggle('on', k === n); b.disabled = !can[k]; });
  history.replaceState(null, '', S.pid ? `#p=${S.pid}&s=${n}` : '#');
  if (n === 3) setTimeout(resizeViewers, 50);
}

function render() {
  renderBar();
  renderProjects();
  renderStep2();
  renderStep3();
  updateButtons();
}

function updateButtons() {
  const busy = isBusy();
  ['#summarizeBtn', '#analyzeBtn', '#generateBtn', '#resummarize'].forEach((sel) => { const b = $(sel); if (b) b.disabled = busy; });
  const g = $('#generateBtn');
  if (g && S.project && S.project.project.model === 'dry-run') { g.disabled = true; g.title = 'The dry-run model cannot build structures: pick a real model on panel 2.'; }
  else if (g) g.title = '';
}

function renderBar() {
  const el = $('#projectBar'), p = S.project;
  if (!p) { el.innerHTML = '<span class="muted">No paper open</span>'; return; }
  el.innerHTML = `<span class="muted">Paper:</span> <strong>${esc(p.project.title)}</strong> <span class="muted">· ${esc(modelTitle(p.project.model))} · ${esc(p.project.effort)} effort</span>`;
}

// ---------------------------------------------------------------- projects

async function loadProjects() {
  S.projects = await api('/api/projects');
  renderProjects();
}

function renderProjects() {
  const el = $('#projectList');
  if (!S.projects.length) { el.innerHTML = '<p class="muted">Nothing yet. Upload a paper to start.</p>'; return; }
  el.innerHTML = S.projects.map((p) => `
    <div class="proj-item ${p.id === S.pid ? 'on' : ''}" data-id="${esc(p.id)}">
      <div class="proj-title">${esc(p.title)}</div>
      <div class="proj-meta">${esc(p.created)} · ${esc(modelTitle(p.model))} · ${esc(p.effort)} effort · ${esc(p.pdf_name)}</div>
      <div class="badges">
        <span class="badge ${p.summary ? 'ok' : ''}">summary</span>
        <span class="badge ${p.plan ? 'ok' : ''}">plan</span>
        <span class="badge ${p.runs ? 'ok' : ''}">${p.runs} run${p.runs === 1 ? '' : 's'}</span>
        ${p.active_job ? '<span class="badge busy">working…</span>' : ''}
      </div>
      <button class="icon del" title="Delete this paper and everything made from it" data-del="${esc(p.id)}">×</button>
    </div>`).join('');
  $$('.proj-item', el).forEach((item) => (item.onclick = (e) => { if (e.target.dataset.del) return; openProject(item.dataset.id, 2); }));
  $$('[data-del]', el).forEach((b) => (b.onclick = async (e) => {
    e.stopPropagation();
    if (!confirm('Delete this paper and all of its results?')) return;
    try {
      await api(`/api/projects/${b.dataset.del}`, {method: 'DELETE'});
      if (S.pid === b.dataset.del) { S.pid = null; S.project = null; S.run = null; showStep(1); }
      await loadProjects();
      render();
    } catch (err) { toast(err.message, 'error'); }
  }));
}

async function openProject(pid, step = 2) {
  try {
    S.pid = pid;
    S.project = await api(`/api/projects/${pid}`);
    S.run = null;
    S.runName = S.project.project.current_run;
    clearInterval(S.timer);
    S.timer = S.job = null;
    $$('.progress').forEach((b) => b.classList.add('hidden'));  // the previous paper's job
    initSettingsFromProject();
    render();
    if (S.project.active_job) watch(S.project.active_job);
    showStep(step);
  } catch (e) { toast(e.message, 'error'); }
}

async function refreshProject() {
  if (!S.pid) return;
  S.project = await api(`/api/projects/${S.pid}`);
  render();
}

function initSettingsFromProject() {
  const p = S.project;
  const base = p.project.last_settings || (p.plan && p.plan.settings) || null;
  S.settings = Object.assign(defaultSettings(), base || {});
  S.settings.model_type = S.settings.model_type || '';
  if (p.summary) {
    const ids = p.summary.systems.map((s) => s.system_id);
    S.settings.systems = S.settings.systems.filter((i) => ids.includes(i));
    if (!S.settings.systems.length && ids.length) S.settings.systems = [ids[0]];
    const ready = p.summary.techniques.filter((t) => t.agent === 'ready').map((t) => t.name);
    S.settings.techniques = S.settings.techniques.filter((t) => ready.includes(t));
    if (!S.settings.techniques.length) S.settings.techniques = ready;
  }
  S.atomsAuto = !base || base.max_atoms === defaultMaxAtoms(base.method, base.realism);
}

// ---------------------------------------------------------------- jobs

function stepOfJob(kind) { return {summarize: 1, analyze: 2, generate: 3}[kind] || 1; }

function renderJob(job) {
  $$('.progress').forEach((b) => b.classList.add('hidden'));
  const box = $(`#progress${stepOfJob(job.kind)}`);
  box.classList.remove('hidden');
  const title = {summarize: 'Reading the paper (Agent 1)', analyze: 'Analyzing the evidence and planning (Agents 2-3)',
                 generate: 'Building and reviewing structures (Agents 4-5)'}[job.kind] || job.kind;
  const lines = job.log.slice(-120);
  box.innerHTML = `<div class="job-head"><span class="spinner ${job.status}"></span><strong>${esc(title)}</strong>
      <span class="badge ${job.status}">${esc(job.status)}</span><span class="muted">${elapsed(job)}</span></div>
    ${job.error ? `<div class="error">${esc(job.error)}</div>` : ''}
    <pre class="log">${lines.map(esc).join('\n') || 'starting…'}</pre>`;
  const pre = $('pre', box);
  pre.scrollTop = pre.scrollHeight;
}

function watch(jobId) {
  clearInterval(S.timer);
  const tick = async () => {
    try {
      const job = await api(`/api/jobs/${jobId}`);
      S.job = job;
      renderJob(job);
      updateButtons();
      if (job.status === 'done' || job.status === 'failed') {
        clearInterval(S.timer);
        S.timer = null;
        await refreshProject();
        await loadProjects();
        if (job.status === 'failed') { toast(job.error || 'The step failed', 'error'); return; }
        if (job.kind === 'summarize') { toast('Summary ready: pick the material system on panel 2'); showStep(2); }
        if (job.kind === 'analyze') { toast('Plan ready: check it below, then continue to panel 3'); $('#s2results').scrollIntoView({behavior: 'smooth'}); }
        if (job.kind === 'generate' && job.result && job.result.run) { S.runName = job.result.run; await loadRun(S.runName); toast(`Run finished: ${statusText(job.result.status)}`); }
      }
    } catch (e) { clearInterval(S.timer); S.timer = null; toast(e.message, 'error'); }
  };
  tick();
  S.timer = setInterval(tick, 2000);
}

// ---------------------------------------------------------------- panel 1

function renderModelList() {
  $('#modelList').innerHTML = S.info.models.map((m) => `
    <label class="radio-card"><input type="radio" name="model" value="${esc(m.name)}" ${m.default ? 'checked' : ''}>
      <span class="rc-title">${esc(m.title)}</span><span class="rc-price">${esc(m.price)}</span>
      ${m.note ? `<span class="rc-note">${esc(m.note)}</span>` : ''}</label>`).join('');
}

function selectedModel() {
  const r = $('input[name=model]:checked');
  return r ? r.value : (S.info.models.find((m) => m.default) || S.info.models[0]).name;
}

function wireStep1() {
  const input = $('#pdf'), drop = $('#drop');
  input.onchange = () => { $('#dropText').textContent = input.files[0] ? input.files[0].name : 'Drop a PDF here or click to choose'; };
  ['dragenter', 'dragover'].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add('over'); }));
  ['dragleave', 'drop'].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.remove('over'); }));
  drop.addEventListener('drop', (e) => { if (e.dataTransfer.files[0]) { input.files = e.dataTransfer.files; input.onchange(); } });
  $('#summarizeBtn').onclick = startSummary;
  $('#keyBtn').onclick = async () => {
    try {
      await api('/api/key', {method: 'POST', body: {key: $('#keyInput').value}});
      S.info.has_key = true;
      $('#keyBox').classList.add('hidden');
      toast('Key set for this server session');
    } catch (e) { toast(e.message, 'error'); }
  };
}

async function startSummary() {
  const file = $('#pdf').files[0];
  if (!file) { toast('Choose a PDF first', 'error'); return; }
  const form = new FormData();
  form.append('pdf', file);
  form.append('model', selectedModel());
  form.append('effort', $('#effort').value);
  form.append('analyze_figures', $('#figures').checked ? 'true' : 'false');
  form.append('max_figures', $('#maxFigures').value);
  const btn = $('#summarizeBtn');
  btn.disabled = true;
  try {
    const r = await api('/api/projects', {method: 'POST', form});
    S.pid = r.project_id;
    S.project = await api(`/api/projects/${S.pid}`);
    S.run = null;
    initSettingsFromProject();
    await loadProjects();
    render();
    watch(r.job_id);
  } catch (e) { toast(e.message, 'error'); btn.disabled = false; }
}

// ---------------------------------------------------------------- panel 2

function renderStep2() {
  const top = $('#s2top'), res = $('#s2results'), p = S.project;
  if (!p || !p.summary) {
    top.innerHTML = '<div class="card"><p class="muted">Upload and summarize a paper on panel 1 first.</p></div>';
    res.innerHTML = '';
    return;
  }
  const m = p.summary.metadata, st = S.settings, sum = p.project.summary || {};
  const sysRows = p.summary.systems.map((s) => {
    const lat = (s.structure.lattice_parameters || []).map((x) => `${x.quantity}=${x.value}${x.unit || ''}`).join(', ');
    const vars = Object.entries(s.composition_variables || {}).map(([k, v]) => `${k}=${v}`).join(', ');
    const on = st.systems.includes(s.system_id);
    return `<tr class="${on ? 'on' : ''}" data-sys="${esc(s.system_id)}"><td><input type="checkbox" ${on ? 'checked' : ''}></td>
      <td><b>${esc(s.system_id)}</b></td><td>${esc(s.name)}${vars ? `<div class="muted small">${esc(vars)}</div>` : ''}</td>
      <td>${esc(s.form || '')}</td><td>${esc(s.structure.space_group || s.structure.crystal_system || '')}</td>
      <td>${esc(lat)}</td><td>${esc(s.synthesis.method || '')}</td></tr>`;
  }).join('');
  const techChips = p.summary.techniques.map((t) => {
    const ready = t.agent === 'ready', on = st.techniques.includes(t.name);
    const why = t.agent === 'stub' ? 'no analysis agent for this technique yet' : 'not in the technique vocabulary';
    return `<label class="chip ${ready ? '' : 'off'}" title="${ready ? '' : why}">
      <input type="checkbox" data-tech="${esc(t.name)}" ${on ? 'checked' : ''} ${ready ? '' : 'disabled'}> ${esc(t.name)}
      <span class="muted">${t.count}</span>${ready ? '' : ' <span class="muted">(no agent)</span>'}</label>`;
  }).join('');
  const evidence = p.summary.techniques.map((t) => {
    const items = p.summary.evidence.filter((e) => e.technique === t.name);
    return `<details class="ev"><summary><b>${esc(t.name)}</b> · ${items.length} item${items.length === 1 ? '' : 's'}${t.figures ? ` · ${t.figures} from figures` : ''}</summary>
      ${items.map((e) => `<div class="ev-item"><div class="ev-text"><span class="eid">${esc(e.evidence_id)}</span>
          <span class="muted small">${esc(e.source_ref)} · ${esc(e.system_ids.join(', '))}</span><div>${esc(e.summary)}</div>
          ${e.measurements.length ? `<div class="small">${e.measurements.map((x) => `${esc(x.quantity)} = ${esc(x.value)} ${esc(x.unit || '')}`).join(' · ')}</div>` : ''}
          ${e.quote ? `<div class="quote">“${esc(e.quote)}”</div>` : ''}</div>
        ${e.image_url ? `<a href="${e.image_url}" target="_blank"><img class="thumb" src="${e.image_url}" alt=""></a>` : ''}</div>`).join('')}
    </details>`;
  }).join('');
  const modelOptions = S.info.models.map((mm) => `<option value="${esc(mm.name)}" ${mm.name === p.project.model ? 'selected' : ''}>${esc(mm.title)} · ${esc(mm.price)}</option>`).join('');
  top.innerHTML = `
    <div class="card">
      <h2>${esc(m.title)}</h2>
      <p class="muted small">${esc((m.authors || []).join(', '))}${m.year ? ` · ${m.year}` : ''}${m.doi ? ` · DOI ${esc(m.doi)}` : ''}
        · read by ${esc(modelTitle(p.project.model))}${sum.seconds ? ` in ${sum.seconds} s` : ''}${sum.cost != null ? ` · ≈ $${sum.cost.toFixed(3)}` : ''}</p>
      <p>${esc(m.summary)}</p>
      ${sum.errors && sum.errors.length ? `<div class="warn">Some steps had problems:<br>${sum.errors.map(esc).join('<br>')}</div>` : ''}
      <div class="row-end"><button class="ghost small" id="resummarize">Read the paper again</button></div>
    </div>
    <div class="card">
      <h2>Material systems <span class="muted">— tick the one(s) to build</span></h2>
      <table class="tbl" id="sysTable"><thead><tr><th></th><th>ID</th><th>Name</th><th>Form</th><th>Space group</th><th>Lattice parameters</th><th>Synthesis</th></tr></thead>
        <tbody>${sysRows}</tbody></table>
    </div>
    <div class="card"><h2>Evidence by technique <span class="muted">${p.summary.evidence.length} items</span></h2>${evidence || '<p class="muted">No evidence was extracted.</p>'}</div>
    <div class="card">
      <h2>Analyze the evidence and plan</h2>
      <div class="field"><label>Techniques to analyze <span class="muted">one agent each (Agent 2)</span></label><div class="chips" id="techChips">${techChips || '<span class="muted">none found</span>'}</div></div>
      <div class="settings-grid">
        <label>Hypotheses to plan <span class="muted">one structure each</span>
          <select id="maxPlans">${[1, 2, 3].map((n) => `<option ${st.max_plans === n ? 'selected' : ''}>${n}</option>`).join('')}</select></label>
        <label>Model <select id="projModel">${modelOptions}</select></label>
        <label>Reasoning effort <select id="projEffort">${S.info.efforts.map((e) => `<option ${e === p.project.effort ? 'selected' : ''}>${e}</option>`).join('')}</select></label>
      </div>
      <p class="hint">The planner (Agent 3) uses the structure settings of panel 3 (currently ${esc(settingsLabel(st))}). The evidence analysis is kept and reused when you plan again or change the settings.</p>
      <div class="actions"><button class="primary" id="analyzeBtn">Analyze evidence &amp; plan</button><span class="hint">One to three minutes.</span></div>
    </div>`;

  $$('#sysTable tbody tr').forEach((tr) => {
    tr.onclick = (e) => {
      const box = $('input', tr);
      if (e.target !== box) box.checked = !box.checked;
      const id = tr.dataset.sys;
      st.systems = box.checked ? [...new Set([...st.systems, id])] : st.systems.filter((x) => x !== id);
      tr.classList.toggle('on', box.checked);
      renderPlanStatus();
    };
  });
  $$('#techChips input').forEach((c) => (c.onchange = () => {
    st.techniques = $$('#techChips input').filter((x) => x.checked).map((x) => x.dataset.tech);
    renderPlanStatus();
  }));
  $('#maxPlans').onchange = () => { st.max_plans = +$('#maxPlans').value; renderPlanStatus(); };
  $('#projModel').onchange = () => updateProjectSettings({model: $('#projModel').value});
  $('#projEffort').onchange = () => updateProjectSettings({effort: $('#projEffort').value});
  $('#analyzeBtn').onclick = startAnalyze;
  $('#resummarize').onclick = async () => {
    if (!confirm('Read the paper again with the current model? The evidence analysis and plan made from the old summary are discarded (finished runs are kept).')) return;
    try { const r = await api(`/api/projects/${S.pid}/summarize`, {method: 'POST', body: {}}); showStep(1); watch(r.job_id); }
    catch (e) { toast(e.message, 'error'); }
  };
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

function validSettings() {
  if (!S.settings.systems.length) { toast('Tick at least one material system', 'error'); return false; }
  if (!S.settings.techniques.length) { toast('Tick at least one technique to analyze', 'error'); return false; }
  return true;
}

function payload() { return {...S.settings, model_type: S.settings.model_type || null}; }

async function startAnalyze() {
  if (!validSettings()) return;
  const btn = $('#analyzeBtn');
  btn.disabled = true;
  try { const r = await api(`/api/projects/${S.pid}/analyze`, {method: 'POST', body: payload()}); watch(r.job_id); }
  catch (e) { toast(e.message, 'error'); btn.disabled = false; }
}

function renderStep2Results() {
  const res = $('#s2results'), p = S.project, plan = p.plan;
  const char = plan ? p.characterizations.find((c) => c.key === plan.characterization) : p.characterizations[0];
  let html = '';
  if (char) {
    html += `<div class="card"><h2>What the evidence says <span class="muted">Agent 2 · ${esc(char.reports.join(', '))} · ${esc(modelTitle(char.model))}, ${esc(char.effort)} effort · ${char.seconds} s</span></h2>
      ${char.reports_detail.map((r) => `<details class="ev" open><summary><b>${esc(r.technique)}</b> · ${r.observations} observations · ${r.constraints.length} structural constraints</summary>
        ${r.constraints.length ? `<table class="tbl small"><thead><tr><th>Property</th><th>Value</th><th>Systems</th><th>Confidence</th><th>How</th></tr></thead><tbody>
          ${r.constraints.map((c) => `<tr><td>${esc(c.property)}</td><td>${esc(c.value)} ${esc(c.unit || '')}</td><td>${esc(c.system_ids.join(', '))}</td>
            <td><span class="badge conf-${esc(c.confidence)}">${esc(c.confidence)}</span></td><td class="muted">${esc(c.derivation)}</td></tr>`).join('')}</tbody></table>` : '<p class="muted small">No structural constraints from this technique.</p>'}
        ${r.findings.length ? `<ul class="small">${r.findings.map((f) => `<li>${esc(f.claim)} <span class="muted">(${esc(f.confidence)})</span></li>`).join('')}</ul>` : ''}
        ${r.caveats.length ? `<details><summary class="muted small">Caveats</summary><ul class="small">${r.caveats.map((c) => `<li>${esc(c)}</li>`).join('')}</ul></details>` : ''}
      </details>`).join('')}
      ${Object.keys(char.skipped || {}).length ? `<p class="muted small">Not analyzed: ${esc(Object.entries(char.skipped).map(([k, v]) => `${k} (${v})`).join('; '))}</p>` : ''}
      ${char.errors && char.errors.length ? `<div class="warn">${char.errors.map(esc).join('<br>')}</div>` : ''}
    </div>`;
  }
  if (plan) {
    const ps = plan.plans.plans;
    html += `<div class="card"><h2>The plan <span class="muted">Agent 3 · ${ps.length} hypothes${ps.length === 1 ? 'is' : 'es'} · for ${esc(settingsLabel(plan.settings))} · ${plan.seconds} s${plan.cost != null ? ` · ≈ $${plan.cost.toFixed(3)}` : ''}</span></h2>
      <p class="muted small">${esc(plan.plans.rationale)}</p>
      ${ps.map(planCard).join('')}
      <div class="actions"><button class="primary" id="toStep3">Continue to structures →</button></div></div>`;
  }
  res.innerHTML = html;
  const b = $('#toStep3');
  if (b) b.onclick = () => showStep(3);
}

function planCard(pl) {
  return `<div class="plan">
    <div class="plan-head"><b>${esc(pl.plan_id)}</b> <span class="badge">${esc(pl.model_type)}</span> <span class="muted small">systems ${esc(pl.system_ids.join(', '))}</span></div>
    <div><b>Hypothesis:</b> ${esc(pl.hypothesis)}</div>
    <div class="small">${esc(pl.description)}</div>
    ${pl.crystals.length ? `<table class="tbl small"><thead><tr><th>Crystal</th><th>Formula</th><th>Space group</th><th>Lattice (Å)</th><th>Source</th></tr></thead><tbody>
      ${pl.crystals.map((c) => `<tr><td>${esc(c.name)}</td><td>${esc(c.formula)}</td><td>${esc(c.space_group)}</td><td>${esc(latticeText(c.lattice))}</td>
        <td>${esc(c.lattice.source)}${c.lattice.evidence_ids.length ? ` (${esc(c.lattice.evidence_ids.join(', '))})` : ''}</td></tr>`).join('')}</tbody></table>` : ''}
    ${pl.targets.length ? `<div class="small"><b>Targets:</b> ${pl.targets.map((t) => `${esc(t.property)} = ${esc(t.value)}${t.tolerance ? ` ± ${esc(t.tolerance)}` : ''}`).join(' · ')}</div>` : ''}
    ${pl.assumptions.length ? `<details><summary class="small muted">${pl.assumptions.length} assumption${pl.assumptions.length === 1 ? '' : 's'}</summary><ul class="small">${pl.assumptions.map((a) => `<li>${esc(a)}</li>`).join('')}</ul></details>` : ''}
  </div>`;
}

// ---------------------------------------------------------------- panel 3

function wireStep3() {
  $$('#method button').forEach((b) => (b.onclick = () => { S.settings.method = b.dataset.v; syncAtoms(); renderStep3Form(); }));
  $('#realism').oninput = () => { S.settings.realism = +$('#realism').value; syncAtoms(); renderStep3Form(); };
  $('#maxAtoms').onchange = () => {
    S.settings.max_atoms = Math.max(10, Math.min(50000, +$('#maxAtoms').value || 10));
    S.atomsAuto = false;
    renderStep3Form();
  };
  $('#atomsAuto').onchange = () => { S.atomsAuto = $('#atomsAuto').checked; syncAtoms(); renderStep3Form(); };
  $('#modelType').onchange = () => { S.settings.model_type = $('#modelType').value; renderPlanStatus(); };
  $('#seed').onchange = () => { S.settings.seed = +$('#seed').value || 0; };
  $('#maxRevisions').onchange = () => { S.settings.max_revisions = Math.max(0, Math.min(2, +$('#maxRevisions').value || 0)); };
  $('#maxRepairs').onchange = () => { S.settings.max_repairs = Math.max(0, Math.min(3, +$('#maxRepairs').value || 0)); };
  $('#generateBtn').onclick = startGenerate;
  $('#runSelect').onchange = () => loadRun($('#runSelect').value);
  S.info.model_types.forEach((t) => $('#modelType').add(new Option(t, t)));
}

function syncAtoms() {
  if (S.atomsAuto) S.settings.max_atoms = defaultMaxAtoms(S.settings.method, S.settings.realism);
}

function renderStep3() {
  if (!S.project) return;
  renderStep3Form();
  renderRuns();
}

function renderStep3Form() {
  const st = S.settings;
  $$('#method button').forEach((b) => b.classList.toggle('on', b.dataset.v === st.method));
  $('#realism').value = st.realism;
  $('#realismVal').textContent = Number(st.realism).toFixed(2);
  $('#maxAtoms').value = st.max_atoms;
  $('#atomsAuto').checked = S.atomsAuto;
  $('#modelType').value = st.model_type || '';
  $('#seed').value = st.seed;
  $('#maxRevisions').value = st.max_revisions;
  $('#maxRepairs').value = st.max_repairs;
  renderPlanStatus();
}

function planMatches(plan) {
  const a = plan.settings, b = S.settings, p = S.project.project;
  const same = (x, y) => JSON.stringify([...x].sort()) === JSON.stringify([...y].sort());
  return plan.model === p.model && plan.effort === p.effort && same(a.systems, b.systems) && same(a.techniques, b.techniques)
    && a.max_plans === b.max_plans && a.method === b.method && Math.abs(a.realism - b.realism) < 1e-9
    && a.max_atoms === b.max_atoms && (a.model_type || '') === (b.model_type || '');
}

function renderPlanStatus() {
  const p = S.project, el = $('#planStatus');
  if (!p) return;
  const names = p.summary ? Object.fromEntries(p.summary.systems.map((s) => [s.system_id, s.name])) : {};
  const chosen = S.settings.systems.map((i) => `${i} (${names[i] || i})`).join(', ') || 'no system chosen';
  if (!p.plan) {
    el.innerHTML = `<b>No plan yet.</b> Generate analyzes the evidence and plans with these settings first (as panel 2 would), then builds. Systems: ${esc(chosen)}.`;
    return;
  }
  el.innerHTML = planMatches(p.plan)
    ? `✅ The plan on panel 2 (${esc(settingsLabel(p.plan.settings))}, for ${esc(chosen)}) matches these settings: Generate builds it directly.`
    : `↻ These settings differ from the plan on panel 2 (${esc(settingsLabel(p.plan.settings))}): Generate plans again with the new limits first (about a minute), then builds. The evidence analysis is reused.`;
}

async function startGenerate() {
  if (!S.pid || !validSettings()) return;
  const btn = $('#generateBtn');
  btn.disabled = true;
  try { const r = await api(`/api/projects/${S.pid}/generate`, {method: 'POST', body: payload()}); watch(r.job_id); }
  catch (e) { toast(e.message, 'error'); btn.disabled = false; }
}

function renderRuns() {
  const p = S.project, sel = $('#runSelect'), bar = $('#runsBar');
  if (!p || !p.runs.length) { bar.classList.add('hidden'); $('#runDetail').innerHTML = ''; S.run = null; return; }
  bar.classList.remove('hidden');
  sel.innerHTML = p.runs.map((r) => `<option value="${esc(r.name)}">${esc(r.created)} · ${esc(statusText(r.status))} · ${r.n_accepted}/${r.n_structures} accepted · ${esc(settingsLabel(r.settings))} · ${esc(Object.values(r.systems || {}).join(', '))}</option>`).join('');
  if (!S.runName || !p.runs.some((r) => r.name === S.runName)) S.runName = p.runs[0].name;
  sel.value = S.runName;
  if (!S.run || S.run.name !== S.runName) loadRun(S.runName);
}

async function loadRun(name) {
  try {
    S.runName = name;
    S.run = await api(`/api/projects/${S.pid}/runs/${name}`);
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
        <span class="grow"></span><a class="btn small" href="${r.zip_url}">Download run (.zip)</a></div>
      ${r.errors.length ? `<div class="warn">${r.errors.map(esc).join('<br>')}</div>` : ''}
      ${d ? `<p><b>Reviewer (Agent 5):</b> ${esc(d.reasoning)}</p>${d.unresolved.length ? `<p class="small muted">Unresolved: ${esc(d.unresolved.join('; '))}</p>` : ''}`
          : '<p class="muted">The run ended before the reviewer saw any structure.</p>'}
      ${r.usage.input ? `<p class="small muted">Tokens: ${r.usage.input.toLocaleString()} in / ${r.usage.output.toLocaleString()} out${r.usage.cost != null ? ` · ≈ $${r.usage.cost.toFixed(3)}` : ''} (planner, builder and reviewer)</p>` : ''}
    </div>
    ${r.structures.length ? r.structures.map((s) => structureCard(s, verdicts[s.structure_id])).join('')
      : '<div class="card"><p class="muted">No structure was built in this run: see the errors above and the final report.</p></div>'}
    ${r.report ? `<details class="card"><summary><b>Final report</b> <span class="muted small">(final_report.md)</span></summary><div class="md">${renderMd(r.report)}</div></details>` : ''}`;
  r.structures.forEach(mountViewer);
}

function structureCard(s, verdict) {
  const i = s.info, [a, b, c] = i.cell_lengths_A, id = cssId(s.structure_id);
  const bad = s.validation.checks.filter((x) => x.status !== 'pass');
  return `<div class="card structure" data-sid="${esc(s.structure_id)}">
    <div class="run-head"><h3>${esc(s.structure_id)}</h3>
      ${s.accepted ? '<span class="badge ok">accepted</span>' : (verdict ? `<span class="badge ${verdict.resembles_paper ? 'ok' : 'bad'}">${verdict.resembles_paper ? 'resembles the paper' : 'does not resemble the paper'}</span>` : '')}
      <span class="badge ${s.validation.passed ? 'ok' : 'bad'}">geometry checks ${s.validation.passed ? 'passed' : 'failed'}</span></div>
    <div class="struct-grid">
      <div><div class="viewer" id="v-${id}"></div>
        <div class="viewer-bar"><button data-style="ballstick" class="on">Ball &amp; stick</button><button data-style="spacefill">Space filling</button>
          <button data-spin>Spin</button><button data-reset>Reset view</button><span class="legend" id="lg-${id}"></span></div>
        <p class="hint">Drag to rotate · scroll to zoom · right-drag to move</p></div>
      <div>
        <div class="metrics"><div><b>${i.atoms.toLocaleString()}</b><span>atoms</span></div><div><b>${i.valence_electrons.toLocaleString()}</b><span>valence e⁻</span></div><div><b>${esc(i.periodic)}</b><span>periodic</span></div></div>
        <table class="tbl small"><tbody>
          <tr><td>Formula</td><td>${esc(i.formula)}</td></tr><tr><td>Cell (Å)</td><td>${a} × ${b} × ${c}</td></tr>
          ${i.volume_A3 != null ? `<tr><td>Volume (Å³)</td><td>${i.volume_A3.toLocaleString()}</td></tr><tr><td>Density (g/cm³)</td><td>${i.density_g_cm3}</td></tr>` : ''}
        </tbody></table>
        <div class="small" style="margin-top:8px"><b>Compute estimate</b> <span class="muted">(order of magnitude)</span></div>
        <table class="tbl small"><tbody>${Object.entries(s.estimate).map(([k, v]) => `<tr><td>${esc(k)}</td><td>${esc(v)}</td></tr>`).join('')}</tbody></table>
        <div class="downloads">${Object.entries(s.downloads).map(([k, u]) => `<a class="btn small" href="${u}" download>${esc(k)}</a>`).join('')}</div>
      </div>
    </div>
    ${s.plan ? `<p class="small"><b>Hypothesis (${esc(s.plan.model_type)}):</b> ${esc(s.plan.hypothesis)}</p>` : ''}
    ${s.method ? `<p class="small muted">How it was built: ${esc(s.method)}</p>` : ''}
    ${s.target_checks.length ? `<table class="tbl small"><thead><tr><th>Target</th><th>Measured</th><th></th></tr></thead><tbody>
      ${s.target_checks.map((t) => `<tr><td>${esc(t.property)}</td><td>${esc(t.measured)}</td><td>${t.satisfied ? '✅' : '❌'}</td></tr>`).join('')}</tbody></table>` : ''}
    ${s.deviations.length ? `<p class="small muted">Deviations from the plan: ${esc(s.deviations.join('; '))}</p>` : ''}
    ${verdict && verdict.issues.length ? `<p class="small"><b>Reviewer issues:</b> ${esc(verdict.issues.join('; '))}</p>` : ''}
    ${bad.length ? `<p class="small muted">Checks: ${esc(bad.map((x) => `${x.name} (${x.status}: ${x.detail})`).join('; '))}</p>` : ''}
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

async function mountViewer(s) {
  const id = cssId(s.structure_id), box = $(`#v-${id}`);
  if (!box) return;
  if (!window.$3Dmol) { box.innerHTML = '<p class="muted" style="padding:12px">The 3D viewer needs 3Dmol.js from the internet; the files below still download.</p>'; return; }
  try {
    const data = await api(`/api/projects/${S.pid}/runs/${S.runName}/structures/${encodeURIComponent(s.structure_id)}/model`);
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
  } catch (e) { box.innerHTML = `<p class="muted" style="padding:12px">Viewer failed: ${esc(e.message)}</p>`; }
}

function resizeViewers() {
  Object.values(S.viewers).forEach((v) => { try { v.viewer.resize(); v.viewer.render(); } catch (e) { /* viewer gone */ } });
}

init();
