'use strict';
const $ = id => document.getElementById(id);
const label = value => value === 0 ? 'Left-hand imagery' : 'Right-hand imagery';
let token = '', selected = null, listVersion = 0, detailVersion = 0;
const node = (tag, text = '', cls = '') => {const item = document.createElement(tag); item.textContent = text; if (cls) item.className = cls; return item;};
function message(text, error = false) {$('message').textContent = text; $('message').className = error ? 'error' : '';}
async function api(path, options) {const response = await fetch(path, options); const data = await response.json(); if (!response.ok) throw new Error(data.error || 'Request failed'); return data;}
function fact(title, value) {const box = node('div'); box.append(node('span', title), node('strong', value)); return box;}
async function loadRecords() {
  const version = ++listVersion;
  const query = new URLSearchParams({q: $('search').value, status: $('status').value, queue: $('queue').checked ? '1' : '0'});
  try {
    const data = await api('/api/records?' + query);
    if (version !== listVersion) return;
    token = data.token;
    $('total').textContent = data.summary.trials; $('queueCount').textContent = data.summary.queue;
    $('reviewed').textContent = data.summary.reviewed; $('excluded').textContent = data.summary.excluded;
    $('context').textContent = `${data.summary.batches} recorded batches · Model ${data.model} · Separate workbench memory`;
    $('resultCount').textContent = `${data.records.length} results`;
    $('rows').replaceChildren();
    for (const record of data.records) {
      const tr = node('tr');
      if (selected && selected.batch === record.batch && selected.trial === record.trial) tr.className = 'selected';
      const id = node('td'), button = node('button', `${record.batch} / ${record.trial}`, 'record-button');
      button.addEventListener('click', () => openRecord(record.batch, record.trial)); id.append(button);
      const status = node('td'); status.append(node('span', record.status, `badge ${record.status}`));
      tr.append(id, node('td', record.pred === 0 ? 'Left' : 'Right'), node('td', record.prob.toFixed(2)), status);
      $('rows').append(tr);
    }
    if (!data.records.length) {const td = node('td', 'No matching records. Change your filters.'); td.colSpan = 4; const tr = node('tr'); tr.append(td); $('rows').append(tr);}
  } catch (error) {message(error.message, true);}
}
async function openRecord(batch, trial, policy = 'predicted') {
  const version = ++detailVersion;
  try {
    const data = await api('/api/detail?' + new URLSearchParams({batch, trial, policy}));
    if (version !== detailVersion) return;
    selected = data.record;
    const r = data.record, panel = $('detail'); panel.replaceChildren();
    const heading = node('div', '', 'panel-heading'); heading.append(node('h2', `${batch} / Trial ${trial}`), node('span', r.status, `badge ${r.status}`));
    const body = node('div', '', 'detail-body'), facts = node('div', '', 'facts');
    facts.append(fact('Original model prediction', label(r.pred)), fact('Model probability', r.prob.toFixed(3)),
      fact('Run-standardized C3 / C4', `${r.c3.toFixed(3)} / ${r.c4.toFixed(3)}`), fact('Original reference result', r.verdict));
    body.append(facts, node('p', r.status === 'reviewed' ? `Current human label: ${label(r.effective_label)}. The original prediction remains unchanged.` : r.status === 'excluded' ? 'Excluded from subsequent memory references. Original evidence is retained.' : 'No active human label. The model prediction is not a verified observation.', 'small'));
    body.append(node('h3', 'Earlier similar records'));
    const policyLabel = node('label', 'Reference source'), policySelect = node('select');
    for (const [value, text] of [['predicted', 'Eligible predictions and reviewed records'], ['reviewed', 'Human-reviewed records only']]) {const option = node('option', text); option.value = value; policySelect.append(option);}
    policySelect.value = policy; policySelect.addEventListener('change', () => openRecord(batch, trial, policySelect.value)); policyLabel.append(policySelect); body.append(policyLabel);
    body.append(node('p', data.retrieval_note, 'small'));
    for (const n of data.neighbors) {const box = node('div', '', 'neighbor'), button = node('button', `${n.batch} / Trial ${n.trial}`); button.addEventListener('click', () => openRecord(n.batch, n.trial, policy)); box.append(button, node('div', `${label(n.pred)} · distance ${n.distance.toFixed(3)} · ${n.status}`)); body.append(box);}
    if (!data.neighbors.length) body.append(node('p', 'No earlier records match this reference policy.', 'small'));
    body.append(node('h3', 'Record a review'));
    const form = node('form', '', 'review-form'); form.addEventListener('submit', e => e.preventDefault());
    const reviewerLabel = node('label', 'Reviewer'), reviewer = node('input'); reviewer.required = true; reviewer.maxLength = 100; reviewer.value = localStorage.getItem('eeg-reviewer') || ''; reviewerLabel.append(reviewer);
    const reasonLabel = node('label', 'Evidence or reason'), reason = node('textarea'); reason.required = true; reason.maxLength = 2000; reason.placeholder = 'What supports your review? Do not infer a label from model confidence alone.'; reasonLabel.append(reason);
    const actions = node('div', '', 'actions');
    for (const [text, action, reviewLabel, cls] of [['Confirm prediction', 'label', r.pred, 'primary'], ['Label left', 'label', 0, ''], ['Label right', 'label', 1, ''], ['Exclude', 'exclude', null, 'exclude'], ['Reopen', 'reopen', null, '']]) {
      const button = node('button', text, cls); button.type = 'button';
      button.addEventListener('click', async () => {
        if (!form.reportValidity()) return;
        const buttons = actions.querySelectorAll('button'); buttons.forEach(b => b.disabled = true);
        try {
          const result = await api('/api/review', {method: 'POST', headers: {'Content-Type': 'application/json', 'X-Review-Token': token}, body: JSON.stringify({batch, trial, action, label: reviewLabel, reviewer: reviewer.value, reason: reason.value, expected_version: r.review_id})});
          localStorage.setItem('eeg-reviewer', reviewer.value);
          message(`Review event ${result.review_event} saved. Original prediction and report evidence are preserved.`);
          await loadRecords(); await openRecord(batch, trial, policy);
        } catch (error) {message(error.message, true); buttons.forEach(b => b.disabled = false);}
      }); actions.append(button);
    }
    form.append(reviewerLabel, reasonLabel, actions); body.append(form, node('h3', 'Review history'));
    for (const event of data.reviews) {const box = node('div', '', 'event'); box.append(node('strong', `#${event.id} · ${event.reviewer} · ${event.action}${event.reviewed_label === null ? '' : ' / ' + label(event.reviewed_label)}`), node('div', event.reason), node('div', event.created, 'small')); body.append(box);}
    if (!data.reviews.length) body.append(node('p', 'No review events yet.', 'small'));
    panel.append(heading, body);
    await loadRecords();
  } catch (error) {message(error.message, true);}
}
$('filters').addEventListener('submit', e => {e.preventDefault(); loadRecords();});
loadRecords();
