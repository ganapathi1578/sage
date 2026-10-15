'use strict';

const $ = (selector) => document.querySelector(selector);
const state = {
  csrf: null,
  username: null,
  datasets: [],
  models: [],
  selectedVideoId: null,
  selectedExample: null,
  selectedUpload: null,
  sourceMode: 'dataset',
  videoOffset: 0,
  videoTotal: 0,
  questionOffset: 0,
  questionTotal: 0,
  currentIndexKey: '',
  indexPollToken: 0,
  videoSearchTimer: null,
  originalPrompt: null,
  loadingInference: false,
};

function toast(message, type = 'success') {
  const el = document.createElement('div');
  el.className = `toast ${type === 'error' ? 'error' : ''}`;
  el.textContent = message;
  $('#toast-stack').appendChild(el);
  setTimeout(() => el.remove(), 5000);
}

async function api(url, options = {}) {
  const headers = new Headers(options.headers || {});
  if (options.body && !(options.body instanceof FormData) && !headers.has('Content-Type')) {
    headers.set('Content-Type', 'application/json');
  }
  if (options.method && options.method.toUpperCase() !== 'GET' && state.csrf) {
    headers.set('X-CSRF-Token', state.csrf);
  }
  const response = await fetch(url, { credentials: 'same-origin', ...options, headers });
  const contentType = response.headers.get('content-type') || '';
  const body = contentType.includes('application/json') ? await response.json() : await response.text();
  if (response.status === 401 && url !== '/api/login') {
    showLogin();
    throw new Error('Your session expired. Sign in again.');
  }
  if (!response.ok && response.status !== 202) {
    const detail = body && typeof body === 'object' ? body.detail : body;
    throw new Error(detail || `Request failed (${response.status})`);
  }
  return { body, status: response.status };
}

function setBusy(button, busy, busyLabel = 'Working…') {
  if (!button.dataset.originalText) button.dataset.originalText = button.textContent.trim();
  button.disabled = busy;
  button.textContent = busy ? busyLabel : button.dataset.originalText;
}

function showLogin() {
  $('#app-view').hidden = true;
  $('#login-view').hidden = false;
  state.csrf = null;
  state.username = null;
}

function showApp(username, csrf) {
  state.username = username;
  state.csrf = csrf;
  $('#login-view').hidden = true;
  $('#app-view').hidden = false;
  $('#user-label').textContent = username;
  $('#user-avatar').textContent = (username[0] || 'R').toUpperCase();
}

$('#login-form').addEventListener('submit', async (event) => {
  event.preventDefault();
  const error = $('#login-error');
  error.hidden = true;
  const button = event.currentTarget.querySelector('button[type="submit"]');
  setBusy(button, true, 'Signing in…');
  try {
    const { body } = await api('/api/login', {
      method: 'POST',
      body: JSON.stringify({ username: $('#login-username').value.trim(), password: $('#login-password').value }),
    });
    $('#login-password').value = '';
    showApp(body.username, body.csrf_token);
    await initializeWorkspace();
  } catch (err) {
    error.textContent = err.message;
    error.hidden = false;
  } finally {
    setBusy(button, false);
  }
});

$('#logout-button').addEventListener('click', async () => {
  try { await api('/api/logout', { method: 'POST', body: '{}' }); } catch (_) { /* still clear local view */ }
  showLogin();
});

async function initializeApp() {
  try {
    const { body } = await api('/api/session');
    if (body.authenticated) {
      showApp(body.username, body.csrf_token);
      await initializeWorkspace();
    } else {
      showLogin();
      if (!body.auth_configured) {
        $('#login-error').textContent = 'The server administrator has not configured the authentication credentials yet.';
        $('#login-error').hidden = false;
      }
    }
  } catch (err) {
    showLogin();
    $('#login-error').textContent = err.message;
    $('#login-error').hidden = false;
  }
}

async function initializeWorkspace() {
  $('#header-status').textContent = 'Authenticated workspace';
  await Promise.all([loadModels(), loadDatasets(), loadRuntimeStatus()]);
  startIndexWatch();
}

async function loadRuntimeStatus() {
  try {
    const { body } = await api('/api/status');
    $('#device-label').textContent = body.gpu_name ? `${body.gpu_name}` : body.device || 'CPU';
    const missing = [];
    if (!body.paths?.checkpoint_root_ready) missing.push('No experiment checkpoints found.');
    if (!body.paths?.video_feature_store_ready) missing.push('Video feature store is not configured.');
    if (!body.paths?.text_store_root_ready) missing.push('Text token store root is not configured.');
    if (missing.length) {
      $('#setup-warning').textContent = missing.join(' ');
      $('#setup-warning').hidden = false;
    } else {
      $('#setup-warning').hidden = true;
    }
  } catch (err) {
    $('#device-label').textContent = 'Unavailable';
    toast(err.message, 'error');
  }
}

async function loadModels() {
  try {
    const { body } = await api('/api/models');
    state.models = body.runs || [];
    const select = $('#model-run');
    const prior = select.value;
    select.replaceChildren();
    if (!state.models.length) {
      select.add(new Option('No checkpoints found', ''));
      $('#model-meta').innerHTML = '<div class="empty-mini">No best/last checkpoints were found under outputs/experiments.</div>';
      $('#load-model-button').disabled = true;
      return;
    }
    state.models.forEach((run) => {
      const label = `${run.run} · ${run.architecture || 'model'}`;
      select.add(new Option(label, run.run));
    });
    if (state.models.some((m) => m.run === prior)) select.value = prior;
    $('#load-model-button').disabled = false;
    updateCheckpointChoices();
    select.onchange = updateCheckpointChoices;
    $('#model-checkpoint').onchange = updateModelMeta;
    updateModelMeta();
  } catch (err) {
    toast(`Could not list checkpoints: ${err.message}`, 'error');
  }
}

function selectedRun() {
  return state.models.find((run) => run.run === $('#model-run').value) || null;
}

function updateCheckpointChoices() {
  const run = selectedRun();
  const select = $('#model-checkpoint');
  const prior = select.value;
  select.replaceChildren();
  (run?.checkpoints || []).forEach((checkpoint) => {
    select.add(new Option(checkpoint.name === 'best_checkpoint.pt' ? 'Best checkpoint' : 'Last checkpoint', checkpoint.name));
  });
  if (!run?.checkpoints?.length) select.add(new Option('No checkpoints', ''));
  if ((run?.checkpoints || []).some((c) => c.name === prior)) select.value = prior;
  else if (run?.checkpoints?.some((c) => c.name === 'best_checkpoint.pt')) select.value = 'best_checkpoint.pt';
  updateModelMeta();
}

function updateModelMeta() {
  const run = selectedRun();
  if (!run) return;
  const checkpoint = (run.checkpoints || []).find((c) => c.name === $('#model-checkpoint').value);
  const rows = [
    ['Architecture', run.architecture || 'Unknown'],
    ['Training scale', run.dataset_scale || '—'],
    ['Parameters', run.parameters ? Number(run.parameters).toLocaleString() : '—'],
    ['Latest epoch', run.latest_epoch ?? '—'],
    ['Val accuracy', run.latest_validation_accuracy == null ? '—' : `${(100 * run.latest_validation_accuracy).toFixed(2)}%`],
    ['Checkpoint', checkpoint ? `${(checkpoint.size_bytes / (1024 * 1024)).toFixed(1)} MiB` : 'Missing'],
  ];
  $('#model-meta').replaceChildren();
  rows.forEach(([label, value]) => {
    const row = document.createElement('div'); row.className = 'meta-pair';
    const span = document.createElement('span'); span.textContent = label;
    const strong = document.createElement('strong'); strong.textContent = String(value);
    row.append(span, strong); $('#model-meta').appendChild(row);
  });
}

$('#load-model-button').addEventListener('click', async () => {
  const run = $('#model-run').value;
  const checkpoint = $('#model-checkpoint').value;
  if (!run || !checkpoint) return toast('Choose an experiment and checkpoint first.', 'error');
  const button = $('#load-model-button'); setBusy(button, true, 'Loading weights…');
  $('#model-dot').style.background = 'var(--yellow)';
  try {
    const { body } = await api('/api/model/load', { method: 'POST', body: JSON.stringify({ run, checkpoint }) });
    updateActiveModelMeta(body);
    $('#model-dot').style.background = 'var(--accent)';
    toast(`Loaded ${run} · ${checkpoint}`);
  } catch (err) {
    $('#model-dot').style.background = 'var(--red)';
    toast(err.message, 'error');
  } finally { setBusy(button, false); }
});

function updateActiveModelMeta(info) {
  if (!info?.loaded) return;
  $('#model-meta').replaceChildren();
  const rows = [
    ['Architecture', info.architecture], ['Training scale', info.dataset_scale],
    ['Checkpoint epoch', info.epoch], ['Global step', info.global_step],
    ['Best val score', info.best_validation_score == null ? '—' : Number(info.best_validation_score).toFixed(4)],
    ['Device / AMP', `${info.device} / ${info.amp}`],
  ];
  rows.forEach(([label, value]) => {
    const row = document.createElement('div'); row.className = 'meta-pair';
    const span = document.createElement('span'); span.textContent = label;
    const strong = document.createElement('strong'); strong.textContent = String(value ?? '—');
    row.append(span, strong); $('#model-meta').appendChild(row);
  });
}

async function loadDatasets() {
  const { body } = await api('/api/datasets');
  state.datasets = body.datasets || [];
  const select = $('#dataset-scale');
  select.replaceChildren();
  state.datasets.forEach((dataset) => {
    const count = dataset.splits?.train;
    const suffix = count ? ` · ${Number(count).toLocaleString()} train` : '';
    select.add(new Option(`${dataset.scale}${suffix}`, dataset.scale));
  });
  if (!state.datasets.length) {
    select.add(new Option('No compact datasets found', ''));
    $('#index-status').className = 'index-status failed';
    $('#index-status').textContent = 'No compact dataset directories found under UniProp/data/analysis.';
  }
  select.onchange = () => { resetDatasetSelection(); startIndexWatch(); };
  $('#dataset-split').onchange = () => { resetDatasetSelection(); startIndexWatch(); };
}

function resetDatasetSelection() {
  state.selectedVideoId = null;
  state.selectedExample = null;
  state.videoOffset = 0; state.questionOffset = 0;
  state.questionTotal = 0; state.videoTotal = 0;
  $('#video-select').replaceChildren(new Option('Indexing split…', ''));
  $('#video-count').textContent = 'Preparing dataset index';
  $('#question-list').replaceChildren(emptyState('Preparing dataset', 'The selected split is being indexed for fast filtering.'));
  clearQuestionEditor();
  showDatasetVideo(null);
  $('#more-videos').hidden = true; $('#more-questions').hidden = true;
}

function selectedScale() { return $('#dataset-scale').value; }
function selectedSplit() { return $('#dataset-split').value; }

function setIndexStatus(status) {
  const el = $('#index-status');
  el.className = `index-status ${status.status || ''}`;
  let text = 'Dataset index not started';
  if (status.status === 'building') {
    const pct = Math.round((status.progress || 0) * 100);
    text = `Indexing ${Number(status.processed_rows || 0).toLocaleString()} / ${Number(status.total_rows || 0).toLocaleString()} rows (${pct}%)`;
    if (status.current_file) text += ` · ${status.current_file}`;
  } else if (status.status === 'complete') {
    text = `${Number(status.processed_rows || 0).toLocaleString()} examples indexed`;
  } else if (status.status === 'failed') {
    text = `Index failed: ${status.error || 'Unknown error'}`;
  }
  el.replaceChildren();
  const dot = document.createElement('span'); dot.className = 'status-led';
  const label = document.createElement('span'); label.textContent = text;
  el.append(dot, label);
}

async function startIndexWatch() {
  const scale = selectedScale(); const split = selectedSplit();
  if (!scale || !split) return;
  const token = ++state.indexPollToken;
  state.currentIndexKey = `${scale}/${split}`;
  setIndexStatus({ status: 'building', progress: 0, processed_rows: 0, total_rows: 0 });
  const poll = async () => {
    if (token !== state.indexPollToken) return;
    try {
      const { body } = await api(`/api/index-status?scale=${encodeURIComponent(scale)}&split=${encodeURIComponent(split)}`);
      if (token !== state.indexPollToken) return;
      setIndexStatus(body);
      if (body.status === 'building') {
        setTimeout(poll, 2500);
      } else if (body.status === 'complete') {
        await Promise.all([loadVideos(true), loadFilters()]);
        if (!state.selectedVideoId) await loadQuestions();
      } else if (body.status === 'failed') {
        toast(body.error || 'Dataset index failed', 'error');
      }
    } catch (err) {
      setIndexStatus({ status: 'failed', error: err.message });
      toast(err.message, 'error');
    }
  };
  await poll();
}

async function loadVideos(reset = false) {
  const scale = selectedScale(), split = selectedSplit();
  if (reset) {
    state.videoOffset = 0;
    $('#video-select').replaceChildren();
  }
  const search = $('#video-search').value.trim();
  try {
    const { body, status } = await api(`/api/videos?scale=${encodeURIComponent(scale)}&split=${split}&search=${encodeURIComponent(search)}&limit=250&offset=${state.videoOffset}`);
    if (status === 202 || body.status === 'building') {
      setIndexStatus(body); return;
    }
    state.videoTotal = body.total || 0;
    const select = $('#video-select');
    if (reset) select.replaceChildren();
    (body.videos || []).forEach((video) => {
      const label = `${video.video_id} · ${Number(video.num_examples).toLocaleString()} Q`;
      if (![...select.options].some((o) => o.value === video.video_id)) select.add(new Option(label, video.video_id));
    });
    if (reset && !state.selectedVideoId && select.options.length) {
      select.selectedIndex = 0;
      state.selectedVideoId = select.value;
      await onVideoSelected(true);
    } else if (reset && state.selectedVideoId && [...select.options].some((o) => o.value === state.selectedVideoId)) {
      select.value = state.selectedVideoId;
    } else if (reset && select.options.length) {
      select.selectedIndex = 0; state.selectedVideoId = select.value; await onVideoSelected(true);
    }
    $('#video-count').textContent = `${Number(state.videoTotal).toLocaleString()} videos in this split`;
    $('#more-videos').hidden = state.videoOffset + (body.videos || []).length >= state.videoTotal;
    $('#more-videos').textContent = 'Load more ↗';
  } catch (err) { toast(err.message, 'error'); }
}

$('#more-videos').addEventListener('click', async () => {
  state.videoOffset += 250;
  await loadVideos(false);
});
$('#video-select').addEventListener('change', async () => {
  state.selectedVideoId = $('#video-select').value || null;
  await onVideoSelected(false);
});
$('#video-search').addEventListener('input', () => {
  clearTimeout(state.videoSearchTimer);
  state.videoSearchTimer = setTimeout(() => loadVideos(true), 300);
});

async function onVideoSelected(fromIndexLoad) {
  const videoId = state.selectedVideoId;
  state.selectedExample = null;
  state.questionOffset = 0; state.questionTotal = 0;
  clearQuestionEditor();
  if (state.sourceMode === 'dataset') showDatasetVideo(videoId);
  else updateSelectedVideoFooter();
  await Promise.all([loadFilters(), loadQuestions()]);
}

function showDatasetVideo(videoId) {
  $('#video-player').pause?.();
  $('#video-player').removeAttribute('src');
  $('#video-player').load();
  if (!videoId) {
    $('#video-player').hidden = true; $('#video-empty').hidden = false;
    $('#video-label').textContent = 'NO CLIP SELECTED';
    $('#selected-video-id').textContent = '—';
    $('#feature-status').textContent = 'Cached inference';
    return;
  }
  $('#video-empty').hidden = true; $('#video-player').hidden = false;
  $('#video-player').src = `/api/video?video_id=${encodeURIComponent(videoId)}`;
  $('#video-player').load();
  $('#video-label').textContent = 'DATASET VIDEO';
  $('#selected-video-id').textContent = videoId;
  $('#feature-status').textContent = 'Cached features';
  $('#video-player').onerror = () => {
    $('#video-player').hidden = true;
    $('#video-empty').hidden = false;
    $('#video-empty').querySelector('strong').textContent = 'Original video not found';
    $('#video-empty').querySelector('span').textContent = 'Cached model features may still be available for inference.';
  };
}

async function loadFilters() {
  const scale = selectedScale(), split = selectedSplit();
  try {
    const q = new URLSearchParams({ scale, split });
    if (state.selectedVideoId) q.set('video_id', state.selectedVideoId);
    const { body, status } = await api(`/api/filters?${q}`);
    if (status === 202) return;
    populateFilter($('#task-filter'), body.task_types || [], 'All formats');
    populateFilter($('#reasoning-filter'), body.reasoning_families || [], 'All families');
    populateFilter($('#generator-filter'), body.generator_families || [], 'All generators');
  } catch (err) { if (!/still being built/i.test(err.message)) console.warn(err); }
}

function prettyValue(value) {
  const labels = {
    binary: 'Yes / No (binary)',
    three_way: 'Three-way choice',
    single_choice: 'Single choice',
    multi_label: 'Select all that apply',
    action_temporal: 'Action / temporal',
    object_existence: 'Object existence',
    relation_sets: 'Relation sets',
    relation_verification: 'Relation verification',
  };
  return labels[value] || String(value).replaceAll('_', ' ');
}

function populateFilter(select, values, allLabel) {
  const prior = select.value || 'all';
  select.replaceChildren(new Option(allLabel, 'all'));
  values.forEach((value) => select.add(new Option(prettyValue(value), value)));
  if ([...select.options].some((o) => o.value === prior)) select.value = prior;
}
$('#task-filter').addEventListener('change', () => { state.questionOffset = 0; loadQuestions(); });
$('#reasoning-filter').addEventListener('change', () => { state.questionOffset = 0; loadQuestions(); });
$('#generator-filter').addEventListener('change', () => { state.questionOffset = 0; loadQuestions(); });
$('#more-questions').addEventListener('click', async () => { state.questionOffset += 10; await loadQuestions(true); });

async function loadQuestions(append = false) {
  const scale = selectedScale(), split = selectedSplit();
  const q = new URLSearchParams({ scale, split, limit: '10', offset: String(state.questionOffset) });
  if (state.selectedVideoId) q.set('video_id', state.selectedVideoId);
  if ($('#task-filter').value && $('#task-filter').value !== 'all') q.set('task_type', $('#task-filter').value);
  if ($('#reasoning-filter').value && $('#reasoning-filter').value !== 'all') q.set('reasoning_family', $('#reasoning-filter').value);
  if ($('#generator-filter').value && $('#generator-filter').value !== 'all') q.set('generator_family', $('#generator-filter').value);
  try {
    const { body, status } = await api(`/api/questions?${q}`);
    if (status === 202) {
      $('#question-list').replaceChildren(emptyState('Indexing dataset', `${Number(body.processed_rows || 0).toLocaleString()} of ${Number(body.total_rows || 0).toLocaleString()} rows indexed.`));
      return;
    }
    state.questionTotal = body.total || 0;
    $('#question-count').textContent = `${Number(state.questionTotal).toLocaleString()} matching questions`;
    if (!append) $('#question-list').replaceChildren();
    if (!body.examples?.length && !append) {
      $('#question-list').appendChild(emptyState(state.selectedVideoId ? 'No matching questions' : 'Select a video first', state.selectedVideoId ? 'Try another question format or reasoning family.' : 'Choose a dataset video to find the questions associated with it.'));
    }
    for (const example of body.examples || []) $('#question-list').appendChild(renderQuestionCard(example));
    $('#question-page-info').textContent = state.questionTotal ? `Showing ${Math.min(state.questionOffset + (body.examples || []).length, state.questionTotal)} of ${state.questionTotal}` : 'No examples yet';
    $('#more-questions').hidden = state.questionOffset + (body.examples || []).length >= state.questionTotal;
  } catch (err) { toast(err.message, 'error'); }
}

function emptyState(title, description) {
  const el = document.createElement('div'); el.className = 'empty-state';
  const icon = document.createElement('div'); icon.className = 'empty-icon'; icon.textContent = '⌕';
  const strong = document.createElement('strong'); strong.textContent = title;
  const p = document.createElement('p'); p.textContent = description;
  el.append(icon, strong, p); return el;
}

function renderQuestionCard(example) {
  const button = document.createElement('button'); button.className = 'question-item'; button.type = 'button';
  button.dataset.exampleId = String(example.example_id);
  if (state.selectedExample?.example_id === example.example_id) button.classList.add('selected');
  const top = document.createElement('div'); top.className = 'question-item-top';
  const id = document.createElement('span'); id.className = 'question-id'; id.textContent = `EXAMPLE #${example.example_id}`;
  const tags = document.createElement('div'); tags.className = 'tag-row';
  for (const value of [example.task_type, example.reasoning_family].filter(Boolean)) {
    const tag = document.createElement('span'); tag.className = 'mini-tag'; tag.textContent = prettyValue(value); tags.appendChild(tag);
  }
  top.append(id, tags);
  const p = document.createElement('p'); p.textContent = example.question;
  const preview = document.createElement('div'); preview.className = 'question-options-preview';
  preview.textContent = (example.options || []).map((option, i) => `${String.fromCharCode(65+i)}. ${option}`).join('   ·   ');
  button.append(top, p, preview);
  button.addEventListener('click', () => selectExample(example));
  return button;
}

function selectExample(example) {
  state.selectedExample = example;
  if (state.sourceMode === 'dataset' && state.selectedVideoId !== example.video_id) {
    state.selectedVideoId = example.video_id;
    $('#video-select').value = example.video_id;
    showDatasetVideo(example.video_id);
  }
  state.originalPrompt = { question: example.question, options: [...example.options] };
  $('#question-editor').value = example.question;
  $('#options-editor').value = example.options.join('\n');
  $('#answer-mode').value = example.target_mode || 'single_choice';
  $('#prompt-source').textContent = `${example.target_mode === 'multi_label' ? 'Multi-label' : 'Dataset example'} · unmodified`;
  $('#expected-answer').value = '';
  updateEditorCounts(); updateExpectedChoices();
  [...$('#question-list').querySelectorAll('.question-item')].forEach((el) => el.classList.toggle('selected', el.dataset.exampleId === String(example.example_id)));
  $('#feature-status').textContent = state.sourceMode === 'upload' ? 'Upload · on-demand features' : 'Cached features';
}

function clearQuestionEditor() {
  state.originalPrompt = null;
  $('#question-editor').value = '';
  $('#options-editor').value = '';
  $('#answer-mode').value = 'single_choice';
  $('#prompt-source').textContent = 'Custom prompt';
  updateEditorCounts(); updateExpectedChoices();
}

function currentOptions() {
  return $('#options-editor').value.split(/\r?\n/).map((s) => s.trim()).filter(Boolean);
}

function updateEditorCounts() {
  $('#question-length').textContent = `${$('#question-editor').value.length} / 2000`;
  $('#options-count').textContent = `${currentOptions().length} options`;
  if (state.selectedExample) {
    const changed = $('#question-editor').value.trim() !== state.selectedExample.question.trim()
      || JSON.stringify(currentOptions()) !== JSON.stringify(state.selectedExample.options.map((s) => s.trim()));
    $('#prompt-source').textContent = changed ? 'Modified dataset prompt' : `${state.selectedExample.target_mode === 'multi_label' ? 'Multi-label' : 'Dataset example'} · unmodified`;
  }
  updateExpectedChoices();
}
$('#question-editor').addEventListener('input', updateEditorCounts);
$('#options-editor').addEventListener('input', updateEditorCounts);
$('#answer-mode').addEventListener('change', updateExpectedChoices);
$('#restore-options').addEventListener('click', () => {
  if (!state.originalPrompt) return toast('Select a dataset question to restore its original text and options.', 'error');
  $('#options-editor').value = state.originalPrompt.options.join('\n');
  updateEditorCounts();
});
$('#reset-question').addEventListener('click', () => {
  if (state.selectedExample) {
    $('#question-editor').value = state.selectedExample.question;
    $('#options-editor').value = state.selectedExample.options.join('\n');
    $('#answer-mode').value = state.selectedExample.target_mode || 'single_choice';
    $('#expected-answer').value = '';
    updateEditorCounts();
    return;
  }
  clearQuestionEditor();
});

function updateExpectedChoices() {
  const select = $('#expected-answer');
  const label = $('#expected-answer-label');
  const options = currentOptions();
  const multi = $('#answer-mode').value === 'multi_label';
  const prior = new Set([...select.selectedOptions].map((o) => o.value));
  select.multiple = multi;
  select.size = multi ? Math.min(4, Math.max(2, options.length)) : 1;
  label.textContent = multi ? 'Expected answers (optional)' : 'Expected answer (optional)';
  select.replaceChildren();
  if (!multi) select.add(new Option('Not specified', ''));
  options.forEach((text, i) => {
    const item = new Option(`${String.fromCharCode(65+i)} · ${text}`, String(i));
    if (prior.has(String(i))) item.selected = true;
    select.add(item);
  });
  if (!multi && [...select.options].some((o) => o.value === [...prior][0])) select.value = [...prior][0];
  select.title = multi ? 'Select one or more known positive options.' : 'Optional ground truth for a custom or modified question.';
}

$('#dataset-source-tab').addEventListener('click', () => switchSource('dataset'));
$('#upload-source-tab').addEventListener('click', () => switchSource('upload'));
function switchSource(mode) {
  state.sourceMode = mode;
  const datasetMode = mode === 'dataset';
  $('#dataset-source-tab').classList.toggle('active', datasetMode);
  $('#upload-source-tab').classList.toggle('active', !datasetMode);
  $('#dataset-video-source').hidden = !datasetMode;
  $('#upload-video-source').hidden = datasetMode;
  if (datasetMode) {
    state.selectedUpload = null;
    if (state.selectedVideoId) showDatasetVideo(state.selectedVideoId);
    else showDatasetVideo(null);
    $('#feature-status').textContent = 'Cached features';
  } else if (state.selectedUpload) {
    showUploadedVideo(state.selectedUpload);
  } else {
    showDatasetVideo(null);
    $('#video-label').textContent = 'UPLOAD A VIDEO';
    $('#selected-video-id').textContent = 'Custom upload';
    $('#feature-status').textContent = 'On-demand features';
  }
}

$('#upload-drop').addEventListener('click', (event) => {
  if (event.target.id !== 'upload-input') $('#upload-input').click();
});
$('#upload-drop').addEventListener('dragover', (event) => { event.preventDefault(); $('#upload-drop').classList.add('dragover'); });
$('#upload-drop').addEventListener('dragleave', () => $('#upload-drop').classList.remove('dragover'));
$('#upload-drop').addEventListener('drop', (event) => {
  event.preventDefault(); $('#upload-drop').classList.remove('dragover');
  const file = event.dataTransfer.files?.[0]; if (file) uploadFile(file);
});
$('#upload-input').addEventListener('change', (event) => { const file = event.target.files?.[0]; if (file) uploadFile(file); event.target.value = ''; });

async function uploadFile(file) {
  switchSource('upload');
  $('#upload-meta').hidden = false; $('#upload-meta').textContent = `Uploading ${file.name}…`;
  const form = new FormData(); form.append('file', file, file.name);
  try {
    const { body } = await api('/api/uploads', { method: 'POST', body: form });
    state.selectedUpload = body;
    showUploadedVideo(body);
    $('#upload-meta').textContent = `${body.filename} · ${body.width}×${body.height} · ${body.duration_seconds}s · ${formatBytes(body.size_bytes)}. Features will be extracted on first inference.`;
    toast('Video uploaded. It will remain temporary and will not be added to the dataset.');
  } catch (err) {
    $('#upload-meta').textContent = err.message;
    toast(err.message, 'error');
  }
}

function showUploadedVideo(upload) {
  if (!upload) return;
  $('#video-empty').hidden = true; $('#video-player').hidden = false;
  $('#video-player').src = upload.preview_url;
  $('#video-player').load();
  $('#video-label').textContent = 'TEMPORARY UPLOAD';
  $('#selected-video-id').textContent = upload.filename;
  $('#feature-status').textContent = 'On-demand FastViTHD';
}
function formatBytes(bytes) {
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

$('#run-inference').addEventListener('click', async () => {
  if (state.loadingInference) return;
  const run = $('#model-run').value;
  const checkpoint = $('#model-checkpoint').value;
  const question = $('#question-editor').value.trim();
  const options = currentOptions();
  const validation = $('#validation-message'); validation.hidden = true;
  if (!run || !checkpoint) return showValidation('Choose an experiment and checkpoint first.');
  if (state.sourceMode === 'dataset' && !state.selectedVideoId) return showValidation('Select a dataset video first.');
  if (state.sourceMode === 'upload' && !state.selectedUpload) return showValidation('Upload a video first.');
  if (!question) return showValidation('Enter a question.');
  if (options.length < 2 || options.length > 16) return showValidation('Provide 2–16 non-empty answer options, one per line.');
  if ($('#options-editor').value.split(/\r?\n/).some((line) => !line.trim())) return showValidation('Remove empty lines between answer options.');
  if (new Set(options.map((s) => s.toLocaleLowerCase())).size !== options.length) return showValidation('Answer options must be unique.');
  state.loadingInference = true;
  const button = $('#run-inference'); setBusy(button, true, 'Running model inference…');
  $('#results-card').hidden = false;
  $('#result-subtitle').textContent = 'Preparing features and scoring candidates…';
  $('#score-list').replaceChildren();
  try {
    const expected = [...$('#expected-answer').selectedOptions]
      .map((option) => option.value)
      .filter((value) => value !== '')
      .map(Number);
    const body = {
      run, checkpoint, scale: selectedScale(), split: selectedSplit(),
      video_id: state.sourceMode === 'dataset' ? state.selectedVideoId : null,
      upload_id: state.sourceMode === 'upload' ? state.selectedUpload.upload_id : null,
      example_id: state.selectedExample?.example_id ?? null,
      question_sentence_id: state.selectedExample?.query_sentence_id ?? null,
      option_sentence_ids: state.selectedExample?.option_sentence_ids ?? null,
      question_text: question, options, expected_indices: expected,
      task_mode: $('#answer-mode').value,
    };
    const { body: result } = await api('/api/inference', { method: 'POST', body: JSON.stringify(body) });
    renderResults(result);
    $('#results-card').scrollIntoView({ behavior: 'smooth', block: 'start' });
  } catch (err) {
    $('#result-subtitle').textContent = 'Inference failed';
    showValidation(err.message);
    toast(err.message, 'error');
  } finally {
    state.loadingInference = false;
    setBusy(button, false);
  }
});

function showValidation(message) {
  const el = $('#validation-message'); el.textContent = message; el.hidden = false;
}

function renderResults(result) {
  $('#result-subtitle').textContent = `${result.run} · ${result.checkpoint} · epoch ${result.checkpoint_epoch} · step ${Number(result.global_step).toLocaleString()}`;
  $('#predicted-answer').textContent = `${String.fromCharCode(65 + result.prediction_index)}. ${result.prediction_text}`;
  $('#result-model').textContent = result.architecture || result.run;
  $('#result-checkpoint').textContent = `${result.checkpoint} · epoch ${result.checkpoint_epoch}`;
  $('#prediction-context').textContent = result.uploaded_video
    ? 'Custom video encoded with the compatible FastViTHD feature pipeline.'
    : 'Using cached FastViTHD video features. Scores are relative, not calibrated confidence.';
  const correctness = $('#result-correctness');
  correctness.className = 'result-status neutral';
  if (result.top1_correct === true) { correctness.textContent = 'Top-1 correct'; correctness.classList.add(''); }
  else if (result.top1_correct === false) { correctness.textContent = 'Top-1 incorrect'; correctness.classList.add('bad'); }
  else correctness.textContent = 'Prediction only';

  const list = $('#score-list'); list.replaceChildren();
  const multiLabel = result.task_mode === 'multi_label';
  $('#score-label').textContent = multiLabel ? `SIGMOID SCORE · THRESHOLD ≥ ${Number(result.threshold).toFixed(2)}` : 'RELATIVE PROBABILITY';
  const maxProb = multiLabel ? 1 : Math.max(0.0001, ...result.options.map((o) => o.relative_probability));
  result.options.forEach((option) => {
    const thresholdSelected = multiLabel && option.selected_by_threshold;
    const row = document.createElement('div');
    row.className = `score-row ${option.is_top1 ? 'top' : ''} ${thresholdSelected ? 'threshold-selected' : ''} ${(result.ground_truth_indices || []).includes(option.index) ? 'expected' : ''}`;
    const title = document.createElement('div'); title.className = 'score-option';
    const letter = document.createElement('small'); letter.textContent = String.fromCharCode(65 + option.index);
    const text = document.createElement('span'); text.textContent = option.text;
    title.append(letter, text);
    if (thresholdSelected) {
      const picked = document.createElement('small'); picked.className = 'selected-answer-tag'; picked.textContent = 'selected'; title.appendChild(picked);
    }
    const track = document.createElement('div'); track.className = 'bar-track';
    const displayedScore = multiLabel ? option.sigmoid_probability : option.relative_probability;
    const fill = document.createElement('div'); fill.className = 'bar-fill'; fill.style.width = `${Math.min(100, displayedScore / maxProb * 100)}%`; track.appendChild(fill);
    const pct = document.createElement('div'); pct.className = 'score-pct'; pct.textContent = `${(displayedScore * 100).toFixed(1)}%`;
    const logit = document.createElement('div'); logit.className = 'score-logit'; logit.textContent = `z ${option.logit.toFixed(3)}`;
    row.append(title, track, pct, logit); list.appendChild(row);
  });
  const gt = $('#result-ground-truth');
  if (result.ground_truth_indices !== null && result.ground_truth_indices !== undefined) {
    gt.hidden = false;
    gt.className = `ground-truth ${result.top1_correct ? 'correct' : 'incorrect'}`;
    const prefix = result.top1_correct ? 'Correct top-1. ' : 'Incorrect top-1. ';
    const source = result.ground_truth_source ? ` (${result.ground_truth_source})` : '';
    const answers = (result.ground_truth_texts || []).map((text, i) => text).join(' · ') || '(empty target set)';
    gt.textContent = `${prefix}Expected: ${answers}${source}.`;
    if (result.exact_match !== null) gt.textContent += ` Exact set match: ${result.exact_match ? 'yes' : 'no'}.`;
  } else {
    gt.hidden = true; gt.textContent = '';
  }
  $('#result-model').textContent = `${result.run} · ${result.architecture}`;
}

initializeApp();
