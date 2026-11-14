// The ILotto dashboard page. Plain JavaScript, no build step, no libraries.
//
// The server owns everything that matters: the list of models and their
// settings (GET /api/state), all validation, and all numbers. This file draws
// what the server returns and sends back what the user chose.
//
// Two kinds of state are kept apart on purpose:
//   draft  - the settings in the controls, saved in the browser between visits;
//   shown  - the last completed prediction, with the settings that produced it.
// Changing a control never changes a result; it only marks the result as old.
"use strict";

const TOKEN = document.querySelector('meta[name="lotto-token"]').content;
const $ = (selector, root = document) => root.querySelector(selector);
const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];
const esc = value => String(value).replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

let STATE = null;          // GET /api/state
let draft = null;          // {model, params, seed, rows, date}
let shown = null;          // the prediction on screen
let comparison = null;     // the comparison on screen
let lastAttempt = null;    // id of the Lab attempt to show first

// ---------------------------------------------------------------- formatting
const percent = (p, digits = 2) => (100 * p).toFixed(p < 0.001 ? 5 : digits) + "%";
const oneIn = p => Math.round(1 / p).toLocaleString("en");
// Large counts shortened for a tile: 1,627,349 -> 1.63M.
const compact = n => n >= 1e6 ? (n / 1e6).toFixed(2) + "M" : n >= 1e4 ? Math.round(n / 1e3) + "k" : Math.round(n).toLocaleString("en");
const day = iso => new Date(iso + "T12:00:00Z").toLocaleDateString("en-GB",
  { weekday: "short", day: "numeric", month: "short", year: "numeric", timeZone: "UTC" });
const clock = iso => new Date(iso).toLocaleTimeString("en-GB", { hour: "2-digit", minute: "2-digit", timeZone: "Asia/Jerusalem" });
const balls = (row, hits = null) =>
  row.slice(0, 6).map(n => `<span class="chip${hits && hits.has(n) ? " hit" : ""}">${n}</span>`).join("") +
  `<span class="chip strong${hits && hits.strong === row[6] ? " hit" : ""}" title="strong number">${row[6]}</span>`;

// ----------------------------------------------------------------------- API
async function api(method, path, body) {
  const options = { method, headers: {} };
  if (method !== "GET") options.headers["X-Lotto-Token"] = TOKEN;
  if (body !== undefined) { options.headers["Content-Type"] = "application/json"; options.body = JSON.stringify(body); }
  let response;
  try { response = await fetch(path, options); }
  catch { throw { message: "The dashboard server is not answering. Is `uv run lotto ui` still running?", fields: {} }; }
  const data = await response.json().catch(() => ({}));
  if (!response.ok) throw data.error || { message: `Request failed (${response.status}).`, fields: {} };
  return data;
}

// Start a job and wait for it. `onProgress` gets each status; returns the result.
// The returned promise has a `cancel()` method.
function runJob(path, body, onProgress) {
  let jobId = null, stopped = false;
  const promise = (async () => {
    let job = await api("POST", path, body);
    jobId = job.id;
    while (["queued", "running", "cancel_requested"].includes(job.state)) {
      onProgress(job);
      await new Promise(resolve => setTimeout(resolve, document.hidden ? 1500 : 350));
      job = await api("GET", `/api/jobs/${jobId}`);
    }
    if (job.state === "succeeded") return job.result;
    if (job.state === "cancelled") throw { cancelled: true, message: "Cancelled.", fields: {} };
    throw job.error || { message: "The job failed.", fields: {} };
  })();
  promise.cancel = () => { if (jobId && !stopped) { stopped = true; api("DELETE", `/api/jobs/${jobId}`).catch(() => {}); } };
  return promise;
}

function progressHTML(job, label) {
  const p = job.progress, known = p.total > 0;
  return `<div class="card"><div class="progress"><span>${esc(job.state === "queued" ? "Waiting for the previous job" : p.stage || label)}</span>
    <div class="meter${known ? "" : " busy"}"><i style="width:${known ? Math.round(100 * p.done / p.total) : 35}%"></i></div>
    <button class="btn" type="button" data-cancel>Cancel</button></div></div>`;
}

// --------------------------------------------------------------------- theme
function setTheme(theme) {
  document.documentElement.dataset.theme = theme;
  $("#theme").textContent = theme === "dark" ? "Light theme" : "Dark theme";
  try { localStorage.setItem("lotto.theme", theme); } catch { /* private mode */ }
}
$("#theme").addEventListener("click", () => setTheme(document.documentElement.dataset.theme === "dark" ? "light" : "dark"));

// ---------------------------------------------------------------------- tabs
function showTab(name) {
  $$("#tabs button").forEach(b => b.setAttribute("aria-selected", String(b.dataset.tab === name)));
  $$(".screen").forEach(s => s.classList.toggle("on", s.id === name));
  if (name === "lab") renderLabConfig();
  if (name === "log") loadLog();
}
$("#tabs").addEventListener("click", e => e.target.dataset.tab && showTab(e.target.dataset.tab));
$("#tabs").addEventListener("keydown", e => {
  if (!["ArrowLeft", "ArrowRight"].includes(e.key)) return;
  const tabs = $$("#tabs button"), i = tabs.indexOf(document.activeElement);
  const next = tabs[(i + (e.key === "ArrowRight" ? 1 : tabs.length - 1)) % tabs.length];
  next.focus(); showTab(next.dataset.tab);
});

// ------------------------------------------------------------------- top bar
function renderTopBar() {
  $("#next").textContent = day(STATE.next_draw) + (STATE.next_draw === STATE.today ? " (today)" : "");
  $("#archive").textContent = `${STATE.archive.draws.toLocaleString("en")} draws through ${day(STATE.archive.last_date)}`;
  const r = STATE.refresh;
  $("#refreshstate").textContent = r.offline ? "Offline mode: not checking for new draws."
    : r.state === "not_checked" ? "Not checked for new draws in this session."
    : r.state === "updated" ? `New draws added at ${clock(r.at)}.`
    : r.state === "unchanged" ? `Checked at ${clock(r.at)}: no new draws.`
    : `Check failed at ${clock(r.at)}; using the saved archive.`;
  $("#refresh").disabled = r.offline;
}
$("#refresh").addEventListener("click", async () => {
  const button = $("#refresh");
  button.disabled = true; $("#refreshstate").textContent = "Checking pais.co.il…";
  try { await runJob("/api/refresh", {}, () => {}); } catch (error) { showBanner(error.message); }
  STATE = await api("GET", "/api/state");
  renderTopBar(); renderControls(); markStale();
});
function showBanner(message) {
  const banner = $("#banner");
  banner.hidden = !message; banner.textContent = message || "";
}

// ------------------------------------------------------------ Predict: controls
const modelSpec = id => STATE.models.find(m => m.id === id);
const defaults = spec => Object.fromEntries(spec.params.map(p => [p.name, structuredClone(p.default)]));
function usesSeed(spec, params) {
  if (spec.seed_default === null) return false;
  return !spec.seed_only_when || Object.entries(spec.seed_only_when).every(([k, v]) => params[k] === v);
}
function saveDraft() { try { localStorage.setItem("lotto.draft", JSON.stringify(draft)); } catch { /* private mode */ } }
function loadDraft() {
  try {
    const saved = JSON.parse(localStorage.getItem("lotto.draft"));
    if (saved && modelSpec(saved.model)) return saved;
  } catch { /* fall through */ }
  return null;
}
function freshDraft(modelId) {
  const spec = modelSpec(modelId);
  return { model: modelId, params: defaults(spec), seed: spec.seed_default, rows: draft ? draft.rows : 10, date: "" };
}
function applyPreset(preset) {
  draft = { ...freshDraft(preset.model), params: structuredClone(preset.params), seed: preset.seed ?? modelSpec(preset.model).seed_default };
  saveDraft(); renderControls(); markStale();
}

function renderControls() {
  const spec = modelSpec(draft.model), select = $("#model");
  if (!select.options.length) {
    for (const family of [...new Set(STATE.models.map(m => m.family))]) {
      const group = document.createElement("optgroup"); group.label = family;
      STATE.models.filter(m => m.family === family).forEach(m => {
        const option = new Option(m.label + (m.available ? "" : " (not installed)"), m.id);
        option.disabled = !m.available; group.append(option);
      });
      select.append(group);
    }
  }
  select.value = draft.model;
  $("#modeldesc").textContent = spec.description + (spec.available ? "" : " " + spec.unavailable_reason);

  const box = $("#params");
  box.innerHTML = spec.params.length ? "" : '<p class="hint">This model has no settings.</p>';
  for (const p of spec.params) {
    const id = `param-${p.name}`, value = draft.params[p.name];
    let control;
    if (p.kind === "choice") {
      control = `<select id="${id}" data-param="${p.name}">${p.choices.map(c => `<option${c === value ? " selected" : ""}>${esc(c)}</option>`).join("")}</select>`;
    } else if (p.kind === "subset") {
      control = p.choices.map(c => `<label class="check"><input type="checkbox" data-param="${p.name}" value="${c}"${value.includes(c) ? " checked" : ""}> ${esc(STATE.preset_names[c] || c)}</label>`).join("");
    } else {
      const step = p.kind === "float" ? "any" : "1";
      control = `<input id="${id}" data-param="${p.name}" type="number" min="${p.low}" max="${p.high}" step="${step}" value="${value ?? ""}"${p.kind === "int_or_all" ? ' placeholder="all draws"' : ""}>`;
    }
    const range = p.kind === "choice" || p.kind === "subset" ? "" : ` Allowed: ${p.low} to ${p.high.toLocaleString("en")}${p.kind === "int_or_all" ? ", or empty for all" : ""}.`;
    box.insertAdjacentHTML("beforeend", `<div><label for="${id}">${esc(p.label)}</label>${control}
      <p class="hint">${esc(p.help)}${range}</p><p class="fielderror" data-error="${p.name}"></p></div>`);
  }
  $("#seedbox").hidden = !usesSeed(spec, draft.params);
  $("#seed").value = draft.seed ?? "";
  $("#rows").innerHTML = STATE.rows.map(n => `<button type="button" aria-pressed="${n === draft.rows}" data-rows="${n}">${n}</button>`).join("");
  const date = $("#date");
  date.disabled = !spec.any_date;
  date.value = spec.any_date && draft.date ? draft.date : STATE.next_draw;
  $("#datehint").textContent = spec.any_date
    ? "Any date. A past date is a look back: the model sees only draws before it, and it cannot be saved as a prediction."
    : "The next draw. Only the date-conditioned and seed-world models take another date.";
  $("#generate").disabled = !spec.available;
}

function readParams() {
  const spec = modelSpec(draft.model), params = {};
  for (const p of spec.params) {
    if (p.kind === "subset") params[p.name] = $$(`[data-param="${p.name}"]:checked`).map(i => i.value);
    else if (p.kind === "choice") params[p.name] = $(`[data-param="${p.name}"]`).value;
    else {
      const raw = $(`[data-param="${p.name}"]`).value.trim();
      params[p.name] = raw === "" ? (p.kind === "int_or_all" ? null : NaN) : Number(raw);
    }
  }
  return params;
}
function controlsChanged() {
  const spec = modelSpec(draft.model);
  draft.params = readParams();
  draft.seed = $("#seed").value.trim();
  draft.date = spec.any_date && $("#date").value !== STATE.next_draw ? $("#date").value : "";
  $("#seedbox").hidden = !usesSeed(spec, draft.params);
  saveDraft(); markStale();
}
$("#model").addEventListener("change", e => { draft = freshDraft(e.target.value); saveDraft(); renderControls(); markStale(); });
$("#controls").addEventListener("input", e => { if (e.target.id !== "model") controlsChanged(); });
$("#rows").addEventListener("click", e => {
  if (!e.target.dataset.rows) return;
  draft.rows = Number(e.target.dataset.rows);
  $$("#rows button").forEach(b => b.setAttribute("aria-pressed", String(b === e.target)));
  saveDraft(); markStale();
});
$("#reset").addEventListener("click", () => { draft = freshDraft(draft.model); saveDraft(); renderControls(); markStale(); });
$("#dice").addEventListener("click", () => {
  const spec = modelSpec(draft.model);
  const big = crypto.getRandomValues(new BigUint64Array(1))[0];
  $("#seed").value = (big % (BigInt(spec.seed_max) + 1n)).toString();   // BigInt: a 64-bit seed does not fit a JS number
  controlsChanged();
});

function request() {
  const spec = modelSpec(draft.model), body = { model: draft.model, params: draft.params, rows: draft.rows };
  if (usesSeed(spec, draft.params)) body.seed = String(draft.seed);
  if (draft.date) body.date = draft.date;
  return body;
}
function sameSettings(result) {
  if (!result) return false;
  const r = request();
  return result.model === r.model && JSON.stringify(result.params) === JSON.stringify(r.params)
    && (result.seed ?? null) === (r.seed ?? null) && result.rows === r.rows
    && result.target_date === (r.date || STATE.next_draw) && result.archive.sha256 === STATE.archive.sha256;
}
function markStale() {
  const note = $("#stale");
  if (note) note.hidden = sameSettings(shown);
  renderLabConfig();
}
function showFieldErrors(root, error) {
  $$("[data-error]", root).forEach(el => { el.textContent = (error && error.fields && error.fields[el.dataset.error]) || ""; });
}

// ------------------------------------------------------------- Predict: run
let predictRun = 0;
$("#controls").addEventListener("submit", async e => {
  e.preventDefault();
  const run = ++predictRun, out = $("#result");   // a newer click wins over an older job finishing late
  showFieldErrors($("#controls"), null); showBanner("");
  const job = runJob("/api/predict", request(), status => {
    if (run !== predictRun) return;
    out.innerHTML = progressHTML(status, "Working");
    $("[data-cancel]", out).addEventListener("click", () => job.cancel());
  });
  try {
    const result = await job;
    if (run !== predictRun) return;
    shown = result; renderResult();
  } catch (error) {
    if (run !== predictRun) return;
    showFieldErrors($("#controls"), error);
    const inField = error.fields && Object.keys(error.fields).length;
    out.innerHTML = shown ? "" : emptyResult();
    if (shown) renderResult();
    if (!error.cancelled && !inField) showBanner(error.message);
    if (inField) showBanner("Some settings are not valid; see the messages beside them.");
  }
});

const emptyResult = () => '<div class="empty">Choose a model and press <b>Generate</b> to see its ticket, a form and the odds.</div>';

// A grid of numbered cells shaded by value, one hue from light to dark.
// `null` values are numbers the model excluded; picks are ringed.
function gridHTML(values, picks, describe, strong = false) {
  const finite = values.filter(v => v !== null), lo = Math.min(...finite), hi = Math.max(...finite);
  return `<div class="slip${strong ? " strong" : ""}">` + values.map((v, i) => {
    const n = i + 1, pick = picks.includes(n) ? " pick" : "";
    if (v === null) return `<div class="cell out${pick}" title="${n}: excluded by the model" tabindex="0">${n}</div>`;
    const t = hi === lo ? 0 : (v - lo) / (hi - lo);
    return `<div class="cell${pick}" tabindex="0" title="${esc(describe(n, v))}" style="background:color-mix(in srgb,var(--series-1) ${Math.round(8 + 92 * t)}%,var(--surface));color:${t > 0.55 ? "#fff" : "var(--text-primary)"}">${n}</div>`;
  }).join("") + "</div>";
}

function settingsText(r) {
  const parts = Object.entries(r.params).map(([k, v]) => `${k.replace(/_/g, " ")} ${v === null ? "all" : Array.isArray(v) ? v.length + " models" : v}`);
  if (r.seed !== null) parts.push(`seed ${r.seed}`);
  return parts.join(" · ");
}

function renderResult() {
  const r = shown, o = r.odds, out = $("#result");
  const total = r.scores.reduce((a, v) => a + (v ?? 0), 0);
  const describe = (n, v) => r.score_kind === "probability"
    ? `${n}: ${(600 * v / total).toFixed(1)}% chance of being drawn, according to the model`
    : `${n}: score ${Number.isInteger(v) ? v : v.toFixed(3)} (higher is preferred; not a probability)`;
  const lookBack = r.mode === "retrospective";
  const actualHits = r.actual ? Object.assign(new Set(r.actual.slice(0, 6)), { strong: r.actual[6] }) : null;
  const rowsLabel = r.rows === 1 ? "1 row" : `${r.rows} rows`;
  const ref = r.reference;
  const record = ref
    ? `On the published scoreboard (draws through ${day(ref.archive_last_date)}) these exact settings averaged
       <b>${ref.mean_matches.toFixed(3)}</b> matched numbers per draw over ${ref.n_draws} draws. A random ticket averages
       ${STATE.baseline_matches.toFixed(3)}. After correction for ${ref.strategies_compared} strategies this is
       ${ref.significant ? "unusual; see the Lab before trusting it" : "not significant"}.`
    : lookBack ? "A look back has no scoreboard record."
    : r.benchmark ? "These settings are untested: the published scoreboard measured different ones."
    : "A toy generator: it is not part of the scoreboard.";
  const canSave = !lookBack;

  out.innerHTML = `
    <div class="note" id="stale" hidden>Settings changed. Press Generate to update; the result below is from the earlier settings.</div>
    <div class="card">
      <h2>${esc(r.model_label)}</h2>
      <p class="sub">${esc(settingsText(r) || "no settings")} · ${rowsLabel} · draw of ${day(r.target_date)}</p>
      ${lookBack ? `<p class="note">A look back, not a prediction: ${r.benchmark ? `the model was given only the ${r.history_draws.toLocaleString("en")} draws before ${day(r.target_date)}` : "this date has already passed"}.</p>` : ""}
      <div class="balls">${r.ticket.slice(0, 6).map(n => `<div class="ball">${n}</div>`).join("")}
        <span class="plus">strong</span><div class="ball strong">${r.ticket[6]}</div></div>
      <p class="sub">${r.ticket_kind === "sampled" ? "A ticket sampled from the model with this seed." : "Best single ticket: the model's six highest-scored numbers and its top strong number."}
        ${r.rows > 1 ? "It need not appear as a row of the form below." : ""}</p>
      ${r.actual ? `<p class="sub" style="margin-top:10px">Actual result that day: ${balls(r.actual)} · the ticket matched
        <b>${r.ticket.slice(0, 6).filter(n => actualHits.has(n)).length} of 6</b>${r.ticket[6] === r.actual[6] ? " and the strong number" : ""}.</p>` : ""}
    </div>
    <details class="odds">
      <summary><span class="oddsline"><b>${percent(o.p_any_prize)}</b> any prize</span>
        <span class="oddsline"><b>${percent(o.p_three_plus_and_strong)}</b> 3+ with strong</span>
        <span class="oddsline"><b>1 in ${compact(1 / o.p_first_prize)}</b> first prize</span>
        <span class="oddsline"><b>${o.cost} NIS</b> ${rowsLabel}</span>
        <span class="oddsmore">Fixed odds for ${rowsLabel}; details</span></summary>
      <p class="sub">Exact odds of a ${rowsLabel === "1 row" ? "single row" : rowsLabel + " form"} under a fair draw. They depend only on the number of rows and how they are arranged, so they are the same for every model and every seed. A prize is not necessarily a profit.</p>
      <div class="scroll"><table><thead><tr><th>At least one row matches</th><th class="num">Chance</th><th class="num">1 in</th></tr></thead><tbody>
      ${[["3 or more numbers (any prize)", o.p_at_least["3"]], ["4 or more", o.p_at_least["4"]], ["5 or more", o.p_at_least["5"]], ["all 6", o.p_at_least["6"]],
         ["3 or more, with the strong number", o.p_three_plus_and_strong], ["all 6 and the strong number (first prize)", o.p_first_prize]]
        .map(([k, p]) => `<tr><td>${k}</td><td class="num">${percent(p)}</td><td class="num">${oneIn(p)}</td></tr>`).join("")}</tbody></table></div>
    </details>
    <div class="cols" style="margin-top:16px">
      <div class="card">
        <h2>The form, ${rowsLabel}</h2>
        <div class="scroll"><table><thead><tr><th class="num">Row</th><th>Numbers</th><th>Strong</th>${r.actual ? '<th class="num">Matched</th>' : ""}</tr></thead>
        <tbody>${r.form.map((row, i) => `<tr><td class="num">${i + 1}</td><td class="nowrap">${row.slice(0, 6).map(n => `<span class="chip${actualHits && actualHits.has(n) ? " hit" : ""}">${n}</span>`).join("")}</td>
          <td><span class="chip strong${actualHits && actualHits.strong === row[6] ? " hit" : ""}">${row[6]}</span></td>
          ${r.actual ? `<td class="num">${row.slice(0, 6).filter(n => actualHits.has(n)).length}</td>` : ""}</tr>`).join("")}</tbody></table></div>
        <div class="actions"><button class="btn" type="button" id="copy">Copy rows</button><span class="muted" id="copied" aria-live="polite"></span></div>
      </div>
      <div class="card">
        <h2>Why these numbers</h2>
        <p class="sub">Each number shaded by the model's ${r.score_kind === "probability" ? "probability" : "score"} for it. Point at a number for its value.</p>
        ${gridHTML(r.scores, r.ticket.slice(0, 6), describe)}
        <h3>Strong number</h3>
        ${gridHTML(r.strong_scores, [r.ticket[6]], (n, v) => `strong ${n}: ${Number.isInteger(v) ? v : v.toFixed(3)}`, true)}
        <div class="legend"><span>low<span class="ramp"></span>high</span><span><span class="ringkey"></span>in the ticket</span></div>
      </div>
    </div>
    <div style="margin-top:16px">
      <div class="card">
        <h2>Record</h2>
        <p class="sub">${record}</p>
        <div class="actions"><button class="btn" type="button" id="tolab">Test these settings in the Lab</button></div>
        <h3>Keep it</h3>
        <div class="actions" style="margin-top:6px">
          <button class="btn" type="button" id="save"${canSave ? "" : " disabled"}>Save to log</button>
          <a class="btn" href="/report/${r.id}" target="_blank" rel="noopener">Open report</a>
          <a class="btn" href="/report/${r.id}?format=md" target="_blank" rel="noopener">Markdown</a>
        </div>
        <p class="hint" id="savenote" aria-live="polite">${canSave ? "Saving records this prediction before the draw, so it can be scored afterwards." : "A look back cannot be saved as a prediction."}</p>
      </div>
    </div>`;
  $("#copy").addEventListener("click", async () => {
    const text = r.form.map(row => row.slice(0, 6).join(" ") + "  strong " + row[6]).join("\n");
    try { await navigator.clipboard.writeText(text); $("#copied").textContent = "Copied."; }
    catch { $("#copied").textContent = "Copying is blocked by the browser; select the rows instead."; }
  });
  $("#tolab").addEventListener("click", () => showTab("lab"));
  $("#save").addEventListener("click", async () => {
    try {
      const entry = await api("POST", "/api/log", { result_id: r.id });
      $("#savenote").textContent = entry.timing === "before_draw_day"
        ? `Saved at ${clock(entry.saved_utc)} for the draw of ${day(r.target_date)}. See the Log.`
        : `Saved at ${clock(entry.saved_utc)}, on the draw day itself. The dashboard cannot tell whether that was before the draw, so the entry is kept but left out of the running totals.`;
      $("#save").disabled = true;
    } catch (error) { $("#savenote").textContent = (error.fields && error.fields.result) || error.message; }
  });
  markStale();
}

// ------------------------------------------------------------------- Compare
$("#runcompare").addEventListener("click", async () => {
  const button = $("#runcompare"), status = $("#comparestatus");
  button.disabled = true;
  const job = runJob("/api/compare", {}, s => { status.textContent = s.progress.total ? `${s.progress.done + 1} of ${s.progress.total}: ${s.progress.stage}` : "Starting…"; });
  try { comparison = await job; status.textContent = ""; renderCompare(); }
  catch (error) { status.textContent = error.message; }
  button.disabled = false;
});
function renderCompare() {
  const c = comparison, top = c.consensus.map((v, i) => [v, i + 1]).filter(x => x[0] >= 5).map(x => x[1]);
  $("#compareout").innerHTML = `
    <div class="cols" style="grid-template-columns:minmax(0,1.5fr) minmax(0,1fr);margin-top:16px">
      <div class="card">
        <h2>Draw of ${day(c.target_date)}</h2>
        <p class="sub">Record: mean matched numbers per draw on the published scoreboard; a random ticket averages ${STATE.baseline_matches.toFixed(3)}.</p>
        <div class="scroll"><table><thead><tr><th>Model</th><th>Best single ticket</th><th class="num">Record</th><th></th></tr></thead><tbody>
        ${c.rows.map(r => `<tr><td>${esc(r.name)}</td>
          <td class="nowrap">${r.state === "complete" ? balls(r.ticket) : `<span class="muted">${esc(r.message)}</span>`}</td>
          <td class="num">${r.reference_mean === null ? "—" : r.reference_mean.toFixed(3)}</td>
          <td>${r.state === "complete" ? `<button class="btn" type="button" data-open="${r.preset}">Open</button>` : ""}</td></tr>`).join("")}</tbody></table></div>
      </div>
      <div class="card">
        <h2>Consensus board</h2>
        <p class="sub">How many of the ${c.counted} models put each number in their ticket.</p>
        ${gridHTML(c.consensus, top, (n, v) => `${n}: picked by ${v} of ${c.counted} models`)}
        <div class="legend"><span>0<span class="ramp"></span>${Math.max(...c.consensus)} models</span><span><span class="ringkey"></span>five or more</span></div>
        <p class="sub" style="margin-top:12px">${top.length ? `Ringed: ${top.join(", ")}.` : "No number is picked by five or more models."}
          Agreement describes the models, most of which read the same history. It is not confidence.</p>
        <div class="actions"><a class="btn" href="/report/${c.id}" target="_blank" rel="noopener">Open report</a>
          <a class="btn" href="/report/${c.id}?format=md" target="_blank" rel="noopener">Markdown</a></div>
      </div>
    </div>`;
  $$("[data-open]").forEach(b => b.addEventListener("click", () => {
    applyPreset(STATE.presets.find(p => p.id === b.dataset.open));
    showTab("predict"); $("#generate").click();
  }));
}

// ----------------------------------------------------------------------- Lab
function renderLabConfig() {
  if (!STATE || !draft) return;
  const spec = modelSpec(draft.model), r = request();
  $("#labconfig").innerHTML = `<b>${esc(spec.label)}</b><br><span class="sub">${esc(settingsText({ params: r.params, seed: r.seed ?? null }) || "no settings")}</span>`;
  $("#runlab").disabled = !spec.available;
}
$("#ntest").addEventListener("input", e => { $("#ntestv").textContent = e.target.value; });
$("#runlab").addEventListener("click", async () => {
  const button = $("#runlab"), status = $("#labstatus"), r = request();
  button.disabled = true; $("#laberror").textContent = "";
  const job = runJob("/api/backtest", { model: r.model, params: r.params, seed: r.seed, n_test: Number($("#ntest").value) }, s => {
    status.innerHTML = `${esc(s.progress.stage)} <button class="btn" type="button" data-cancel>Cancel</button>`;
    $("[data-cancel]", status).addEventListener("click", () => job.cancel());
  });
  try { const done = await job; lastAttempt = done.attempt_id; renderLab(done.notebook); }
  catch (error) {
    $("#laberror").textContent = error.fields && Object.keys(error.fields).length ? Object.values(error.fields).join(" ") : error.message;
    renderLab(await api("GET", "/api/notebook"));
  }
  status.textContent = ""; button.disabled = false;
});
function renderLab(notebook) {
  const done = notebook.attempts.filter(a => a.state === "complete");
  const a = done.find(x => x.id === lastAttempt) || done[done.length - 1];
  const verdict = x => x.state !== "complete" ? `<span class="pill">${x.state}</span>`
    : x.significant ? '<span class="pill warn">unusual in this notebook</span>' : '<span class="pill">not significant</span>';
  $("#labout").innerHTML = (a ? `
    <div class="tiles" style="margin-top:0">
      <div class="tile hero"><div class="k">Mean matches per draw</div><div class="v">${a.mean_matches.toFixed(3)}</div><div class="n">chance ${notebook.baseline.toFixed(3)}</div></div>
      <div class="tile"><div class="k">p-value, two-sided</div><div class="v">${a.p_value.toFixed(3)}</div><div class="n">on its own, before correction</div></div>
      <div class="tile"><div class="k">Across ${notebook.completed} attempt${notebook.completed === 1 ? "" : "s"}</div><div class="v" style="font-size:1.1rem;padding-top:9px">${a.significant ? "Unusual in this notebook" : "Not significant"}</div><div class="n">${a.significant ? "exploratory: confirm on future draws" : "Holm correction over this notebook"}</div></div>
    </div>
    <div class="card"><h2>${esc(a.label)}</h2>
      <p class="sub">${a.draws} draws scored, ${day(a.first_date)} to ${day(a.last_date)}${a.draws < a.requested_draws ? ` (asked for ${a.requested_draws}; the first 200 draws are kept as history)` : ""}. How many numbers it matched in each draw, beside what chance predicts.</p>
      <div class="chart">${a.chart}</div>
      <p class="hint">A low p-value can also mean matching fewer numbers than chance. Neither direction is evidence until it holds on draws not yet made.</p></div>` : '<div class="empty">Run a backtest to see how the current settings would have scored on past draws.</div>') + `
    <div class="card"><h2>Notebook</h2>
      <p class="sub">Every setting tried, including abandoned runs, kept between visits. This is exploration: choosing new settings after seeing results makes the tries dependent, and the correction covers only the tries listed here.</p>
      ${notebook.attempts.length ? `<div class="scroll"><table><thead><tr><th>Settings</th><th class="num">Draws</th><th class="num">Mean matches</th><th class="num">p</th><th>After correction</th></tr></thead><tbody>
      ${notebook.attempts.slice().reverse().map(x => `<tr><td>${esc(x.label)}</td><td class="num">${x.draws ?? "—"}</td>
        <td class="num">${x.state === "complete" ? x.mean_matches.toFixed(3) : "—"}</td><td class="num">${x.state === "complete" ? x.p_value.toFixed(3) : "—"}</td><td>${verdict(x)}</td></tr>`).join("")}</tbody></table></div>`
      : '<p class="hint">Nothing tried yet.</p>'}</div>`;
}

// ----------------------------------------------------------------------- Log
async function loadLog() {
  let log;
  try { log = await api("GET", "/api/log"); } catch (error) { $("#logout").innerHTML = `<p class="note error">${esc(error.message)}</p>`; return; }
  const s = log.summary;
  const status = e => e.state === "waiting" ? '<span class="pill wait">waiting for the result</span>'
    : e.prize ? '<span class="pill good">prize</span>' : '<span class="pill">no prize</span>';
  $("#logout").innerHTML = `
    <div class="tiles">
      <div class="tile"><div class="k">Saved predictions</div><div class="v">${s.saved}</div><div class="n">${s.waiting} waiting for a result</div></div>
      <div class="tile"><div class="k">Forms with a prize</div><div class="v">${s.with_prize} of ${s.scored}</div><div class="n">chance expects ${s.expected_with_prize.toFixed(1)}</div></div>
      <div class="tile"><div class="k">Left out of the totals</div><div class="v">${s.unverified}</div><div class="n">saved on the draw day itself</div></div>
    </div>
    <div class="card">${log.entries.length ? `<div class="scroll"><table><thead><tr><th>Draw</th><th>Model</th><th>Ticket</th><th>Result</th><th class="num">Best row</th><th>Status</th></tr></thead><tbody>
      ${log.entries.map(e => {
        const hits = e.actual ? Object.assign(new Set(e.actual.slice(0, 6)), { strong: e.actual[6] }) : null;
        return `<tr><td class="nowrap">${day(e.target_date)}</td>
          <td>${esc(e.model_label)}<br><span class="muted">${esc(settingsText(e) || "no settings")} · ${e.rows} row${e.rows === 1 ? "" : "s"}${e.timing === "unverified" ? " · saved on the draw day" : ""}</span>
            <details><summary>Rows</summary>${e.form.map((row, i) => `<div class="nowrap" style="margin:4px 0">${balls(row, hits)}${e.row_scores ? ` <span class="muted">${e.row_scores[i].matches} matched${e.row_scores[i].tier === "no prize" ? "" : " · " + esc(e.row_scores[i].tier)}</span>` : ""}</div>`).join("")}</details></td>
          <td class="nowrap">${balls(e.ticket, hits)}</td>
          <td class="nowrap">${e.actual ? balls(e.actual) : "—"}</td>
          <td class="num">${e.state === "scored" ? e.best_matches + " of 6" : "—"}</td><td>${status(e)}</td></tr>`;
      }).join("")}</tbody></table></div>
      <p class="hint" style="margin-top:10px">Several forms saved for the same draw are scored against the same result, so they rise and fall together. A time recorded on this machine is a personal record, not independent proof.</p>`
      : '<div class="empty" style="border:0">Nothing saved yet. Generate a prediction and press <b>Save to log</b> before the draw.</div>'}</div>`;
}

// ---------------------------------------------------------------------- boot
(async function boot() {
  let theme = "light";
  try { theme = localStorage.getItem("lotto.theme") || "light"; } catch { /* private mode */ }
  setTheme(theme);
  try { STATE = await api("GET", "/api/state"); }
  catch (error) { showBanner(error.message); return; }
  draft = loadDraft() || freshDraft("hot");
  if (!STATE.rows.includes(draft.rows)) draft.rows = 10;
  renderTopBar(); renderControls();
  $("#result").innerHTML = emptyResult();
  renderLab(await api("GET", "/api/notebook"));
  // Show something useful straight away, unless that would start a long training run.
  if (modelSpec(draft.model).family !== "Neural") $("#generate").click();
})();
