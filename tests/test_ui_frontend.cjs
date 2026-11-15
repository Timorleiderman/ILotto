// Run the shipped page script with minimal browser I/O stubs. No build or npm dependencies.
const assert = require("node:assert/strict");
const { readFileSync } = require("node:fs");
const { test } = require("node:test");
const vm = require("node:vm");

const source = readFileSync(`${__dirname}/../lotto/ui/static/app.js`, "utf8");
function page() {
  const elements = new Map(), saved = new Map();
  const element = selector => {
    if (!elements.has(selector)) elements.set(selector, {
      content: "test-token", value: "", textContent: "", hidden: false, validity: { badInput: false },
      addEventListener() {}, classList: { toggle() {} },
    });
    return elements.get(selector);
  };
  const context = vm.createContext({
    document: { querySelector: element, querySelectorAll: () => [], addEventListener() {}, documentElement: { dataset: {} } },
    localStorage: { getItem: key => saved.get(key) ?? null, setItem: (key, value) => saved.set(key, value) },
    matchMedia: () => ({ matches: false }), structuredClone,
    fetch: () => new Promise(() => {}), // pause boot before /api/state returns
    setTimeout,
  });
  vm.runInContext(source, context);
  const run = code => vm.runInContext(code, context);
  run(`STATE = {rows: [1, 2, 10, 14], preset_names: {}, models: [
    {id: "ensemble", seed_default: null, params: [{name: "members", kind: "subset", choices: ["hot", "cold"], default: ["hot"]}]},
    {id: "date_cb", seed_default: 0, params: [{name: "ticket", kind: "choice", choices: ["top six", "sampled"], default: "top six"}]},
    {id: "hot", seed_default: null, params: [{name: "window", kind: "int_or_all", low: 10, high: 1500, default: null}]}
  ]}`);
  return { context, run, element, save: value => saved.set("lotto.draft", JSON.stringify(value)) };
}

test("malformed persisted settings recover to valid defaults", () => {
  const p = page();
  for (const members of [7, [], "hot", ["missing"]]) {
    p.save({ model: "ensemble", params: { members } });
    assert.equal(p.run("JSON.stringify(loadDraft().params.members)"), '["hot"]');
  }
  p.save({ model: "ensemble", params: { members: ["cold", "hot", "cold"] } });
  assert.equal(p.run("JSON.stringify(loadDraft().params.members)"), '["hot","cold"]');
  p.save({ model: "date_cb", params: { ticket: 1 } });
  assert.equal(p.run("loadDraft().params.ticket"), "top six");
  for (const window of [-1, 10.5, 1501, "100"]) {
    p.save({ model: "hot", params: { window } });
    assert.equal(p.run("loadDraft().params.window"), null);
  }
  p.save({ model: "hot", params: { window: 100 } });
  assert.equal(p.run("loadDraft().params.window"), 100);
});

test("an unfinished number stays invalid after JSON instead of becoming all draws", () => {
  const p = page();
  p.run('draft = freshDraft("hot")');
  const input = p.element('[data-param="window"]');
  assert.equal(p.run("JSON.stringify(readParams())"), '{"window":null}');
  input.validity.badInput = true;
  assert.equal(p.run("JSON.stringify(readParams())"), '{"window":""}');
  input.validity.badInput = false;
  input.value = "100";
  assert.equal(p.run("JSON.stringify(readParams())"), '{"window":100}');
});

test("a failed cancellation is visible and can be retried", async () => {
  const p = page();
  let deletes = 0, poll;
  p.context.setTimeout = callback => { poll = callback; };
  p.context.fetch = async (path, options) => {
    if (options.method === "DELETE" && ++deletes === 1) throw new Error("offline");
    return { ok: true, json: async () => ({ id: "job", state: deletes === 2 ? "cancelled" : "running" }) };
  };
  const job = p.run('runJob("/api/predict", {}, () => {})');
  await new Promise(setImmediate);
  await job.cancel();
  assert.match(p.element("#banner").textContent, /Could not cancel.*try Cancel again/);
  await job.cancel();
  assert.equal(deletes, 2);
  assert.equal(p.element("#banner").hidden, true);
  poll();
  await assert.rejects(job, error => error.cancelled === true);
});

test("invalid notebook history explains exclusion and cannot become the headline result", () => {
  const p = page();
  p.run(`renderLab({completed: 0, baseline: 0.973, attempts: [
    {id: "old", state: "invalid", label: "Sampled", p_value: 0.00001, message: "Actually tested <top six>."}
  ]})`);
  const html = p.element("#labout").innerHTML;
  assert.match(html, /Actually tested &lt;top six&gt;/);
  assert.doesNotMatch(html, /Mean matches per draw/);
  assert.equal(p.run("pValue(0.00001)"), "<0.001");
});
