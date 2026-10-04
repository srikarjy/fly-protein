/** Coordinator of the replay demo. One ReplayStore holds the current step; every control writes to it and every panel reads from it.
 *  This file wires DOM, renderers and the frame loop; it holds no scientific state (that lives in the immutable StepViews). */
import { $, getJSON, reducedMotion } from "../lib/dom";
import { esc } from "../lib/charts";
import type { Circuit, Landscape, Trace } from "../lib/types";
import { SETTLED, glideReveal, revealAt, specOf, type Reveal, type StepSpec } from "./choreo";
import { Circuit2D, Landscape2D } from "./gfx/fallback2d";
import { webglAvailable, type InspectHit, type LandscapeRenderer } from "./gfx/common";
import type { CircuitHit } from "./gfx/circuit3d";
import { fmt, inspectVariant, nf, pctText } from "./models";
import { Panels } from "./panels";
import { Playback } from "./playback";
import { sha256Int32, verifyFlyReplay } from "./replay";
import { circuitShown, createEngine, type StepView } from "./state";
import { ReplayStore, deepLinkFor, parseDeepLink, type StepChange } from "./store";
import { renderTrend } from "./trend";

type CircuitRenderer = { resize(): void; render(v: StepView, r: Reveal, now: number, dt: number): boolean; setTheme(): void; dispose(): void };

const LAYOUT = (B: number) => `
  <p class="integrity" role="note"><b>Replay, not simulation.</b> Visual motion is interpolated for presentation. Protein variants, neural activations, learner scores, fitness values and state transitions are replayed from recorded benchmark runs. This is an in-silico ProteinGym benchmark, not a simulation of a fly brain, and makes no wet-lab claims.</p>
  <div class="demo-controls" role="group" aria-label="Playback controls">
    <button id="d-play" class="btn primary" type="button">Start</button>
    <button id="d-pause" class="btn" type="button" disabled>Pause</button>
    <button id="d-prev" class="btn" type="button" aria-label="Previous measurement">‹ Prev</button>
    <button id="d-step" class="btn" type="button" aria-label="Next measurement">Next ›</button>
    <button id="d-reset" class="btn" type="button">Reset</button>
    <label class="scrub"><span>Measurement</span><span class="track"><input id="d-scrub" type="range" min="0" max="${B}" value="0" aria-label="Measurement number (timeline)"/><span class="ticks" id="d-ticks" aria-hidden="true"></span></span></label>
    <label class="stepnum"><span class="sr">Go to measurement</span><input id="d-num" type="number" min="0" max="${B}" value="0" inputmode="numeric" aria-label="Go to measurement number"/></label>
    <label class="speed">Speed <select id="d-speed" aria-label="Playback speed"><option value="0.5">0.5×</option><option value="1" selected>1×</option><option value="2">2×</option><option value="4">4×</option><option value="8">8×</option></select></label>
    <label class="toggle"><input id="d-stages" type="checkbox" checked/> Stage animation</label>
    <output id="d-status" class="sr"></output>
  </div>
  <div class="budget" aria-hidden="true"><div class="budget-bar"><i id="d-budget-bar"></i></div><span id="d-budget-text"></span></div>
  <div class="demo-grid">
    <section class="panel" id="p-fly" aria-label="Fly learner">
      <header><h3><i class="dot" style="background:var(--c-fly)"></i>Fly learner</h3><p>Sparse mushroom-body-style code; learns which Kenyon cells predict fitness gains from each move's reward.</p></header>
      <div class="chips" id="fly-chips"></div>
      <div class="mapwrap" id="map-fly"></div>
      <p class="legend map-legend"><span><i class="lg ring-top"></i>assay top 1%</span><span><i class="lg dot-fly"></i>measured</span><span><i class="lg ring-cand"></i>candidate neighbour (size = move probability)</span><span><i class="lg ring-best"></i>best so far</span></p>
    </section>
    <section class="panel" id="p-bo" aria-label="Gaussian-process Bayesian optimisation">
      <header><h3><i class="dot" style="background:var(--c-bo)"></i>GP-BO <span class="muted">(Matérn-5/2, expected improvement)</span></h3><p>Fits a Gaussian process to what it has measured, then measures the candidate with the highest expected improvement.</p></header>
      <div class="chips" id="bo-chips"></div>
      <div class="mapwrap" id="map-bo"></div>
      <div class="sub-h">Top-5 candidates by expected improvement <span class="muted">· model prediction before measuring</span></div>
      <div id="bo-top" class="botop"></div>
      <div id="bo-note" class="memory"></div>
    </section>
  </div>
  <section class="panel brain" id="p-brain" aria-label="Inside the fly learner's circuit">
    <header><h3><i class="dot" style="background:var(--c-fly)"></i>Inside the fly learner <span class="muted">· one recorded step at a time</span></h3>
      <p>Each step: the variant's ESM-2 embedding drives the input channels, the fixed PN→KC wiring turns it into a sparse Kenyon-cell code, the learned weights score the 15 neighbours, one mutation is taken, its fitness is measured, and the weights are updated by the reward-prediction error.</p></header>
    <ol class="pipeline" id="pipeline" aria-label="Pipeline of the current step"></ol>
    <details class="brain-details" id="brain-details" open>
      <summary>Circuit view <span class="muted" id="kc-summary"></span></summary>
      <div class="brain-grid">
        <div class="circuit-wrap"><div class="circuit-host" id="circuit"></div>
          <p class="legend circuit-legend"><span><i class="lg dot-fly"></i>positive input / weight</span><span><i class="lg dot-bo"></i>negative</span><span><i class="lg dot-off"></i>silent Kenyon cell</span><span><i class="lg ring-up"></i>+a after update</span><span><i class="lg ring-down"></i>−a after update</span></p></div>
        <aside class="readout"><div class="sub-h">Learner readout <span class="muted">· last decision</span></div><div id="fly-bars" class="bars"></div><div id="fly-memory" class="memory"></div></aside>
      </div>
      <p class="demo-foot">Edges are the real PN→KC wiring from every input channel into each <i>active</i> Kenyon cell only (the other ~60,000 connections are not drawn); edge strength is that input's recorded value. Hover or tap a unit to inspect its recorded value and wiring. The circuit shows the code of the variant the learner decides from; the candidates' own codes are not part of the recording, so their recorded scores are shown instead.</p>
    </details>
  </section>
  <div class="trend"><div class="sub-h">Best percentile found vs. measurements <span class="muted">· recorded, both runs share the starting variant and the ${B}-measurement budget</span></div><div id="d-trend"></div></div>
  <p class="demo-foot" id="d-verify"></p>
  <p class="demo-foot">The map is a 2-D UMAP of the ESM-2 8M embeddings, coloured by every variant's measured fitness. It is a picture for the viewer: the search methods never see it, they only see fitness of variants they have measured. The fly walks the 15-nearest-neighbour graph in the original embedding space, so one move can jump across the picture. A step is one budget-spending measurement; between measurements the fly can make free moves onto variants it has already measured (recorded, and drawn as part of the step's path). Pink rings mark the assay's top 1%.</p>
  <div class="dtip" id="d-tip" role="tooltip" hidden></div>`;

export async function initDemo() {
  const root = $("#demo-body");
  const rawTrace = fetch(`${import.meta.env.BASE_URL}data/demo_trace.json`).then((r) => { if (!r.ok) throw new Error(`demo_trace.json: ${r.status}`); return r.arrayBuffer(); });
  const [land, traceBytes, circ] = await Promise.all([getJSON<Landscape>("demo_landscape.json"), rawTrace, getJSON<Circuit>("demo_circuit.json")]);
  const trace = JSON.parse(new TextDecoder().decode(traceBytes)) as Trace;
  const engine = createEngine(trace, land, circ);
  const store = new ReplayStore(engine);
  const B = engine.budget;
  root.innerHTML = LAYOUT(B);

  const panels = new Panels(engine);
  const els = {
    play: $<HTMLButtonElement>("#d-play"), pause: $<HTMLButtonElement>("#d-pause"), prev: $<HTMLButtonElement>("#d-prev"), step: $<HTMLButtonElement>("#d-step"), reset: $<HTMLButtonElement>("#d-reset"),
    scrub: $<HTMLInputElement>("#d-scrub"), num: $<HTMLInputElement>("#d-num"), speed: $<HTMLSelectElement>("#d-speed"), stages: $<HTMLInputElement>("#d-stages"),
    status: $("#d-status"), tip: $("#d-tip"), details: $<HTMLDetailsElement>("#brain-details"),
  };
  let reduced = reducedMotion();
  const mqReduce = window.matchMedia("(prefers-reduced-motion: reduce)");
  const mobile = window.matchMedia("(max-width: 720px)");

  // ---- timeline markers: where the recorded runs restart or hand over (all from the trace)
  const marks = trace.fly.events.filter((e) => e.type === "restart").map((e) => ({ q: e.q, label: `fly learner restarts at a random variant (measurement ${e.q})` }));
  marks.push({ q: 10, label: "GP-BO: random initial design ends, Bayesian steps begin (measurement 10)" });
  $("#d-ticks").innerHTML = marks.map((m) => `<i style="left:calc(8px + (100% - 16px) * ${m.q / B})" title="${esc(m.label)}"></i>`).join("");

  // ---- renderers (three.js is a separate chunk, loaded only when the demo is, and only if WebGL works)
  let glReady = false;
  const lands: Record<"fly" | "bo", LandscapeRenderer | null> = { fly: null, bo: null };
  let circuit: CircuitRenderer | null = null;
  const hooks = () => ({
    onInspect: (hit: InspectHit | null) => (hit ? showTip(landTip(hit.variant), hit.clientX, hit.clientY) : hideTip()),
    onLost: () => fallBack(),
  });
  const landTip = (i: number) => {
    const t = inspectVariant(engine, i);
    return `<b>${esc(t.name)}</b><br>fitness ${fmt(t.fitness, 3)} · ${pctText(t.pct)}<br><span class="muted">${t.flyStep ? `fly learner measured it at step ${t.flyStep}` : "not measured by the fly learner"}<br>${t.boStep ? `GP-BO measured it at step ${t.boStep}` : "not measured by GP-BO"}</span>`;
  };
  const circuitTip = (h: CircuitHit) => {
    const v = store.view;
    const c = circuitShown(v);
    if (h.kind === "pn") return `<b>Input channel ${h.index}</b><br>value ${c.pnValues[h.index] >= 0 ? "+" : "−"}${Math.abs(c.pnValues[h.index]).toFixed(4)} <span class="muted">(embedding units; 8-bit ${c.pn[h.index]})</span><br><span class="muted">${esc(engine.land.mutants[c.variant])} · mean-centred ESM-2 channel</span>`;
    const j = c.active.indexOf(h.index);
    const w = v.fly.weightsAfter[h.index];
    return `<b>Kenyon cell ${h.index}</b> · ${j >= 0 ? "active" : "silent"}<br>${j >= 0 ? `input drive ${c.drive[j].toFixed(4)} (cut-off ${c.cutoff[0].toFixed(4)})<br>` : `below the cut-off ${c.cutoff[0].toFixed(4)}<br>`}learned weight ${w >= 0 ? "+" : "−"}${Math.abs(w).toExponential(2)}<br><span class="muted">samples ${circ.meta.per_kc} of ${circ.meta.n_pn} input channels (highlighted)</span>`;
  };
  function showTip(html: string, x: number, y: number) {
    els.tip.innerHTML = html;
    els.tip.hidden = false;
    const r = els.tip.getBoundingClientRect();
    els.tip.style.left = `${Math.min(window.innerWidth - r.width - 8, x + 14)}px`;
    els.tip.style.top = `${Math.min(window.innerHeight - r.height - 8, y + 14)}px`;
  }
  const hideTip = () => { els.tip.hidden = true; };

  function makeFallbacks() {
    for (const m of ["fly", "bo"] as const) { lands[m]?.dispose(); lands[m] = new Landscape2D($(`#map-${m}`), engine, m, hooks()); }
    circuit?.dispose();
    $("#circuit").classList.add("flat");
    circuit = new Circuit2D($("#circuit"), engine);
    renderNow();
  }
  function fallBack() { glReady = false; makeFallbacks(); }

  async function createCircuitGL() {
    if (circuit) return;
    if (!glReady) { circuit = new Circuit2D($("#circuit"), engine); return; }
    const gfx = await import("./gfx/webgl");
    circuit = new gfx.CircuitGL($("#circuit"), engine, { onInspect: (h) => (h ? showTip(circuitTip(h), h.clientX, h.clientY) : hideTip()), onLost: () => fallBack() }, reduced);
    observe($("#circuit"), () => circuit?.resize());
  }

  async function createRenderers() {
    const wantGL = webglAvailable() && !new URLSearchParams(location.search).has("renderer2d");
    if (wantGL) {
      try {
        const gfx = await import("./gfx/webgl");
        for (const m of ["fly", "bo"] as const) lands[m] = new gfx.LandscapeGL($(`#map-${m}`), engine, m, hooks());
        glReady = true;
      } catch (e) {
        console.warn("WebGL renderer unavailable, using the 2D fallback:", e);
        for (const m of ["fly", "bo"] as const) { lands[m]?.dispose(); lands[m] = null; $(`#map-${m}`).replaceChildren(); }
      }
    }
    if (!glReady) for (const m of ["fly", "bo"] as const) lands[m] = new Landscape2D($(`#map-${m}`), engine, m, hooks());
    if (mobile.matches) els.details.open = false; // keep the heavy circuit scene closed on small screens
    if (els.details.open) await createCircuitGL();
  }
  function observe(el: Element, fn: () => void) {
    let t: number | undefined;
    new ResizeObserver(() => { window.clearTimeout(t); t = window.setTimeout(() => { fn(); renderNow(); }, 60); }).observe(el);
  }

  // ---- animation + frame loop
  interface Anim { start: number; spec: StepSpec; mode: "staged" | "glide"; speed: number; glideMs: number }
  let anim: Anim | null = null;
  let raf = 0, last = 0;
  const speed = () => +els.speed.value;
  const interval = () => (reduced ? 900 : 700) / speed();

  function currentReveal(now: number): Reveal {
    if (!anim) return SETTLED;
    const el = Math.max(0, now - anim.start);
    return anim.mode === "staged" ? revealAt(el * anim.speed, anim.spec) : glideReveal(el / anim.glideMs, anim.spec);
  }

  function draw(view: StepView, r: Reveal, now: number, dt: number): boolean {
    let busy = false;
    busy = !!lands.fly?.render(view, r, now, dt) || busy;
    busy = !!lands.bo?.render(view, r, now, dt) || busy;
    if (els.details.open) busy = !!circuit?.render(view, r, now, dt) || busy;
    panels.update(view, r);
    panels.frame(view, r);
    return busy;
  }

  function renderNow() {
    const busy = draw(store.view, currentReveal(performance.now()), performance.now(), 16);
    if (busy || anim) requestFrame();
  }

  function requestFrame() { if (!raf) raf = requestAnimationFrame(frame); }

  function frame(now: number) {
    raf = 0;
    const dt = last ? Math.min(64, now - last) : 16;
    last = now;
    const view = store.view;
    const r = currentReveal(now);
    const busy = draw(view, r, now, dt);
    if (anim && r.settled) {
      const was = anim;
      anim = null;
      settle(view, was.mode === "staged" ? Math.max(40, 220 / was.speed) : Math.max(0, interval() - was.glideMs));
    }
    if (busy || anim) requestFrame(); else last = 0;
  }

  function settle(view: StepView, dwell: number) {
    els.status.textContent = `${view.k} / ${B}`;
    const text = Panels.describe(engine, view);
    $("#map-fly canvas").setAttribute("aria-label", `Fly learner on the protein landscape. ${text}`);
    $("#map-bo canvas").setAttribute("aria-label", `GP-BO on the protein landscape. ${text}`);
    (document.querySelector("#circuit canvas") as HTMLElement | null)?.setAttribute("aria-label", `Circuit: ${$("#kc-summary").textContent}. Input channels, wiring and Kenyon-cell activity of the variant the learner decides from.`);
    $("#d-live").textContent = text;
    playback.settled(dwell);
  }

  // ---- store -> everything
  const playback = new Playback(store, syncButtons);
  const live = document.createElement("div");
  live.id = "d-live"; live.className = "sr"; live.setAttribute("aria-live", "polite");
  root.append(live);

  store.subscribe((c: StepChange) => {
    els.scrub.value = String(c.k);
    els.num.value = String(c.k);
    syncButtons();
    renderTrend($("#d-trend"), engine, c.k);
    const sequential = c.k === c.prevK + 1 && (c.source === "play" || c.source === "next");
    const spec = specOf(c.view);
    if (sequential && !reduced) {
      const staged = els.stages.checked && speed() < 8;
      anim = staged
        ? { start: performance.now(), spec, mode: "staged", speed: speed(), glideMs: 0 }
        : { start: performance.now(), spec, mode: "glide", speed: speed(), glideMs: Math.max(90, Math.min(300, interval() * 0.85)) };
    } else anim = null;
    if (!anim) {
      const busy = draw(c.view, SETTLED, performance.now(), 16);
      settle(c.view, interval());
      if (busy) requestFrame();
    } else requestFrame();
    if (!playback.playing && c.source !== "deeplink") writeHash(c.k);
  });

  function syncButtons() {
    els.play.disabled = playback.playing;
    els.pause.disabled = !playback.playing;
    els.play.textContent = store.k > 0 && store.k < B ? "Resume" : "Start";
    els.prev.disabled = store.k === 0;
    els.step.disabled = store.k === B && !playback.playing;
  }

  // ---- controls: every one goes through the store
  const user = (fn: () => void) => () => { playback.pause(); fn(); };
  els.play.onclick = () => playback.play();
  els.pause.onclick = () => { playback.pause(); writeHash(store.k); };
  els.prev.onclick = user(() => store.prev());
  els.step.onclick = user(() => store.next());
  els.reset.onclick = user(() => { store.reset(); });
  els.scrub.oninput = () => { playback.pause(); store.setStep(+els.scrub.value, "scrub"); };
  els.num.onchange = () => { playback.pause(); const v = Math.max(0, Math.min(B, Math.round(+els.num.value || 0))); els.num.value = String(v); store.setStep(v, "input"); };
  els.speed.onchange = () => { if (anim) { anim = null; renderNow(); settle(store.view, interval()); } };
  root.addEventListener("keydown", (ev: KeyboardEvent) => {
    const t = ev.target as HTMLElement;
    if (/^(INPUT|SELECT|TEXTAREA|SUMMARY)$/.test(t.tagName)) return;
    if (ev.key === "ArrowRight") { playback.pause(); store.next(); ev.preventDefault(); }
    else if (ev.key === "ArrowLeft") { playback.pause(); store.prev(); ev.preventDefault(); }
    else if (ev.key === "Home") { playback.pause(); store.reset(); ev.preventDefault(); }
    else if (ev.key === "End") { playback.pause(); store.setStep(B, "input"); ev.preventDefault(); }
    else if (ev.key === " " && t.tagName === "CANVAS") { playback.toggle(); ev.preventDefault(); }
  });

  // ---- URL hash <-> step (deep link #demo-N)
  let hashTimer: number | undefined;
  function writeHash(k: number) {
    window.clearTimeout(hashTimer);
    hashTimer = window.setTimeout(() => { try { history.replaceState(null, "", deepLinkFor(k)); } catch { /* sandboxed frames */ } }, 250);
  }
  window.addEventListener("hashchange", () => { const k = parseDeepLink(location.hash, B); if (k !== null && k !== store.k) { playback.pause(); store.setStep(k, "deeplink"); } });

  mqReduce.addEventListener("change", () => { reduced = mqReduce.matches; els.stages.disabled = reduced; if (reduced) { els.stages.checked = false; anim = null; renderNow(); } });
  document.addEventListener("visibilitychange", () => { if (document.hidden) playback.pause(); });
  window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => { for (const r of [lands.fly, lands.bo, circuit]) r?.setTheme(); renderNow(); });
  if (reduced) { els.speed.value = "1"; els.stages.checked = false; els.stages.disabled = true; }

  await createRenderers();
  // registered only now: <details open> fires an initial async "toggle", which must not race the renderer set-up
  els.details.addEventListener("toggle", () => { if (els.details.open) createCircuitGL().then(renderNow); });
  for (const m of ["fly", "bo"] as const) observe($(`#map-${m}`), () => lands[m]?.resize());
  const deep = parseDeepLink(location.hash, B);
  if (deep !== null) store.setStep(deep, "deeplink"); // opens the replay at that measurement, every view in sync
  els.scrub.value = els.num.value = String(store.k);
  renderTrend($("#d-trend"), engine, store.k);
  draw(store.view, SETTLED, performance.now(), 16);
  settle(store.view, interval());
  syncButtons();
  window.addEventListener("resize", () => renderTrend($("#d-trend"), engine, store.k));
  if (deep !== null) requestAnimationFrame(() => $("#demo").scrollIntoView({ behavior: "auto", block: "start" }));

  // ---- in-browser integrity check of the replay (unchanged scientific check, plus the circuit file's binding to this trace)
  const v = verifyFlyReplay(trace);
  const h1 = await sha256Int32(engine.queryOrder.fly), h2 = await sha256Int32(engine.queryOrder.bo);
  const okHash = h1 === trace.verified.fly_log_sha256 && h2 === trace.verified.gpbo_log_sha256;
  let bound: boolean | null = null;
  try { const d = await crypto.subtle.digest("SHA-256", traceBytes); bound = [...new Uint8Array(d)].map((b) => b.toString(16).padStart(2, "0")).join("") === circ.trace_sha256; } catch { /* insecure context */ }
  $("#d-verify").innerHTML = v.ok && (okHash || h1 === null) && bound !== false
    ? `<b>Replay self-check passed in your browser.</b> Re-applying the ${trace.fly.events.filter((e) => e.type === "move").length} logged weight updates reproduces the logged weight statistics at all ${v.checked} checkpoints (max relative error ${v.maxRelErr.toExponential(1)})${h1 ? `; the two query logs hash to ${h1.slice(0, 10)}… and ${h2!.slice(0, 10)}…, identical to the benchmark run (assay ${esc(trace.assay)}, seed ${trace.seed} of 20, fixed in advance)` : ""}${bound ? `; the circuit file (${nf.format(circ.meta.n_kc)} Kenyon cells, wiring, drive values) is bound to this trace by SHA-256` : ""}.`
    : `<b>Replay self-check FAILED</b> (weights ok: ${v.ok}, hashes ok: ${okHash}, circuit bound: ${bound}). Do not trust this demo.`;
}
