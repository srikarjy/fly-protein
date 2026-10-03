import { interpolateViridis } from "d3-scale-chromatic";
import { M_COLOR, esc, linePath, linear, spreadLabels, ticks } from "../lib/charts";
import { $, getJSON, reducedMotion } from "../lib/dom";
import type { Landscape, Trace } from "../lib/types";
import { BoReplay, FlyReplay, sha256Int32, softmax, verifyFlyReplay } from "./replay";

const css = (n: string) => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const T_SOFTMAX = 0.1; // temperature used by the recorded run (src/flyprotein/bench/methods.py: fly(T=0.1))

function sizeCanvas(c: HTMLCanvasElement) {
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const w = c.clientWidth;
  const h = c.clientHeight;
  if (c.width !== Math.round(w * dpr) || c.height !== Math.round(h * dpr)) {
    c.width = Math.round(w * dpr);
    c.height = Math.round(h * dpr);
  }
  const g = c.getContext("2d")!;
  g.setTransform(dpr, 0, 0, dpr, 0, 0);
  return { g, w, h };
}

export async function initDemo() {
  const root = $("#demo-body");
  const [land, trace] = await Promise.all([getJSON<Landscape>("demo_landscape.json"), getJSON<Trace>("demo_trace.json")]);
  const fly = new FlyReplay(trace);
  const bo = new BoReplay(trace.gpbo.events);
  const B = trace.budget;
  const pctOf = (i: number) => land.pct[i];
  const bestCurve = (order: number[]) => { let b = 0; return order.map((i) => (b = Math.max(b, pctOf(i)))); };
  const flyBest = bestCurve(fly.queryOrder);
  const boBest = bestCurve(bo.queryOrder);
  const top1 = land.pct.map((p) => p >= 99);

  root.innerHTML = `
  <div class="demo-controls" role="group" aria-label="Playback">
    <button id="d-play" class="btn primary">Start</button>
    <button id="d-pause" class="btn" disabled>Pause</button>
    <button id="d-step" class="btn">Step</button>
    <button id="d-reset" class="btn">Reset</button>
    <label class="speed">Speed <select id="d-speed" aria-label="Playback speed"><option value="1">1×</option><option value="2" selected>2×</option><option value="4">4×</option><option value="8">8×</option></select></label>
    <label class="scrub">Measurement <input id="d-scrub" type="range" min="0" max="${B}" value="0" aria-label="Measurement number"/></label>
    <output id="d-status" aria-live="polite">0 / ${B} measured</output>
  </div>
  <div class="demo-grid">
    <section class="panel" id="p-fly" aria-label="Fly learner">
      <header><h3><i class="dot" style="background:var(--c-fly)"></i>Fly learner</h3><p>Sparse mushroom-body-style code; learns which Kenyon cells predict fitness gains from each move's reward.</p></header>
      <div class="chips" id="fly-chips"></div>
      <div class="mapwrap"><canvas id="map-fly" aria-label="Protein landscape map with the fly learner's measured variants and last decision"></canvas></div>
      <div class="sub-h">Last decision <span class="muted">· 15 neighbours in embedding space, scored by the learner's current weights</span></div>
      <div id="fly-bars" class="bars"></div>
      <div class="sub-h">PN layer <span class="muted">· 320 ESM-2 embedding channels of the current variant (centred, scaled to its largest channel)</span></div>
      <canvas id="pn" class="strip" aria-label="Input channels"></canvas>
      <div class="sub-h">Kenyon-cell layer <span class="muted">· 2,000 units, 100 active (5%) · colour = learned weight</span></div>
      <canvas id="kc" class="kcgrid" aria-label="Kenyon cell activation and learned weights"></canvas>
      <div id="fly-memory" class="memory"></div>
    </section>
    <section class="panel" id="p-bo" aria-label="Gaussian-process Bayesian optimisation">
      <header><h3><i class="dot" style="background:var(--c-bo)"></i>GP-BO <span class="muted">(Matérn-5/2, expected improvement)</span></h3><p>Fits a Gaussian process to what it has measured, then measures the candidate with the highest expected improvement.</p></header>
      <div class="chips" id="bo-chips"></div>
      <div class="mapwrap"><canvas id="map-bo" aria-label="Protein landscape map with GP-BO's measured variants and top candidates"></canvas></div>
      <div class="sub-h">Top-5 candidates by expected improvement <span class="muted">· model prediction before measuring</span></div>
      <div id="bo-top" class="botop"></div>
      <div id="bo-note" class="memory"></div>
    </section>
  </div>
  <div class="trend"><div class="sub-h">Best percentile found vs. measurements <span class="muted">· recorded, both runs share the starting variant and the 200-measurement budget</span></div><div id="d-trend"></div></div>
  <p class="demo-foot" id="d-verify"></p>
  <p class="demo-foot">The map is a 2-D UMAP of the ESM-2 8M embeddings, coloured by every variant's measured fitness. It is a picture for the viewer: the search methods never see it, they only see fitness of variants they have measured. The fly walks the 15-nearest-neighbour graph in the original embedding space, so one move can jump across the picture. Pink rings mark the assay's top 1%.</p>`;

  // ---- base map (drawn once per size/theme)
  const baseCache = new Map<HTMLCanvasElement, HTMLCanvasElement>();
  const PAD = 12;
  const px = (c: { w: number; h: number }, i: number) => PAD + land.x[i] * (c.w - 2 * PAD);
  const py = (c: { w: number; h: number }, i: number) => c.h - PAD - land.y[i] * (c.h - 2 * PAD);
  function baseOf(cv: HTMLCanvasElement) {
    const { w, h } = { w: cv.clientWidth, h: cv.clientHeight };
    let b = baseCache.get(cv);
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    if (!b || b.width !== Math.round(w * dpr) || b.dataset.theme !== document.documentElement.dataset.theme) {
      b = document.createElement("canvas");
      b.width = Math.round(w * dpr);
      b.height = Math.round(h * dpr);
      const g = b.getContext("2d")!;
      g.setTransform(dpr, 0, 0, dpr, 0, 0);
      for (let i = 0; i < land.n; i++) {
        g.fillStyle = interpolateViridis(land.pct[i] / 100);
        g.globalAlpha = 0.55;
        g.beginPath();
        g.arc(px({ w, h }, i), py({ w, h }, i), 2.1, 0, 6.2832);
        g.fill();
      }
      g.globalAlpha = 1;
      g.strokeStyle = css("--c-top1");
      g.lineWidth = 1.1;
      for (let i = 0; i < land.n; i++) if (top1[i]) { g.beginPath(); g.arc(px({ w, h }, i), py({ w, h }, i), 3.6, 0, 6.2832); g.stroke(); }
      baseCache.set(cv, b);
    }
    return b;
  }

  function mutOf(i: number) { return land.mutants[i]; }
  const fmtPct = (p: number) => `${p.toFixed(p >= 99.95 ? 2 : 1)}th percentile`;

  function drawFlyMap(st: ReturnType<FlyReplay["seek"]>) {
    const cv = $<HTMLCanvasElement>("#map-fly");
    const { g, w, h } = sizeCanvas(cv);
    g.clearRect(0, 0, w, h);
    g.drawImage(baseOf(cv), 0, 0, w, h);
    const c = { w, h };
    g.lineWidth = 1.2;
    g.strokeStyle = css("--c-fly");
    g.globalAlpha = 0.45;
    g.beginPath();
    st.recent.forEach((e, i) => { const [a, b] = [e.pos, e.nxt!]; if (i === 0) g.moveTo(px(c, a), py(c, a)); g.lineTo(px(c, b), py(c, b)); });
    g.stroke();
    g.globalAlpha = 1;
    for (const i of st.queried) { g.beginPath(); g.arc(px(c, i), py(c, i), 3.4, 0, 6.2832); g.fillStyle = css("--panel"); g.fill(); g.lineWidth = 1.6; g.strokeStyle = css("--c-fly"); g.stroke(); }
    const lm = st.lastMove && st.last.type === "move" ? st.lastMove : null;
    if (lm) {
      const nb = trace.fly.nbrs[String(lm.pos)];
      const p = softmax(lm.scores!, T_SOFTMAX);
      nb.forEach((j, k) => { g.beginPath(); g.arc(px(c, j), py(c, j), 2.5 + 11 * p[k], 0, 6.2832); g.strokeStyle = css("--ink"); g.globalAlpha = 0.5; g.lineWidth = 1; g.stroke(); });
      g.globalAlpha = 1;
      g.beginPath(); g.moveTo(px(c, lm.pos), py(c, lm.pos)); g.lineTo(px(c, lm.nxt!), py(c, lm.nxt!)); g.strokeStyle = css("--c-fly"); g.lineWidth = 2.4; g.stroke();
      g.beginPath(); g.arc(px(c, lm.pos), py(c, lm.pos), 6, 0, 6.2832); g.strokeStyle = css("--muted"); g.lineWidth = 2; g.stroke();
    }
    const cur = curFly(st);
    g.beginPath(); g.arc(px(c, cur), py(c, cur), 7, 0, 6.2832); g.fillStyle = css("--c-fly"); g.fill(); g.lineWidth = 2.4; g.strokeStyle = css("--bg"); g.stroke();
  }

  function drawBoMap(k: number, ev: (typeof trace.gpbo.events)[number] | undefined) {
    const cv = $<HTMLCanvasElement>("#map-bo");
    const { g, w, h } = sizeCanvas(cv);
    g.clearRect(0, 0, w, h);
    g.drawImage(baseOf(cv), 0, 0, w, h);
    const c = { w, h };
    bo.queryOrder.slice(0, k).forEach((i, n) => {
      g.beginPath(); g.arc(px(c, i), py(c, i), 3.6, 0, 6.2832);
      g.fillStyle = n < 10 ? css("--bg") : css("--panel"); g.fill(); g.lineWidth = 1.6; g.strokeStyle = n < 10 ? css("--muted") : css("--c-bo"); g.stroke();
    });
    if (ev?.top) {
      ev.top.forEach((t, r) => {
        g.beginPath(); g.arc(px(c, t.i), py(c, t.i), 9, 0, 6.2832); g.strokeStyle = css("--c-bo"); g.lineWidth = r === 0 ? 2.6 : 1.4; g.globalAlpha = r === 0 ? 1 : 0.7; g.stroke(); g.globalAlpha = 1;
        g.fillStyle = css("--ink"); g.font = "600 10px ui-sans-serif, system-ui"; g.textAlign = "center"; g.fillText(String(r + 1), px(c, t.i), py(c, t.i) - 12);
      });
    }
    if (!ev) { const st0 = trace.start; g.beginPath(); g.arc(px(c, st0), py(c, st0), 7, 0, 6.2832); g.fillStyle = css("--muted"); g.fill(); g.lineWidth = 2.4; g.strokeStyle = css("--bg"); g.stroke(); }
    if (ev) { g.beginPath(); g.arc(px(c, ev.chosen), py(c, ev.chosen), 7, 0, 6.2832); g.fillStyle = css("--c-bo"); g.fill(); g.lineWidth = 2.4; g.strokeStyle = css("--bg"); g.stroke(); }
  }

  const curFly = (st: ReturnType<FlyReplay["seek"]>) => (st.last.type === "move" ? st.last.nxt! : st.last.pos);

  function drawStrip(cv: HTMLCanvasElement, vals: number[] | null) {
    const { g, w, h } = sizeCanvas(cv);
    g.clearRect(0, 0, w, h);
    if (!vals) return;
    const bw = w / vals.length;
    vals.forEach((v, i) => { const m = Math.abs(v) / 127; g.fillStyle = v >= 0 ? css("--c-fly") : css("--c-bo"); g.globalAlpha = 0.85; g.fillRect(i * bw, h / 2 - (v >= 0 ? m * (h / 2) : 0), Math.max(1, bw - 0.4), m * (h / 2)); });
    g.globalAlpha = 1;
    g.fillStyle = css("--line"); g.fillRect(0, h / 2, w, 1);
  }

  function drawKC(cv: HTMLCanvasElement, st: ReturnType<FlyReplay["seek"]>) {
    const { g, w, h } = sizeCanvas(cv);
    g.clearRect(0, 0, w, h);
    const cols = 125, rows = Math.ceil(trace.n_kc / cols);
    const cw = w / cols, ch = h / rows;
    const cur = String(curFly(st));
    const lit = new Set(trace.fly.codes[cur]);
    let wmax = 1e-12;
    for (const v of fly.w) wmax = Math.max(wmax, Math.abs(v));
    for (let i = 0; i < trace.n_kc; i++) {
      const wi = fly.w[i];
      const t = Math.tanh((2.2 * wi) / wmax);
      g.fillStyle = lit.has(i) ? (wi === 0 ? css("--ink") : t > 0 ? css("--c-fly") : css("--c-bo")) : css("--kc-off");
      g.globalAlpha = lit.has(i) ? (wi === 0 ? 0.55 : 0.4 + 0.6 * Math.abs(t)) : 1;
      g.fillRect((i % cols) * cw + 0.3, Math.floor(i / cols) * ch + 0.3, cw - 0.6, ch - 0.6);
    }
    g.globalAlpha = 1;
    const lm = st.lastMove;
    if (lm && st.last.type === "move" && (lm.a ?? 0) !== 0) {
      const a = new Set(trace.fly.codes[String(lm.nxt)]), b = new Set(trace.fly.codes[String(lm.pos)]);
      g.lineWidth = 1;
      for (const i of a) if (!b.has(i)) { g.strokeStyle = css("--accent"); g.strokeRect((i % cols) * cw + 0.2, Math.floor(i / cols) * ch + 0.2, cw - 0.4, ch - 0.4); }
      for (const i of b) if (!a.has(i)) { g.strokeStyle = css("--muted"); g.strokeRect((i % cols) * cw + 0.2, Math.floor(i / cols) * ch + 0.2, cw - 0.4, ch - 0.4); }
    }
  }

  function render(k: number) {
    $<HTMLInputElement>("#d-scrub").value = String(k);
    $("#d-status").textContent = `${k} / ${B} measured`;
    // ---------- fly
    if (k === 0) {
      fly.seek(0);
      $("#fly-chips").innerHTML = chips([["measured", `0 / ${B}`], ["start", "shared starting variant"]]);
      $("#fly-bars").innerHTML = `<p class="muted">Press Start. Both methods begin at the same variant and spend the same budget of ${B} measurements.</p>`;
      $("#fly-memory").innerHTML = "";
      drawStrip($("#pn"), null);
      drawKC($("#kc"), fly.seek(1));
      drawFlyMap({ ...fly.seek(1), queried: [], recent: [], lastMove: null, last: { ...fly.seek(1).last, type: "init" } });
      $("#bo-chips").innerHTML = chips([["measured", `0 / ${B}`], ["start", "shared starting variant"]]);
      $("#bo-top").innerHTML = "";
      $("#bo-note").innerHTML = "";
      drawBoMap(0, undefined);
      drawTrend(0);
      return;
    }
    const st = fly.seek(k);
    const cur = curFly(st);
    const e = st.last;
    $("#fly-chips").innerHTML = chips([
      ["measured", `${k} / ${B}`], ["best found", fmtPct(flyBest[k - 1])], ["now at", `${mutOf(cur)} · fitness ${land.fitness[cur].toFixed(2)} (${fmtPct(pctOf(cur))})`],
      ["moves", `${st.moves} · ${st.revisits} (${st.moves ? Math.round((100 * st.revisits) / st.moves) : 0}%) onto already-measured variants`], ["episode", `${st.episode}${st.last.type === "restart" ? " · restarted, weights reset" : ""}`],
    ]);
    const lm = st.lastMove;
    if (lm && st.last.type === "move") {
      const p = softmax(lm.scores!, T_SOFTMAX);
      const sc = lm.scores!;
      const mx = Math.max(0.02, ...sc.map(Math.abs));
      const bw = 100 / sc.length;
      $("#fly-bars").innerHTML = `<div class="barrow">${sc.map((s, j) => `<div class="bar ${j === lm.j ? "chosen" : ""}" title="neighbour ${j + 1}: score ${s.toFixed(4)}, choice probability ${(100 * p[j]).toFixed(1)}%"><span class="b" style="height:${Math.max(2, (Math.abs(s) / mx) * 44)}px;${s < 0 ? "background:var(--c-bo)" : ""}"></span><span class="pp">${(100 * p[j]).toFixed(0)}</span></div>`).join("")}</div><p class="muted small">Bars: learner score of each neighbour (sum of learned Kenyon-cell weights). Numbers: probability the move rule gives each neighbour (softmax, T = ${T_SOFTMAX}). Chosen: <b>${esc(mutOf(lm.nxt!))}</b>, ${lm.new ? "newly measured" : "already measured (free revisit)"}${sc.every((s) => s === 0) ? ". All scores are 0 (no learned weights yet), so the move is random." : `. The most likely neighbour had probability ${(100 * Math.max(...p)).toFixed(0)}% (uniform would be ${(100 / p.length).toFixed(1)}%): the learned scores steer the move only ${Math.max(...p) < 0.15 ? "weakly" : "somewhat"}.`}</p>`;
      void bw;
      const a = new Set(trace.fly.codes[String(lm.nxt)]), b = new Set(trace.fly.codes[String(lm.pos)]);
      let up = 0, down = 0;
      for (const i of a) if (!b.has(i)) up++;
      for (const i of b) if (!a.has(i)) down++;
      $("#fly-memory").innerHTML = `<b>Memory update</b> (recorded): reward ${lm.reward!.toFixed(3)} · prediction error δ ${lm.delta!.toFixed(3)} · weight step a = ${lm.a!.toExponential(2)} on ${up} cells raised (+a), ${down} lowered (−a). Weights now: ${e.nz} non-zero of ${trace.n_kc}, Σ|w| = ${e.wabs.toFixed(4)}.`;
    } else {
      $("#fly-bars").innerHTML = `<p class="muted">${st.last.type === "restart" ? "New episode: restarted at a fresh random variant" + (st.last.reset ? ", learned weights reset to zero." : ".") : "First measurement: the shared starting variant."}</p>`;
      $("#fly-memory").innerHTML = `<b>Memory</b>: ${e.nz} non-zero weights of ${trace.n_kc}, Σ|w| = ${e.wabs.toFixed(4)}.`;
    }
    drawStrip($("#pn"), trace.fly.pn[String(cur)]);
    drawKC($("#kc"), st);
    drawFlyMap(st);
    // ---------- GP-BO
    const bs = bo.seek(k);
    const be = bs.last;
    $("#bo-chips").innerHTML = chips([
      ["measured", `${k} / ${B}`], ["best found", fmtPct(boBest[k - 1])], ["last measured", `${mutOf(be.chosen)} · fitness ${be.f.toFixed(2)} (${fmtPct(pctOf(be.chosen))})`],
      ["phase", be.type === "init" ? "initial design (random, 10 measurements)" : `Bayesian step ${k - 10}`],
    ]);
    if (be.type === "bo" && be.top) {
      $("#bo-top").innerHTML = `<table class="botable"><thead><tr><th>#</th><th>variant</th><th>expected improvement</th><th>predicted fitness</th></tr></thead><tbody>${be.top.map((t, r) => `<tr class="${t.i === be.chosen ? "chosen" : ""}"><td>${r + 1}</td><td>${esc(mutOf(t.i))}</td><td>${t.ei.toFixed(3)}</td><td>${t.mu.toFixed(2)} ± ${t.sd.toFixed(2)}</td></tr>`).join("")}</tbody></table>`;
      $("#bo-note").innerHTML = `<b>Surrogate</b>: GP on ${be.n_obs} measured variants, kernel length-scale ${be.ell} (× median pairwise distance), noise ${be.s2}. Measured: <b>${esc(mutOf(be.chosen))}</b> → fitness ${be.f.toFixed(2)} (prediction was ${be.top[0].mu.toFixed(2)} ± ${be.top[0].sd.toFixed(2)}).`;
    } else {
      $("#bo-top").innerHTML = `<p class="muted">The first 10 measurements are a random initial design (the shared start plus 9 random variants); the Gaussian process is fitted afterwards.</p>`;
      $("#bo-note").innerHTML = "";
    }
    drawBoMap(k, be);
    drawTrend(k);
  }

  const chips = (xs: string[][]) => xs.map(([a, b]) => `<span class="chip"><em>${esc(a)}</em> ${esc(b)}</span>`).join("");

  // ---------- trend chart
  function drawTrend(k: number) {
    const narrow = $("#d-trend").clientWidth < 640;
    const W = narrow ? 460 : 900, H = narrow ? 260 : 230, l = narrow ? 40 : 54, r = narrow ? 12 : 150, t = 12, b = 40;
    const lo = Math.floor(Math.min(flyBest[0], boBest[0]) / 5) * 5;
    const x = linear(0, B, l, W - r), y = linear(lo, 100.2, H - b, t);
    const xs = Array.from({ length: B }, (_, i) => i + 1);
    let s = "";
    for (const v of ticks(lo, 100, 4)) s += `<line class="grid" x1="${l}" x2="${W - r}" y1="${y(v)}" y2="${y(v)}"/><text class="tick" x="${l - 8}" y="${y(v) + 4}" text-anchor="end">${v}</text>`;
    for (const v of [0, 50, 100, 150, 200]) s += `<text class="tick" x="${x(v)}" y="${H - b + 17}" text-anchor="middle">${v}</text>`;
    s += `<line class="axis" x1="${l}" x2="${W - r}" y1="${H - b}" y2="${H - b}"/><text class="axlabel" x="${(l + W - r) / 2}" y="${H - 6}" text-anchor="middle">fitness measurements</text>`;
    s += `<line class="grid" x1="${l}" x2="${W - r}" y1="${y(99)}" y2="${y(99)}" stroke-dasharray="3 3"/><text class="tick" x="${x(150)}" y="${y(99) + 17}" text-anchor="middle">assay top 1% (dashed line)</text>`;
    const labels: { y: number; col: string; name: string }[] = [];
    for (const [bst, col, name] of [[flyBest, M_COLOR.fly, "Fly learner"], [boBest, M_COLOR.gpbo_ei, "GP-BO"]] as [number[], string, string][]) {
      s += `<path d="${linePath(xs, bst, x, y)}" fill="none" stroke="${col}" stroke-width="1.4" opacity="0.28"/>`;
      if (k > 0) s += `<path d="${linePath(xs.slice(0, k), bst.slice(0, k), x, y)}" fill="none" stroke="${col}" stroke-width="2.8"/><circle cx="${x(k)}" cy="${y(bst[k - 1])}" r="4.5" fill="${col}"/>`;
      labels.push({ y: y(bst[B - 1]), col, name });
    }
    const sp = spreadLabels(labels.map((l) => l.y), 18, t + 12, H - b - 6);
    if (!narrow) labels.forEach((l, i) => { s += `<text class="mlabel" x="${W - r + 8}" y="${sp[i] + 4}" fill="${l.col}">${l.name}</text>`; });
    else s += `<text class="mlabel" x="${l + 8}" y="${t + 14}" fill="${M_COLOR.gpbo_ei}">GP-BO</text><text class="mlabel" x="${l + 70}" y="${t + 14}" fill="${M_COLOR.fly}">Fly learner</text>`;
    $("#d-trend").innerHTML = `<svg class="chart" viewBox="0 0 ${W} ${H}" role="img" aria-label="Best percentile found versus measurements for both methods">${s}</svg>`;
  }

  // ---------- controls
  let k = 0, timer: number | undefined;
  const speed = () => +$<HTMLSelectElement>("#d-speed").value;
  function setK(v: number) { k = Math.max(0, Math.min(B, v)); render(k); if (k >= B) pause(); }
  function play() {
    if (k >= B) k = 0;
    $<HTMLButtonElement>("#d-play").disabled = true; $<HTMLButtonElement>("#d-pause").disabled = false;
    window.clearInterval(timer);
    timer = window.setInterval(() => setK(k + 1), 700 / speed());
  }
  function pause() { window.clearInterval(timer); timer = undefined; $<HTMLButtonElement>("#d-play").disabled = false; $<HTMLButtonElement>("#d-pause").disabled = true; $("#d-play").textContent = k > 0 && k < B ? "Resume" : "Start"; }
  $("#d-play").onclick = play;
  $("#d-pause").onclick = pause;
  $("#d-step").onclick = () => { pause(); setK(k + 1); };
  $("#d-reset").onclick = () => { pause(); setK(0); $("#d-play").textContent = "Start"; };
  $("#d-speed").onchange = () => { if (timer) play(); };
  $("#d-scrub").oninput = (ev) => { pause(); setK(+(ev.target as HTMLInputElement).value); };
  if (reducedMotion()) $<HTMLSelectElement>("#d-speed").value = "1";
  let rt: number | undefined;
  window.addEventListener("resize", () => { window.clearTimeout(rt); rt = window.setTimeout(() => render(k), 120); });
  window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => { baseCache.clear(); render(k); });
  const deep = location.hash.match(/^#demo-(\d+)$/);
  setK(deep ? Math.min(B, +deep[1]) : 0); // deep link: #demo-90 opens the replay at measurement 90

  // ---------- in-browser integrity check of the replay
  const v = verifyFlyReplay(trace);
  const h1 = await sha256Int32(fly.queryOrder), h2 = await sha256Int32(bo.queryOrder);
  const okHash = h1 === trace.verified.fly_log_sha256 && h2 === trace.verified.gpbo_log_sha256;
  $("#d-verify").innerHTML = v.ok && (okHash || h1 === null)
    ? `<b>Replay self-check passed in your browser.</b> Re-applying the ${trace.fly.events.filter((e) => e.type === "move").length} logged weight updates reproduces the logged weight statistics at all ${v.checked} checkpoints (max relative error ${v.maxRelErr.toExponential(1)})${h1 ? `; the two query logs hash to ${h1.slice(0, 10)}… and ${h2!.slice(0, 10)}…, identical to the benchmark run (assay ${esc(trace.assay)}, seed ${trace.seed} of 20, fixed in advance)` : ""}.`
    : `<b>Replay self-check FAILED</b> (weights ok: ${v.ok}, hashes ok: ${okHash}). Do not trust this demo.`;
}
