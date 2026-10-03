import { scaleSequential } from "d3-scale";
import { interpolateViridis } from "d3-scale-chromatic";
import { DEFAULTS, Fly, type Mode } from "./fly/agent";
import { evaluate } from "./fly/evaluate";
import { type Landscape, applyMutations, parseLandscape, quantile } from "./fly/landscape";
import { mulberry32 } from "./fly/rng";

type UiMode = Mode;
const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;
const css = (name: string) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

const HINTS: Record<UiMode, string> = {
  chemotaxis: "Baseline, not a brain model. Sees the true fitness of all 15 neighbors and steps toward better ones.",
  mushroom:
    "Fly-inspired circuit with random wiring. Never sees fitness of neighbors, only their Kenyon codes; learns code→value from the fitness change it experiences after each move (dopamine-like error).",
  random: "Control: ignores fitness entirely.",
  hemibrain:
    "Same learner as the mushroom body mode, but PN→KC wiring is the real hemibrain synapse-count matrix (135 PNs → 1,785 KCs). Input is a fixed random projection of the embedding to 135 channels. Not a simulation of the whole fly brain.",
};

interface Ghost { mode: Mode; path: number[]; best: number }
const MODE_COLOR: Record<Mode, string> = { chemotaxis: "#f08c00", mushroom: "#1c7ed6", random: "#868e96", hemibrain: "#2f9e44" };
const MODE_NAME: Record<Mode, string> = { chemotaxis: "Chemotaxis", mushroom: "Mushroom body", random: "Random walk", hemibrain: "Hemibrain wiring" };
let ghosts: Ghost[] = [];
let theme: "auto" | "light" | "dark" = "auto";

let L: Landscape;
let fly: Fly;
let start = 0;
let playing = false;
let rng = mulberry32(1);
let color: (f: number) => string;
let mapCanvas = $<HTMLCanvasElement>("map");
let base: HTMLCanvasElement;
let mapW = 0;
const pad = 14;

const px = (i: number) => pad + L.x[i] * (mapW - 2 * pad);
const py = (i: number) => pad + (1 - L.y[i]) * (mapW - 2 * pad);

function params() {
  const mode = $<HTMLSelectElement>("mode").value as UiMode;
  const temperature = mode === "random" ? DEFAULTS.random.temperature : parseFloat($<HTMLInputElement>("temp").value);
  return { mode, params: { ...DEFAULTS[mode], temperature } };
}

function keepGhost() {
  if (fly && fly.path.length > 1) ghosts.push({ mode: fly.mode, path: fly.path.slice(), best: fly.best });
  renderGhosts();
}

function renderGhosts() {
  const ul = $("ghosts");
  ul.innerHTML = "";
  ghosts.forEach((g) => {
    const li = document.createElement("li");
    li.innerHTML = `<i style="background:${MODE_COLOR[g.mode]}"></i>${MODE_NAME[g.mode]}: ${g.path.length - 1} steps, best ${g.best.toFixed(2)}`;
    ul.appendChild(li);
  });
}

/** sameStart keeps earlier trails so modes can be compared from one start. */
function resetFly(newStart = start, sameStart = false) {
  if (sameStart) keepGhost();
  else { ghosts = []; renderGhosts(); }
  start = newStart;
  const { mode, params: p } = params();
  fly = new Fly(L, mode, start, p);
  rng = mulberry32(start * 7919 + 1); // same start + mode => same walk
  $("mode-hint").textContent = HINTS[mode];
  saveHash();
  draw();
}

function saveHash() {
  const { mode } = params();
  const q = new URLSearchParams({ mode, t: $<HTMLInputElement>("temp").value, s: $<HTMLInputElement>("speed").value, start: String(start) });
  if (fly && fly.path.length > 1) q.set("n", String(fly.path.length - 1));
  history.replaceState(null, "", `#${q}`);
}

function loadHash() {
  const q = new URLSearchParams(location.hash.slice(1));
  const mode = q.get("mode");
  if (mode && ["chemotaxis", "mushroom", "random", "hemibrain"].includes(mode) && (mode !== "hemibrain" || L.hemiCodes)) $<HTMLSelectElement>("mode").value = mode;
  const t = parseFloat(q.get("t") ?? "");
  if (t >= 0.01 && t <= 0.5) $<HTMLInputElement>("temp").value = String(t);
  const s = parseFloat(q.get("s") ?? "");
  if (s >= 1 && s <= 60) $<HTMLInputElement>("speed").value = String(s);
  const st = parseInt(q.get("start") ?? "", 10);
  return st >= 0 && st < L.n ? st : null;
}

function applyTheme() {
  if (theme === "auto") document.documentElement.removeAttribute("data-theme");
  else document.documentElement.setAttribute("data-theme", theme);
  $("theme").textContent = `Theme: ${theme}`;
  try { localStorage.setItem("fw-theme", theme); } catch { /* storage may be blocked */ }
  if (L && fly) draw();
}

function sizeCanvas(c: HTMLCanvasElement) {
  const dpr = window.devicePixelRatio || 1;
  const r = c.getBoundingClientRect();
  c.width = Math.round(r.width * dpr);
  c.height = Math.round(r.height * dpr);
  return { w: r.width, h: r.height, dpr };
}

function paintBase() {
  const { w, dpr } = sizeCanvas(mapCanvas);
  mapW = w;
  base = document.createElement("canvas");
  base.width = mapCanvas.width;
  base.height = mapCanvas.height;
  const g = base.getContext("2d")!;
  g.scale(dpr, dpr);
  // draw low-fitness first so bright points sit on top
  const order = Array.from({ length: L.n }, (_, i) => i).sort((a, b) => L.fitness[a] - L.fitness[b]);
  for (const i of order) {
    g.fillStyle = color(L.fitness[i]);
    g.beginPath();
    g.arc(px(i), py(i), 2.4, 0, 6.2832);
    g.fill();
  }
}

function draw() {
  const dpr = window.devicePixelRatio || 1;
  const g = mapCanvas.getContext("2d")!;
  g.setTransform(1, 0, 0, 1, 0, 0);
  g.clearRect(0, 0, mapCanvas.width, mapCanvas.height);
  g.drawImage(base, 0, 0);
  g.scale(dpr, dpr);
  for (const gh of ghosts) {
    g.strokeStyle = MODE_COLOR[gh.mode];
    g.lineWidth = 1.5;
    g.globalAlpha = 0.6;
    g.beginPath();
    gh.path.forEach((v, j) => (j ? g.lineTo(px(v), py(v)) : g.moveTo(px(v), py(v))));
    g.stroke();
  }
  const trail = MODE_COLOR[fly.mode];
  g.strokeStyle = trail;
  g.lineWidth = 1.5;
  g.globalAlpha = 0.85;
  g.beginPath();
  fly.path.forEach((v, j) => (j ? g.lineTo(px(v), py(v)) : g.moveTo(px(v), py(v))));
  g.stroke();
  g.globalAlpha = 1;
  // start marker, then the fly
  g.strokeStyle = css("--text");
  g.lineWidth = 1.5;
  g.strokeRect(px(start) - 4, py(start) - 4, 8, 8);
  g.fillStyle = trail;
  g.strokeStyle = css("--panel");
  g.lineWidth = 2;
  g.beginPath();
  g.arc(px(fly.pos), py(fly.pos), 6, 0, 6.2832);
  g.fill();
  g.stroke();
  updatePanel();
  drawKenyon();
}

function updatePanel() {
  $("s-var").textContent = L.mutants[fly.pos] + (fly.pos === start && fly.path.length === 1 ? " (start)" : "");
  $("s-fit").textContent = L.fitness[fly.pos].toFixed(3);
  $("s-steps").textContent = String(fly.path.length - 1);
  $("s-best").textContent = fly.best.toFixed(3);
}

/** 2000 Kenyon cells as a grid; lit cells are the sparse code. In mushroom mode, colour = learned weight sign. */
function drawKenyon() {
  const c = $<HTMLCanvasElement>("kc");
  const { w, h, dpr } = sizeCanvas(c);
  const g = c.getContext("2d")!;
  g.scale(dpr, dpr);
  const cols = 125;
  const V = fly.view;
  const rows = Math.ceil(V.nKc / cols);
  const cw = w / cols;
  const ch = h / rows;
  g.fillStyle = css("--dim");
  for (let i = 0; i < V.nKc; i++) g.fillRect((i % cols) * cw + 0.3, Math.floor(i / cols) * ch + 0.3, cw - 0.6, ch - 0.6);
  const wts = fly.learner?.w;
  let wmax = 1e-9;
  if (wts) for (let i = 0; i < wts.length; i++) wmax = Math.max(wmax, Math.abs(wts[i]));
  for (const i of V.codes[fly.pos]) {
    const learned = wts !== undefined && Math.abs(wts[i]) > 1e-6;
    g.fillStyle = !wts || !learned ? css("--text") : wts[i] > 0 ? "#e8590c" : "#1971c2";
    g.globalAlpha = learned ? 0.35 + 0.65 * Math.min(1, Math.abs(wts![i]) / wmax) : 0.55;
    g.fillRect((i % cols) * cw, Math.floor(i / cols) * ch, cw, ch);
  }
  g.globalAlpha = 1;
  $("kc-note").textContent = `${V.codes[fly.pos].length} of ${V.nKc} lit` + (wts ? " · grey = untrained, orange = learned positive weight, blue = negative" : " · not used by this walker");
}

function nearest(ev: MouseEvent): number {
  const r = mapCanvas.getBoundingClientRect();
  const mx = ev.clientX - r.left;
  const my = ev.clientY - r.top;
  let best = 0;
  let bd = Infinity;
  for (let i = 0; i < L.n; i++) {
    const d = (px(i) - mx) ** 2 + (py(i) - my) ** 2;
    if (d < bd) { bd = d; best = i; }
  }
  return best;
}

function tick(last = 0, acc = 0) {
  return (now: number) => {
    if (!playing) return;
    acc += now - last;
    const per = 1000 / parseFloat($<HTMLInputElement>("speed").value);
    while (acc >= per) { fly.step(rng); acc -= per; }
    draw();
    requestAnimationFrame(tick(now, acc));
  };
}

function setPlaying(p: boolean) {
  playing = p;
  if (!p) saveHash();
  $("play").textContent = p ? "Pause" : "Play";
  if (p) requestAnimationFrame((t) => tick(t)(t));
}

async function runEval() {
  const body = document.querySelector("#eval tbody")!;
  const btn = $<HTMLButtonElement>("run-eval");
  btn.disabled = true;
  body.innerHTML = "";
  const rows: [Mode, string, boolean][] = [
    ["random", "Random walk", false],
    ["chemotaxis", "Chemotaxis", false],
    ["mushroom", "Mushroom body", false],
    ["mushroom", "Mushroom body, memory kept across starts", true],
    ...(L.hemiCodes
      ? ([["hemibrain", "Hemibrain wiring", false], ["hemibrain", "Hemibrain wiring, memory kept across starts", true]] as [Mode, string, boolean][])
      : []),
  ];
  const mean = (a: number[]) => a.reduce((x, y) => x + y, 0) / a.length;
  for (const [mode, label, carryOver] of rows) {
    btn.textContent = `Running: ${label}…`;
    await new Promise((r) => setTimeout(r, 20));
    const run = (topFrac: number) => [1, 2, 3].map((seed) => evaluate(L, mode, label, DEFAULTS[mode], { seed, topFrac, carryOver }));
    const a = run(0.1);
    const b = run(0.01);
    const tr = document.createElement("tr");
    const cells = [label, mean(a.map((r) => r.medianBest)).toFixed(2), `${(mean(a.map((r) => r.reachRate)) * 100).toFixed(0)}%`, `${(mean(b.map((r) => r.reachRate)) * 100).toFixed(0)}%`];
    cells.forEach((t) => { const td = document.createElement("td"); td.textContent = t; tr.appendChild(td); });
    body.appendChild(tr);
  }
  btn.textContent = "Run evaluation";
  btn.disabled = false;
}

async function main() {
  const raw = await (await fetch(`${import.meta.env.BASE_URL}data.json`)).json();
  L = parseLandscape(raw);
  const lo = quantile(L.fitness, 0.01);
  const hi = quantile(L.fitness, 0.99);
  const scale = scaleSequential(interpolateViridis).domain([lo, hi]).clamp(true);
  color = (f) => scale(f);
  $("ramp").style.background = `linear-gradient(90deg, ${[0, 0.25, 0.5, 0.75, 1].map((t) => interpolateViridis(t)).join(",")})`;
  $("subtitle").textContent = `${L.assay} · ${L.n.toLocaleString()} variants · measured fitness (ProteinGym) · ${L.embedding.split("/").pop()} embeddings`;
  $("honest").textContent =
    "The 2D map is a lossy UMAP picture. The fly walks the real 15-nearest-neighbor graph in embedding space, so a hop can jump across the map. Chemotaxis is a baseline; the mushroom body mode uses random PN→KC wiring; the hemibrain mode uses real PN→KC synapse counts from the hemibrain connectome (only that circuit, not the whole brain).";

  if (!L.hemiCodes) $<HTMLOptionElement>("mode").querySelector<HTMLOptionElement>('option[value="hemibrain"]')!.remove();
  else $<HTMLSelectElement>("mode").querySelector<HTMLOptionElement>('option[value="hemibrain"]')!.disabled = false;
  paintBase();
  try { const th = localStorage.getItem("fw-theme"); if (th === "light" || th === "dark") theme = th; } catch { /* ignore */ }
  applyTheme();
  start = loadHash() ?? Math.floor(Math.random() * L.n);
  $("temp-out").textContent = $<HTMLInputElement>("temp").value;
  const n0 = parseInt(new URLSearchParams(location.hash.slice(1)).get("n") ?? "0", 10);
  resetFly();
  for (let i = 0; i < Math.min(n0, 5000); i++) fly.step(rng);
  if (n0 > 0) draw();

  $("play").onclick = () => setPlaying(!playing);
  $("step").onclick = () => { setPlaying(false); fly.step(rng); saveHash(); draw(); };
  $("reset").onclick = () => { setPlaying(false); resetFly(start, true); };
  $("keep").onclick = () => { keepGhost(); draw(); };
  $("clear").onclick = () => { ghosts = []; renderGhosts(); draw(); };
  $("theme").onclick = () => { theme = theme === "auto" ? "light" : theme === "light" ? "dark" : "auto"; applyTheme(); };
  $("speed").oninput = saveHash;
  $("mode").onchange = () => { setPlaying(false); resetFly(start, true); };
  $("temp").oninput = () => {
    const t = parseFloat($<HTMLInputElement>("temp").value);
    $("temp-out").textContent = String(t);
    fly.params.temperature = params().params.temperature;
    saveHash();
  };
  $("run-eval").onclick = runEval;
  mapCanvas.onclick = (ev) => { setPlaying(false); resetFly(nearest(ev)); };

  const tip = $("tip");
  mapCanvas.onmousemove = (ev) => {
    const i = nearest(ev);
    const m = L.mutants[i];
    const pos = parseInt(m.slice(1, -1), 10) - 1;
    const seq = applyMutations(L.wt, m);
    const a = Math.max(0, pos - 8);
    const win = `${seq.slice(a, pos)}[${seq[pos]}]${seq.slice(pos + 1, pos + 9)}`;
    tip.hidden = false;
    tip.innerHTML = `<b>${m}</b> · fitness ${L.fitness[i].toFixed(3)}<br><code>${a > 0 ? "…" : ""}${win}…</code>`;
    const r = mapCanvas.getBoundingClientRect();
    tip.style.left = `${Math.min(ev.clientX - r.left + 12, r.width - 200)}px`;
    tip.style.top = `${ev.clientY - r.top + 12}px`;
  };
  mapCanvas.onmouseleave = () => (tip.hidden = true);

  let resizeTimer: number;
  window.addEventListener("resize", () => {
    clearTimeout(resizeTimer);
    resizeTimer = window.setTimeout(() => { paintBase(); draw(); }, 100);
  });
  window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", draw);
}

main().catch((e) => ($("subtitle").textContent = `Failed to load: ${e}`));
