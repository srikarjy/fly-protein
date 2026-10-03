import { M_COLOR, ci, dotPlot, esc, groupedBars, smallLines } from "../lib/charts";
import { $ } from "../lib/dom";

/* eslint-disable @typescript-eslint/no-explicit-any */
type J = any;
const MODELS = ["8M", "150M", "650M"];
const MC = ["var(--c-m1)", "var(--c-m2)", "var(--c-m3)"];
const f2 = (x: number) => x.toFixed(2);
const f3 = (x: number) => x.toFixed(3);
const sgn = (x: number, d = 1) => (x >= 0 ? "+" : "−") + Math.abs(x).toFixed(d);

export function renderResults(R: J) {
  representation(R.representation);
  repAblation(R.representation);
  learner(R.learner);
  transfer(R.transfer);
  hemibrain(R.hemibrain);
  benchmark(R.benchmark);
  findings(R);
}

function table(head: string[], rows: (string | number)[][], cls = "") {
  return `<div class="tablewrap"><table class="${cls}"><thead><tr>${head.map((h) => `<th>${h}</th>`).join("")}</tr></thead><tbody>${rows.map((r) => `<tr>${r.map((c, i) => (i === 0 ? `<th scope="row">${c}</th>` : `<td>${c}</td>`)).join("")}</tr>`).join("")}</tbody></table></div>`;
}

function representation(rep: J) {
  const meth: [string, string, string][] = [["dense_cosine", "Dense cosine", "var(--c-walk)"], ["centered_cosine", "Dense centred cosine", "var(--c-random)"], ["simhash", "SimHash (2,000 bits)", "var(--c-evo)"], ["flyhash", "FlyHash (2,000 KC, 5% active)", "var(--c-fly)"]];
  const groups = MODELS.map((m) => ({ label: `ESM-2 ${m}`, bars: meth.map(([k, n, c]) => ({ name: n, v: rep.models[m][k].fit_rho[0], err: rep.models[m][k].fit_rho[1], color: c })) }));
  const keep = MODELS.map((m) => `${m}: ${Math.round((100 * rep.models[m].flyhash.fit_rho[0]) / rep.models[m].centered_cosine.fit_rho[0])}%`).join(" · ");
  $("#rep-chart").innerHTML =
    `<div class="legend">${meth.map(([, n, c]) => `<span><i style="background:${c}"></i>${n}</span>`).join("")}</div>` +
    groupedBars(groups, { y: [0, 1], yLabel: "kNN fitness-prediction Spearman" }) +
    table(["", ...MODELS.flatMap((m) => [`${m} Spearman`, `${m} recall@15`])],
      meth.map(([k, n]) => [n, ...MODELS.flatMap((m) => [`${f3(rep.models[m][k].fit_rho[0])} ± ${f3(rep.models[m][k].fit_rho[1])}`, f3(rep.models[m][k].recall15[0])])]));
  $("#rep-text").innerHTML = `Larger ESM-2 models carry more fitness signal in their neighbourhoods (dense centred cosine: ${MODELS.map((m) => `${m} ${f3(rep.models[m].centered_cosine.fit_rho[0])}`).join(", ")}). The sparse FlyHash code retains most of it (${keep} of the dense signal), although it recovers fewer of the exact nearest neighbours than a same-length SimHash (recall@15 ${f2(rep.models["650M"].flyhash.recall15[0])} vs ${f2(rep.models["650M"].simhash.recall15[0])}). Metric: for each variant, the mean fitness of its 15 most similar variants (by the code) is correlated with its own fitness; 1,000 query variants, mean ± std over 10 random wirings. The scientific question is not whether the code is lossless but whether it carries useful information.`;
}

function repAblation(rep: J) {
  const lines = (get: (m: string) => [number, number][]) => MODELS.map((m, i) => ({ name: `ESM-2 ${m}`, color: MC[i], pts: get(m) }));
  const sp = smallLines(lines((m) => rep.dim_sweep[m]), { x: [64, 1280], y: [0.55, 0.85], xLabel: "input dimension (random projection)", yLabel: "kNN Spearman", log: true, xTicks: [64, 128, 256, 512, 1024] });
  const kc = smallLines(lines((m) => rep.kc_sweep[m].map((r: number[]) => [r[0], r[1]] as [number, number])), { x: [2000, 16000], y: [0.55, 0.85], xLabel: "Kenyon cells", yLabel: "kNN Spearman", log: true, xTicks: [2000, 4000, 8000, 16000] });
  const sps = smallLines(lines((m) => rep.sparsity_sweep[m]), { x: [0.02, 0.1], y: [0.55, 0.85], xLabel: "fraction of KCs active", yLabel: "kNN Spearman", xTicks: [0.02, 0.05, 0.1], xFmt: (v) => `${Math.round(v * 100)}%` });
  const pts = rep.nav_vs_representation as { model: string; ratio: number; gain: number }[];
  const W = 340, H = 250, l = 52, r = 12, t = 12, b = 46;
  const lx = (v: number) => l + ((Math.log2(v) - Math.log2(1.5)) / (Math.log2(260) - Math.log2(1.5))) * (W - l - r);
  const ly = (v: number) => H - b - ((v + 20) / 50) * (H - b - t);
  let sc = [-20, -10, 0, 10, 20, 30].map((v) => `<line class="${v === 0 ? "zero" : "grid"}" x1="${l}" x2="${W - r}" y1="${ly(v)}" y2="${ly(v)}"/><text class="tick" x="${l - 6}" y="${ly(v) + 4}" text-anchor="end">${v}</text>`).join("");
  sc += [2, 8, 32, 128].map((v) => `<text class="tick" x="${lx(v)}" y="${H - b + 17}" text-anchor="middle">${v}</text>`).join("");
  sc += pts.map((p) => `<circle cx="${lx(p.ratio)}" cy="${ly(p.gain)}" r="3.6" fill="${MC[MODELS.indexOf(p.model)]}" opacity="0.8"><title>ESM-2 ${p.model} · KC/d ${p.ratio}: gain ${p.gain} points</title></circle>`).join("");
  sc += `<text class="axlabel" x="${(l + W - r) / 2}" y="${H - 6}" text-anchor="middle">expansion ratio  KC count / input dim</text><text class="axlabel" transform="translate(13 ${(t + H - b) / 2}) rotate(-90)" text-anchor="middle">navigation gain over random walk</text>`;
  const spread = (a: [number, number][]) => Math.max(...a.map((p) => p[1])) - Math.min(...a.map((p) => p[1]));
  const maxSpread = Math.max(...MODELS.map((m) => Math.max(spread(rep.dim_sweep[m]), spread(rep.kc_sweep[m].map((r: number[]) => [r[0], r[1]])))));
  $("#rep-abl").innerHTML = `<div class="small-grid"><figure>${sp}<figcaption>Input dimension</figcaption></figure><figure>${kc}<figcaption>Kenyon-cell count</figcaption></figure><figure>${sps}<figcaption>Sparsity</figcaption></figure><figure><svg class="chart" viewBox="0 0 ${W} ${H}" role="img" aria-label="Navigation gain versus expansion ratio">${sc}</svg><figcaption>Navigation gain vs. expansion ratio (every grid cell)</figcaption></figure></div><div class="legend">${MODELS.map((m, i) => `<span><i style="background:${MC[i]}"></i>ESM-2 ${m}</span>`).join("")}</div>`;
  $("#rep-abl-text").innerHTML = `Across input dimensions from 64 to the full embedding, 2,000 to 16,000 Kenyon cells and 2–10% sparsity, neighbourhood quality moves by at most ${f3(maxSpread)} Spearman within a model, while the model size moves it by about ${f2(rep.models["650M"].flyhash.fit_rho[0] - rep.models["8M"].flyhash.fit_rho[0])}. Navigation gain (right) does not track the expansion ratio or the representation quality. <b>Representation compression was not responsible for poor navigation.</b> The pipeline never compresses the embedding for random wiring; what changes with model size is the ratio of Kenyon cells to input dimensions, and matching that ratio did not recover performance.`;
}

const CONF: Record<string, string> = { baseline: "current learner", "replay5_T0.3": "replay ×5 + temperature 0.3", "T0.3": "temperature 0.3", replay2: "replay ×2" };
function learner(L: J) {
  const rows = (mem: boolean) => L.held_out.filter((r: J) => r.memory === mem && r.config !== "baseline").map((r: J) => ({ label: CONF[r.config] ?? r.config, v: r.diff, lo: r.diff - 1.96 * r.se, hi: r.diff + 1.96 * r.se, color: "var(--c-fly)" }));
  const tune = [...L.tuning].sort((a: J, b: J) => b.mem - a.mem);
  $("#learner-chart").innerHTML = `<div class="two"><figure><figcaption>Memory kept across starts</figcaption>${dotPlot(rows(true), { x: [-12, 12], xLabel: "change in top-1% gain vs. current learner (points)", w: 480, left: 215 })}</figure><figure><figcaption>No memory</figcaption>${dotPlot(rows(false), { x: [-12, 12], xLabel: "change vs. current learner (points)", w: 480, left: 215 })}</figure></div>` +
    `<details><summary>All ${tune.length} variants tried (tuning seeds 100–104)</summary>${table(["variant", "gain, no memory", "gain, memory"], tune.map((r: J) => [esc(r.config), sgn(r.nomem), sgn(r.mem)]))}<p class="muted small">Mean gain over a random walk in points, averaged over three embeddings and five seeds. Lower learning rates, higher temperatures, the absolute rule and replay were tried; the table shows each one's tuning-seed gain.</p></details>`;
  const best = L.held_out.filter((r: J) => r.memory && r.config !== "baseline").sort((a: J, b: J) => b.diff - a.diff)[0];
  $("#learner-text").innerHTML = `Variants were tuned on seeds 100–104 and compared on held-out seeds 0–9, paired against the current learner. The best candidate (${CONF[best.config] ?? best.config}, memory kept) is ${sgn(best.diff)} ± ${best.se.toFixed(1)} points (SE), not distinguishable from zero; the others are lower. <b>Tuning did not produce a reliable improvement.</b>`;
}

function transfer(T: J) {
  const m = T.main;
  const sel: [string, string, string][] = [["zero", "Start from zero", "var(--c-walk)"], ["zero_mem", "Zero start + memory", "var(--c-random)"], ["pre", "Pretrained", "var(--c-fly)"], ["pre_mem", "Pretrained + memory", "var(--c-fly)"], ["shuf_pre", "Shuffled-fitness control", "var(--c-random)"], ["pre_a0.1", "Prior shrunk to 10% (post hoc)", "var(--c-evo)"]];
  const rows = sel.map(([k, n, c]) => ({ label: n, v: m.arms[k][0], lo: m.arms[k][0] - 1.96 * m.arms[k][1], hi: m.arms[k][0] + 1.96 * m.arms[k][1], color: c }));
  const zs = Object.entries(m.zeroshot) as [string, number[]][];
  const bars = groupedBars(zs.map(([a, v]) => ({ label: a, bars: [{ name: "pretrained on other assays", v: v[0], color: "var(--c-fly)" }, { name: "pretrained on shuffled fitness", v: v[1], color: "var(--c-random)" }] })), { y: [-0.05, 0.16], yLabel: "Spearman of learned value vs. fitness", w: 720, h: 280 });
  $("#transfer-chart").innerHTML = `<div class="two"><figure><figcaption>Navigation on the held-out assay</figcaption>${dotPlot(rows, { x: [-20, 8], xLabel: "top-1% gain over a random walk (points)", w: 560, left: 190 })}</figure><figure><figcaption>Ranking signal in the pretrained weights, held-out assay</figcaption>${bars}</figure></div>`;
  $("#transfer-text").innerHTML = `Leave-one-assay-out over 10 unrelated ProteinGym assays (${m.n_pairs} assay × seed pairs; pre-training never sees the held-out fitness). The pretrained weights rank held-out variants by fitness better than chance in every assay (Spearman ${f2(Math.min(...zs.map(([, v]) => v[0])))}–${f2(Math.max(...zs.map(([, v]) => v[0])))}; about 0 when pre-trained on shuffled fitness), yet pretrained navigation is ${sgn(m.pre_minus_zero[0])} ± ${m.pre_minus_zero[1].toFixed(1)} points vs. starting from zero. <b>Cross-assay pretraining produced weak but consistently positive held-out ranking signal, yet failed to improve query-budgeted navigation and often harmed it.</b> Weak transferable ranking information is insufficient for effective sequential search under the current learner. Same conclusion at a different reward scale and with 4× more pre-training.`;
}

function hemibrain(H: J) {
  const rows = MODELS.map((m) => [`ESM-2 ${m}`, f3(H[m].random_1785.fit_rho), f3(H[m].hemibrain.fit_rho), `${sgn(H[m].diff.fit_rho[0], 3)} ± ${H[m].diff.fit_rho[1].toFixed(3)}`, f2(H[m].random_1785.recall15) + " / " + f2(H[m].hemibrain.recall15), `${sgn(H[m].diff.gain[0])} ± ${H[m].diff.gain[1].toFixed(1)}`, `${sgn(H[m].diff.gain_carry[0])} ± ${H[m].diff.gain_carry[1].toFixed(1)}`]);
  $("#hemi-chart").innerHTML = table(["", "kNN Spearman, random wiring", "kNN Spearman, hemibrain", "difference (±SE)", "recall@15 random / hemibrain", "navigation gain difference, no memory", "with memory"], rows);
  $("#hemi-text").innerHTML = `Real hemibrain PN→KC synapse counts (135 PNs → 1,785 KCs, fetched from neuPrint) were compared with random wiring of the same code length, 10 seeds. The biological wiring gives slightly worse neighbourhood quality and no measurable navigation advantage. The input stage (a fixed random projection to 135 channels) is our choice, not a model of the antennal lobe.`;
}

let benchState = { metric: "top1", budget: "50" };
function benchmark(Bm: J) {
  const names: Record<string, string> = Bm.names;
  const order = ["random_search", "random_walk", "greedy_local", "regevo", "gpbo_ei", "adalead_graph", "zs_wt_650M", "zs_wt_150M", "zs_wt_8M", "zs_mm_150M", "zs_mm_8M", "fly", "fly_memory", "fly_best_tested"];
  const MET: Record<string, { label: string; d: number; unit: string }> = { top1: { label: "Reached assay top 1% (%)", d: 1, unit: "%" }, top01: { label: "Reached assay top 0.1% (%)", d: 1, unit: "%" }, pct: { label: "Best percentile found", d: 2, unit: "" }, auc: { label: "Area under best-percentile curve", d: 2, unit: "" } };
  const draw = () => {
    const { metric, budget } = benchState;
    const agg = Bm[metric];
    const rows = order.map((m) => `<tr class="${m.startsWith("fly") ? "fly" : ""}"><th scope="row"><i style="background:${M_COLOR[m]}"></i>${esc(names[m])}${m.startsWith("zs") ? ' <abbr title="Ranking built from the sequence alone; no observed fitness. The budget counts top-ranked candidates then measured.">†</abbr>' : ""}</th>${Bm.budgets.map((b: number) => `<td class="${String(b) === budget ? "sel" : ""}">${ci(agg[m][String(b)], MET[metric].d)}</td>`).join("")}</tr>`).join("");
    const pair = order.filter((m) => m !== "fly").map((m) => { const t = Bm.paired_fly[m][budget]; return { label: names[m], v: t[0], lo: t[1], hi: t[2], color: M_COLOR[m] }; });
    const assays: string[] = Bm.assays;
    const key = metric === "pct" || metric === "auc" ? `pct@${budget}` : `top1@${budget}`;
    const heatRows = order.map((m) => `<tr class="${m.startsWith("fly") ? "fly" : ""}"><th scope="row">${esc(names[m])}</th>${assays.map((a) => { const v = Bm.per_assay[m][a][key]; const t = key.startsWith("top1") ? v / 100 : Math.max(0, (v - 90) / 10); return `<td style="--h:${t.toFixed(3)}">${key.startsWith("top1") ? Math.round(v) : v.toFixed(1)}</td>`; }).join("")}</tr>`).join("");
    $("#bench-body").innerHTML = `
      <div class="seg" role="group" aria-label="Metric">${Object.entries(MET).map(([k, v]) => `<button data-metric="${k}" aria-pressed="${k === metric}">${v.label}</button>`).join("")}</div>
      <div class="tablewrap"><table class="bench"><caption>${MET[metric].label} by number of measured variants (mean over 10 assays; [95% bootstrap CI over assays]). † = zero-shot: ranking built without any observed fitness.</caption><thead><tr><th></th>${Bm.budgets.map((b: number) => `<th class="${String(b) === budget ? "sel" : ""}"><button class="th" data-budget="${b}">${b} measured</button></th>`).join("")}</tr></thead><tbody>${rows}</tbody></table></div>
      <h4>Paired difference to the fly learner at ${budget} measurements <span class="muted">(reached top 1%, points; positive = method better; 95% CI over assays)</span></h4>
      ${dotPlot(pair, { x: [-30, 60], xLabel: "points of P(reach assay top 1%) above the fly learner", w: 760, left: 270 })}
      <h4>Per assay at ${budget} measurements <span class="muted">(${key.startsWith("top1") ? "% of 20 seeds that reached the assay's top 1%" : "mean best percentile"})</span></h4>
      <div class="tablewrap"><table class="heat"><thead><tr><th></th>${assays.map((a) => `<th>${esc(a.split("_")[0])}</th>`).join("")}</tr></thead><tbody>${heatRows}</tbody></table></div>`;
    document.querySelectorAll<HTMLButtonElement>("#bench-body [data-metric]").forEach((b) => (b.onclick = () => { benchState.metric = b.dataset.metric!; draw(); }));
    document.querySelectorAll<HTMLButtonElement>("#bench-body [data-budget]").forEach((b) => (b.onclick = () => { benchState.budget = b.dataset.budget!; draw(); }));
  };
  draw();
}

function findings(R: J) {
  const B = R.benchmark, rep = R.representation, T = R.transfer.main;
  const t1 = (m: string, b = "50") => B.top1[m][b];
  const pf = (m: string, b = "50") => B.paired_fly[m][b];
  const pr = (m: string, b = "50") => B.paired_random[m][b];
  const D: Record<string, number> = { "8M": 320, "150M": 640, "650M": 1280 };
  const base = (m: string) => { const c = rep.nav_vs_representation.find((p: any) => p.model === m && p.d === D[m] && p.n_kc === 2000); return sgn(c.gain); };
  $("#find-list").innerHTML = [
    `<b>Protein language-model representations contain useful fitness signal.</b> Neighbours in ESM-2 space predict fitness (kNN Spearman ${f2(rep.models["8M"].centered_cosine.fit_rho[0])} for 8M up to ${f2(rep.models["650M"].centered_cosine.fit_rho[0])} for 650M), and GP-BO, built on the same embeddings, is ${sgn(pr("gpbo_ei")[0])} points ahead of random search in P(reach top 1%) at 50 measurements (95% CI [${pr("gpbo_ei")[1].toFixed(1)}, ${pr("gpbo_ei")[2].toFixed(1)}]; the gap closes by 100–200 measurements as random search catches up).`,
    `<b>Fly-inspired sparse coding preserves much of that information.</b> FlyHash keeps ${Math.round((100 * rep.models["650M"].flyhash.fit_rho[0]) / rep.models["650M"].centered_cosine.fit_rho[0])}% of the dense neighbourhood signal at ESM-2 650M with 5% of the units active.`,
    `<b>Biological wiring detail does not improve the task.</b> Real hemibrain PN→KC synapse counts were slightly worse than random wiring of the same size and no better at navigation.`,
    `<b>Larger representations do not automatically improve sequential search.</b> ESM-2 650M has the best neighbourhoods, yet the fly learner's navigation gain (memory kept, points over a random walk) is ${base("8M")} for 8M, ${base("150M")} for 150M and ${base("650M")} for 650M, and input dimension, Kenyon-cell count and sparsity did not change that.`,
    `<b>Memory helps the fly learner somewhat but does not make it competitive.</b> With memory the fly learner reaches the assay top 1% in ${t1("fly_memory", "100")[0].toFixed(0)}% of runs at 100 measurements, against ${t1("fly", "100")[0].toFixed(0)}% without and ${t1("random_search", "100")[0].toFixed(0)}% for random search; replay and a lower learning rate did not help reliably, and cross-assay pretraining (${sgn(T.pre_minus_zero[0])} points vs. zero start) often hurt.`,
    `<b>Established search strategies extract considerably more value from the same representation.</b> At 50 measurements graph-adapted AdaLead is ${sgn(pf("adalead_graph")[0])} points [${pf("adalead_graph")[1].toFixed(1)}, ${pf("adalead_graph")[2].toFixed(1)}] above the fly learner, GP-BO ${sgn(pf("gpbo_ei")[0])} [${pf("gpbo_ei")[1].toFixed(1)}, ${pf("gpbo_ei")[2].toFixed(1)}], regularized evolution ${sgn(pf("regevo")[0])} [${pf("regevo")[1].toFixed(1)}, ${pf("regevo")[2].toFixed(1)}]. Zero-shot ESM-2 650M, which uses no observed fitness to build its ranking, reaches the top 1% in ${t1("zs_wt_650M")[0].toFixed(0)}% of assays within its first 50 measured candidates (wide interval: ${t1("zs_wt_650M")[1].toFixed(0)}–${t1("zs_wt_650M")[2].toFixed(0)}%).`,
    `<b>The bottleneck in this system is the search/decision rule, not representation quality.</b> The fly learner is level with a random walk on the same graph (random walk minus fly: ${sgn(pf("random_walk")[0])} points at 50 measurements, interval [${pf("random_walk")[1].toFixed(1)}, ${pf("random_walk")[2].toFixed(1)}]) and at or below plain random search (${sgn(pr("fly")[0])} [${pr("fly")[1].toFixed(1)}, ${pr("fly")[2].toFixed(1)}]). Random search itself is strong here, which is why the 50–200-measurement range is the informative regime.`,
  ].map((s) => `<li>${s}</li>`).join("");
}
