/** DOM panels (text, numbers, the candidate readout, the pipeline stepper, GP-BO's ranking). They render the view models of the
 *  current StepView and the current Reveal; nothing here holds scientific state of its own. */
import { esc } from "../lib/charts";
import { $ } from "../lib/dom";
import type { Reveal } from "./choreo";
import { boModel, candidateModel, fmt, kcSummary, metricModel, nf, pctText, pipelineModel, signed } from "./models";
import type { Engine, StepView } from "./state";

const chips = (xs: string[][]) => xs.map(([a, b]) => `<span class="chip"><em>${esc(a)}</em> ${esc(b)}</span>`).join("");

export class Panels {
  private sig = "";
  private wStat: HTMLElement | null = null;

  constructor(private engine: Engine) {}

  /** one sentence describing the settled step, for the live region and the canvas labels */
  static describe(engine: Engine, v: StepView): string {
    if (v.k === 0) return `Measurement 0 of ${v.budget}. Both methods start at the same variant, ${engine.land.mutants[v.start]}.`;
    const m = metricModel(engine, v);
    const b = boModel(engine, v);
    const d = v.fly.decision;
    return `Measurement ${v.k} of ${v.budget}. Fly learner ${d ? `moved from ${engine.land.mutants[v.fly.from]} to ${engine.land.mutants[d.chosen]}, fitness ${signed(d.fitnessDelta)}` : "was placed at a variant"}; now at ${m.nowAt}; best so far ${m.best}. GP-BO measured ${b.last!.name}, fitness ${fmt(b.last!.fitness)}; best so far ${b.best}.`;
  }

  update(v: StepView, r: Reveal) {
    const sig = `${v.k}|${r.phase}|${r.settled}`;
    if (sig === this.sig) return;
    this.sig = sig;
    const E = this.engine;
    const m = metricModel(E, v);
    const B = v.budget;

    $("#d-budget-text").textContent = `${v.k} / ${B} measured · ${B - v.k} remaining`;
    $("#d-budget-bar").style.width = `${(100 * v.k) / B}%`;

    $("#fly-chips").innerHTML = v.k === 0
      ? chips([["measured", `0 / ${B}`], ["start", "shared starting variant"]])
      : chips([["best found", m.best], ["now at", m.nowAt], ["moves", `${m.moves} · ${m.revisits} (${m.revisitShare}%) onto already-measured variants`], ["episode", `${m.episode}${m.restarted ? " · restarted, weights reset" : ""}`]]);

    // pipeline
    $("#pipeline").innerHTML = pipelineModel(E, v, r).map((s, i) => `<li class="stage ${s.state}" data-stage="${s.id}"><span class="st-n">${i + 1}</span><b>${esc(s.title)}</b><span class="st-v">${s.value === null ? "—" : esc(s.value)}</span><span class="st-d">${s.detail === null ? "&nbsp;" : esc(s.detail)}</span></li>`).join("");

    // candidate readout
    const cm = candidateModel(E, v);
    const bars = $("#fly-bars");
    if (cm) {
      const mx = Math.max(0.02, ...cm.rows.map((x) => Math.abs(x.score)));
      bars.innerHTML = `<div class="barrow" style="opacity:${0.25 + 0.75 * r.readout}">${cm.rows.map((x, j) => `<div class="bar ${x.chosen ? "chosen" : ""}" title="neighbour ${j + 1}: ${esc(x.name)} · score ${x.score.toFixed(4)} · choice probability ${(100 * x.prob).toFixed(1)}%${x.measuredAt ? ` · measured at step ${x.measuredAt}` : " · not yet measured by this run"}"><span class="b" style="height:${Math.max(2, (Math.abs(x.score) / mx) * 44)}px;${x.score < 0 ? "background:var(--c-bo)" : ""}"></span><span class="pp">${(100 * x.prob).toFixed(0)}</span></div>`).join("")}</div>
        <p class="muted small">Bars: learner score of each of the 15 neighbours (sum of learned Kenyon-cell weights over that neighbour's code). Numbers: probability the move rule gave each neighbour (softmax, T = ${E.circ.meta.temperature}), as recorded. ${r.readout >= 1 || r.settled ? `Chosen: <b>${esc(E.land.mutants[v.fly.decision!.chosen])}</b>, ${v.fly.decision!.newMeasurement ? "newly measured" : "already measured (free revisit)"}. ${cm.allZero ? "All scores are 0 (no learned weights yet), so the move is random." : `The most likely neighbour had probability ${(100 * cm.topProb).toFixed(0)}% (uniform would be ${(100 * cm.uniform).toFixed(1)}%): the learned scores steer the move ${cm.steering === "weak" ? "only weakly" : "somewhat"}.`}` : ""}</p>`;
    } else {
      bars.innerHTML = `<p class="muted">${v.k === 0 ? `Press Start. Both methods begin at the same variant and spend the same budget of ${B} measurements.` : v.fly.kind === "restart" ? "New episode: restarted at a fresh random variant" + (v.fly.weightsReset ? ", learned weights reset to zero." : ".") : "First measurement: the shared starting variant."}</p>`;
    }

    // memory
    const d = v.fly.decision;
    const reached = r.settled || r.update > 0; // the update's numbers appear only when its stage is reached
    $("#fly-memory").innerHTML = v.k === 0 || !reached ? "" : d
      ? `<b>Memory update</b> (recorded): reward ${fmt(d.reward, 3)} · prediction error δ ${fmt(d.delta, 3)} (the learner predicted a gain of ${signed(d.predictedGain, 3)}) · weight step a = ${d.step.toExponential(2)}: ${d.raised.length} cells raised (+a), ${d.lowered.length} lowered (−a). Weights now: ${v.fly.nz} non-zero of ${nf.format(E.circ.meta.n_kc)}, Σ|w| = <span id="w-stat">${v.fly.wabs.toFixed(4)}</span>.`
      : `<b>Memory</b>: ${v.fly.nz} non-zero weights of ${nf.format(E.circ.meta.n_kc)}, Σ|w| = <span id="w-stat">${v.fly.wabs.toFixed(4)}</span>.`;
    this.wStat = document.getElementById("w-stat");

    // GP-BO
    const bm = boModel(E, v);
    $("#bo-chips").innerHTML = v.k === 0
      ? chips([["measured", `0 / ${B}`], ["start", "shared starting variant"]])
      : chips([["best found", bm.best], ["last measured", `${bm.last!.name} · fitness ${fmt(bm.last!.fitness)} (${pctText(bm.last!.pct)})`], ["phase", bm.phase]]);
    if (bm.top.length) {
      $("#bo-top").innerHTML = `<table class="botable"><thead><tr><th>#</th><th>variant</th><th>expected improvement</th><th>predicted fitness</th></tr></thead><tbody>${bm.top.map((t) => `<tr class="${t.chosen ? "chosen" : ""}"><td>${t.rank}</td><td>${esc(t.name)}</td><td><span class="eibar"><i style="width:${(100 * t.eiShare).toFixed(0)}%"></i></span>${t.ei.toFixed(3)}</td><td>${t.mu.toFixed(2)} ± ${t.sd.toFixed(2)}</td></tr>`).join("")}</tbody></table>`;
      const top = bm.top[0];
      $("#bo-note").innerHTML = `<b>Surrogate</b>: GP on ${bm.surrogate!.nObs} measured variants, kernel length-scale ${bm.surrogate!.ell} (× median pairwise distance), noise ${bm.surrogate!.s2}. Measured: <b>${esc(bm.last!.name)}</b> → fitness ${fmt(bm.last!.fitness)} (prediction was ${top.mu.toFixed(2)} ± ${top.sd.toFixed(2)}).`;
    } else {
      $("#bo-top").innerHTML = `<p class="muted">${v.k === 0 ? "Before the first measurement there is nothing to rank." : "The first 10 measurements are a random initial design (the shared start plus 9 random variants); the Gaussian process is fitted afterwards."}</p>`;
      $("#bo-note").innerHTML = "";
    }

    const kc = kcSummary(E, v);
    $("#kc-summary").textContent = kc.text;
  }

  /** per-frame work: the weight total counts from its recorded value before the update to the recorded value after it */
  frame(v: StepView, r: Reveal) {
    if (!this.wStat || !v.fly.decision) return;
    this.wStat.textContent = (v.fly.wabsBefore + (v.fly.wabs - v.fly.wabsBefore) * r.update).toFixed(4);
  }
}
