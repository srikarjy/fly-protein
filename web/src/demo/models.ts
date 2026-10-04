/** View models: recorded values -> the text and numbers the panels show. Pure functions of (engine, StepView, Reveal) so they can be
 *  tested without a DOM. A stage's value stays empty until its stage has been reached; once settled everything is shown. */
import type { Reveal } from "./choreo";
import { circuitShown, type Engine, type StepView } from "./state";

export const fmt = (x: number, d = 2) => (Math.abs(x) < 1e-12 ? (0).toFixed(d) : x.toFixed(d));
export const signed = (x: number, d = 2) => `${x >= 0 ? "+" : "−"}${Math.abs(x).toFixed(d)}`;
export const pctText = (p: number) => `${p.toFixed(p >= 99.95 ? 2 : 1)}th percentile`;
export const nf = new Intl.NumberFormat("en-US");

export type StageState = "pending" | "active" | "done";
export interface Stage { id: string; title: string; state: StageState; value: string | null; detail: string | null }

const stateOf = (r: number): StageState => (r >= 1 ? "done" : r > 0 ? "active" : "pending");

export function kcSummary(engine: Engine, v: StepView) {
  const c = circuitShown(v);
  const total = engine.circ.meta.n_kc;
  const pct = (100 * c.active.length) / total;
  return { active: c.active.length, total, pct, text: `${nf.format(c.active.length)} / ${nf.format(total)} KCs active (${pct.toFixed(pct % 1 ? 1 : 0)}%)` };
}

export function pipelineModel(engine: Engine, v: StepView, r: Reveal): Stage[] {
  const f = v.fly;
  const d = f.decision;
  const c = circuitShown(v);
  const name = (i: number) => engine.land.mutants[i];
  const lo = Math.min(...c.pnValues), hi = Math.max(...c.pnValues);
  const kc = kcSummary(engine, v);
  const start = v.k === 0;
  const stage = (id: string, title: string, reveal: number, value: string | null, detail: string | null): Stage => {
    const st = stateOf(reveal);
    return { id, title, state: st, value: st === "pending" ? null : value, detail: st === "pending" ? null : detail };
  };
  return [
    stage("input", "ESM-2 input", r.input, `${engine.circ.meta.n_pn} PN channels`, `${name(c.variant)} · values ${signed(lo, 3)} to ${signed(hi, 3)}`),
    stage("sparse", "Sparse KC code", r.sparse, kc.text, `drive cut-off ${fmt(c.cutoff[0], 3)}, strongest loser ${fmt(c.cutoff[1], 3)}`),
    stage("readout", "Learner readout", d ? r.readout : start ? 0 : 1, d ? `${d.candidates.length} neighbours scored` : "no decision this step",
      d ? `chosen score ${signed(d.scores[d.chosenIndex], 4)} · probability ${(100 * d.probs[d.chosenIndex]).toFixed(1)}%` : f.kind === "restart" ? "restart at a random unmeasured variant" : "initial placement"),
    stage("mutation", "Mutation", d ? r.move : start ? 0 : 1, d ? `${name(f.from)} → ${name(d.chosen)}` : name(f.at), d ? "neighbour in ESM-2 embedding space" : null),
    stage("fitness", "Measured fitness", start ? 0 : r.measure,
      d ? `${fmt(d.fitnessBefore)} → ${fmt(d.fitnessAfter)} (${signed(d.fitnessDelta)})` : `${fmt(engine.land.fitness[f.at])}`,
      d ? `${pctText(engine.land.pct[d.chosen])} · ${d.newMeasurement ? "newly measured" : "already measured, free revisit"}` : pctText(engine.land.pct[f.at])),
    stage("update", "Memory update", start ? 0 : r.update,
      d ? `reward ${fmt(d.reward, 3)} · δ ${fmt(d.delta, 3)}` : f.weightsReset ? "weights reset to zero" : `${f.nz} non-zero weights`,
      d ? `step a = ${d.step.toExponential(2)} · ${d.raised.length} cells +a, ${d.lowered.length} cells −a` : null),
  ];
}

export interface CandidateRow { variant: number; name: string; score: number; prob: number; chosen: boolean; measuredAt: number | null }

export function candidateModel(engine: Engine, v: StepView) {
  const d = v.fly.decision;
  if (!d) return null;
  const rows: CandidateRow[] = d.candidates.map((c, j) => ({ variant: c, name: engine.land.mutants[c], score: d.scores[j], prob: d.probs[j], chosen: j === d.chosenIndex, measuredAt: engine.measuredAt.fly.get(c) ?? null }));
  const allZero = d.scores.every((s) => s === 0);
  const top = Math.max(...d.probs);
  const uniform = 1 / d.probs.length;
  return { rows, allZero, topProb: top, uniform, steering: allZero ? "none" : top < 0.15 ? "weak" : "some" };
}

export function metricModel(engine: Engine, v: StepView) {
  const f = v.fly;
  const L = engine.land;
  const cur = f.at;
  return {
    measured: `${v.k} / ${v.budget}`,
    remaining: v.remaining,
    nowAt: `${L.mutants[cur]} · fitness ${fmt(L.fitness[cur])} (${pctText(L.pct[cur])})`,
    best: f.bestPct === null ? "—" : pctText(f.bestPct),
    delta: f.decision ? signed(f.decision.fitnessDelta) : "—",
    moves: f.moves,
    revisits: f.revisits,
    revisitShare: f.moves ? Math.round((100 * f.revisits) / f.moves) : 0,
    episode: f.episode,
    restarted: f.kind === "restart",
  };
}

export function boModel(engine: Engine, v: StepView) {
  const b = v.bo;
  const L = engine.land;
  const top = b.top.map((t, r) => ({ rank: r + 1, variant: t.i, name: L.mutants[t.i], ei: t.ei, mu: t.mu, sd: t.sd, chosen: t.i === b.chosen }));
  const maxEi = Math.max(1e-12, ...top.map((t) => t.ei));
  return {
    phase: b.kind === "start" ? "start" : b.kind === "init" ? "initial design (random, 10 measurements)" : `Bayesian step ${v.k - 10}`,
    last: b.chosen === null ? null : { name: L.mutants[b.chosen], fitness: b.f!, pct: L.pct[b.chosen] },
    best: b.bestPct === null ? "—" : pctText(b.bestPct),
    top: top.map((t) => ({ ...t, eiShare: t.ei / maxEi })),
    surrogate: b.nObs === null ? null : { nObs: b.nObs, ell: b.ell!, s2: b.s2! },
  };
}

/** Everything the hover inspector knows about one variant: recorded fitness and when each method measured it. */
export function inspectVariant(engine: Engine, i: number) {
  const L = engine.land;
  return { variant: i, name: L.mutants[i], fitness: L.fitness[i], pct: L.pct[i], flyStep: engine.measuredAt.fly.get(i) ?? null, boStep: engine.measuredAt.bo.get(i) ?? null };
}
