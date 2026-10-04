/** The single source of truth for what the demo shows: an immutable StepView per measurement count k, derived only from the
 *  recorded trace, landscape and circuit files. Nothing in here searches, scores or learns; it looks recorded values up
 *  (and re-applies the recorded weight increments through FlyReplay). Animation never writes into a StepView. */
import type { BoCandidate, Circuit, Landscape, Trace } from "../lib/types";
import { BoReplay, FlyReplay } from "./replay";

export type StepKind = "start" | "init" | "restart" | "move";

/** One recorded fly event of a step: a move (possibly onto an already-measured variant), the initial placement or a restart. */
export interface Hop { eventIndex: number; kind: "init" | "restart" | "move"; from: number; to: number; newMeasurement: boolean }

/** The PN->KC state of one variant, as recorded. */
export interface CircuitState {
  variant: number;
  /** 8-bit PN values as exported (each variant scaled to its own largest channel) */
  pn: readonly number[];
  pnMax: number;
  /** the same values in embedding units */
  pnValues: readonly number[];
  /** active Kenyon cells (ascending) and their exact input drive, same order */
  active: readonly number[];
  drive: readonly number[];
  /** [weakest active drive, strongest inactive drive] */
  cutoff: readonly [number, number];
}

/** The learner's final recorded move of the step and its reward-prediction-error update. */
export interface Decision {
  eventIndex: number;
  candidates: readonly number[];
  scores: readonly number[];
  probs: readonly number[];
  chosenIndex: number;
  chosen: number;
  newMeasurement: boolean;
  fitnessBefore: number;
  fitnessAfter: number;
  fitnessDelta: number;
  reward: number;
  /** reward minus the learner's predicted gain */
  delta: number;
  /** the learner's predicted fitness gain before the update (reward - delta) */
  predictedGain: number;
  /** the weight increment `a` actually applied: +a on `raised` cells, -a on `lowered` cells */
  step: number;
  raised: readonly number[];
  lowered: readonly number[];
}

export interface FlyView {
  kind: StepKind;
  hops: readonly Hop[];
  /** where the fly sat when the previous step settled */
  prevAt: number;
  /** the variant the step's final decision departs from */
  from: number;
  /** where the fly sits when the step has settled */
  at: number;
  decision: Decision | null;
  circuitFrom: CircuitState | null;
  circuitAt: CircuitState;
  /** Kenyon-cell weights immediately before and after the step's final event */
  weightsBefore: readonly number[];
  weightsAfter: readonly number[];
  nz: number;
  wabs: number;
  wabsBefore: number;
  /** the final event reset the learned weights (a restart without memory) */
  weightsReset: boolean;
  moves: number;
  revisits: number;
  episode: number;
  trail: readonly (readonly [number, number])[];
  queried: readonly number[];
  bestVariant: number | null;
  bestPct: number | null;
}

export interface BoView {
  kind: "start" | "init" | "bo";
  chosen: number | null;
  prevChosen: number | null;
  f: number | null;
  top: readonly BoCandidate[];
  nObs: number | null;
  ell: number | null;
  s2: number | null;
  queried: readonly number[];
  bestVariant: number | null;
  bestPct: number | null;
}

export interface StepView {
  k: number;
  budget: number;
  remaining: number;
  start: number;
  fly: FlyView;
  bo: BoView;
}

export interface Engine {
  trace: Trace;
  land: Landscape;
  circ: Circuit;
  budget: number;
  /** wiring[kc * perKc + s] = the s-th PN that Kenyon cell kc samples */
  wiring: Uint16Array;
  /** the variant measured by the k-th measurement (k = 1..), per method */
  queryOrder: { fly: number[]; bo: number[] };
  /** the measurement count at which each method first measured a variant */
  measuredAt: { fly: Map<number, number>; bo: Map<number, number> };
  view(k: number): StepView;
}

/** The circuit the brain view shows: the sparse code of the variant the learner decides from (its input), or of the placed variant when there is no decision. */
export const circuitShown = (v: StepView): CircuitState => v.fly.circuitFrom ?? v.fly.circuitAt;

export function decodeWiring(b64: string, count: number): Uint16Array {
  const bin = atob(b64);
  const out = new Uint16Array(count);
  for (let i = 0; i < count; i++) out[i] = bin.charCodeAt(2 * i) | (bin.charCodeAt(2 * i + 1) << 8);
  return out;
}

function freeze<T>(o: T): T {
  if (o && typeof o === "object" && !Object.isFrozen(o)) {
    Object.freeze(o);
    for (const v of Object.values(o as object)) freeze(v);
  }
  return o;
}

export function createEngine(trace: Trace, land: Landscape, circ: Circuit): Engine {
  const fly = new FlyReplay(trace);
  const bo = new BoReplay(trace.gpbo.events);
  const B = trace.budget;
  const ev = trace.fly.events;
  const wiring = decodeWiring(circ.wiring_b64, circ.meta.n_kc * circ.meta.per_kc);

  const measuredAt = { fly: new Map<number, number>(), bo: new Map<number, number>() };
  fly.queryOrder.forEach((v, i) => measuredAt.fly.set(v, i + 1));
  bo.queryOrder.forEach((v, i) => measuredAt.bo.set(v, i + 1));

  /** index (0-based, into the measurement order) of the best-percentile variant measured within the first k measurements */
  const bestIndex = (order: number[]) => {
    let b = -1;
    return order.map((v, i) => (b = b < 0 || land.pct[v] > land.pct[order[b]] ? i : b));
  };
  const flyBest = bestIndex(fly.queryOrder);
  const boBest = bestIndex(bo.queryOrder);

  const circuits = new Map<number, CircuitState>();
  function circuitOf(i: number): CircuitState {
    let c = circuits.get(i);
    if (!c) {
      const key = String(i);
      const pn = trace.fly.pn[key];
      const pnMax = circ.pn_max[key];
      c = { variant: i, pn, pnMax, pnValues: pn.map((q) => (q / 127) * pnMax), active: trace.fly.codes[key], drive: circ.drive[key], cutoff: circ.cutoff[key] };
      circuits.set(i, c);
    }
    return c;
  }

  const settledAt = (k: number) => {
    if (k <= 0) return trace.start;
    const e = ev[fly.eventsAfter(k) - 1];
    return e.type === "move" ? e.nxt! : e.pos;
  };

  function flyView(k: number): FlyView {
    const prevAt = settledAt(k - 1);
    if (k === 0) {
      fly.seekEvents(0);
      const w0 = Array.from(fly.w);
      return { kind: "start", hops: [], prevAt, from: prevAt, at: prevAt, decision: null, circuitFrom: null, circuitAt: circuitOf(prevAt), weightsBefore: w0, weightsAfter: w0, nz: 0, wabs: 0, wabsBefore: 0, weightsReset: false, moves: 0, revisits: 0, episode: 0, trail: [], queried: [], bestVariant: null, bestPct: null };
    }
    const n0 = fly.eventsAfter(k - 1);
    const n1 = fly.eventsAfter(k);
    const hops: Hop[] = [];
    let cursor = prevAt;
    for (let i = n0; i < n1; i++) {
      const e = ev[i];
      const to = e.type === "move" ? e.nxt! : e.pos;
      hops.push({ eventIndex: i, kind: e.type, from: cursor, to, newMeasurement: e.type !== "move" || !!e.new });
      cursor = to;
    }
    const last = ev[n1 - 1];
    fly.seekEvents(n1 - 1);
    const weightsBefore = Array.from(fly.w);
    fly.seekEvents(n1);
    const weightsAfter = Array.from(fly.w);

    let decision: Decision | null = null;
    if (last.type === "move") {
      const j = last.j!;
      const a = new Set(trace.fly.codes[String(last.nxt)]);
      const b = new Set(trace.fly.codes[String(last.pos)]);
      const before = land.fitness[last.pos];
      const after = land.fitness[last.nxt!];
      decision = {
        eventIndex: n1 - 1, candidates: trace.fly.nbrs[String(last.pos)], scores: last.scores!, probs: circ.probs[n1 - 1]!, chosenIndex: j, chosen: last.nxt!,
        newMeasurement: !!last.new, fitnessBefore: before, fitnessAfter: after, fitnessDelta: after - before, reward: last.reward!, delta: last.delta!,
        predictedGain: last.reward! - last.delta!, step: last.a!, raised: [...a].filter((c) => !b.has(c)), lowered: [...b].filter((c) => !a.has(c)),
      };
    }
    const done = ev.slice(0, n1);
    const moves = done.filter((e) => e.type === "move");
    const at = last.type === "move" ? last.nxt! : last.pos;
    const from = last.pos;
    const bi = flyBest[k - 1];
    return {
      kind: last.type, hops, prevAt, from, at, decision,
      circuitFrom: last.type === "move" ? circuitOf(from) : null, circuitAt: circuitOf(at),
      weightsBefore, weightsAfter, nz: last.nz, wabs: last.wabs, wabsBefore: n1 >= 2 ? ev[n1 - 2].wabs : 0, weightsReset: last.type !== "move" && !!last.reset,
      moves: moves.length, revisits: moves.filter((e) => !e.new).length, episode: done.filter((e) => e.type !== "move").length,
      trail: moves.slice(-24).map((e) => [e.pos, e.nxt!] as const), queried: fly.queryOrder.slice(0, k),
      bestVariant: fly.queryOrder[bi], bestPct: land.pct[fly.queryOrder[bi]],
    };
  }

  function boView(k: number): BoView {
    if (k === 0) return { kind: "start", chosen: null, prevChosen: null, f: null, top: [], nObs: null, ell: null, s2: null, queried: [], bestVariant: null, bestPct: null };
    const e = trace.gpbo.events[k - 1];
    const bi = boBest[k - 1];
    return {
      kind: e.type, chosen: e.chosen, prevChosen: k > 1 ? trace.gpbo.events[k - 2].chosen : null, f: e.f, top: e.top ?? [], nObs: e.n_obs ?? null, ell: e.ell ?? null, s2: e.s2 ?? null,
      queried: bo.queryOrder.slice(0, k), bestVariant: bo.queryOrder[bi], bestPct: land.pct[bo.queryOrder[bi]],
    };
  }

  const cache = new Map<number, StepView>();
  return {
    trace, land, circ, budget: B, wiring, measuredAt, queryOrder: { fly: fly.queryOrder, bo: bo.queryOrder },
    view(k: number): StepView {
      k = Math.max(0, Math.min(B, Math.round(k)));
      let v = cache.get(k);
      if (!v) {
        v = freeze({ k, budget: B, remaining: B - k, start: trace.start, fly: flyView(k), bo: boView(k) });
        cache.set(k, v);
        if (cache.size > 24) cache.delete(cache.keys().next().value!);
      }
      return v;
    },
  };
}
