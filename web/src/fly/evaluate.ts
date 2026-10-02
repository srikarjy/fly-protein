import { DEFAULTS, Fly, type AgentParams, type Mode } from "./agent";
import { MushroomBodyLearner } from "./learner";
import { type Landscape, quantile } from "./landscape";
import { mulberry32 } from "./rng";

export interface EvalRow {
  label: string;
  medianBest: number;
  /** fraction of starts that reached the top 10% within the step budget */
  reachRate: number;
  /** median first-hit step over ALL starts (misses count as never); null if fewer than half reached */
  medianStepsToTop: number | null;
}

export interface EvalOptions {
  starts?: number;
  steps?: number;
  seed?: number;
  topFrac?: number;
}

function median(a: number[]): number {
  return quantile(a, 0.5);
}

/** Same starts, same budget for every row. `carryOver` keeps the learner's synapses across starts. */
export function evaluate(
  L: Landscape,
  mode: Mode,
  label: string,
  params: AgentParams = DEFAULTS[mode],
  opts: EvalOptions & { carryOver?: boolean } = {},
): EvalRow {
  const { starts = 50, steps = 200, seed = 1, topFrac = 0.1 } = opts;
  const threshold = quantile(L.fitness, 1 - topFrac);
  const startRng = mulberry32(seed);
  const runRng = mulberry32(seed + 1000);
  const shared = opts.carryOver && mode === "mushroom" ? new MushroomBodyLearner(L.nKc, params.lr, params.rule) : undefined;
  const bests: number[] = [];
  const hits: number[] = [];
  for (let s = 0; s < starts; s++) {
    const start = Math.floor(startRng() * L.n);
    const fly = new Fly(L, mode, start, params, shared);
    let hit = L.fitness[start] >= threshold ? 0 : -1;
    for (let t = 1; t <= steps; t++) {
      fly.step(runRng);
      if (hit < 0 && L.fitness[fly.pos] >= threshold) hit = t;
    }
    bests.push(fly.best);
    hits.push(hit >= 0 ? hit : Infinity);
  }
  return {
    label,
    medianBest: median(bests),
    reachRate: hits.filter(Number.isFinite).length / starts,
    medianStepsToTop: Number.isFinite(median(hits)) ? median(hits) : null,
  };
}
