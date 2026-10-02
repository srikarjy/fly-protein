import { chemotaxisStep } from "./chemotaxis";
import { MushroomBodyLearner, type LearningRule } from "./learner";
import type { Landscape } from "./landscape";
import { type Rng } from "./rng";

export type Mode = "random" | "chemotaxis" | "mushroom";

export interface AgentParams {
  /** softmax temperature over neighbors */
  temperature: number;
  /** learning rate of the KC->MBON synapses (mushroom mode) */
  lr: number;
  /** "difference": learn fitness change along the path; "absolute": learn each variant's fitness */
  rule?: LearningRule;
}

export const DEFAULTS: Record<Mode, AgentParams> = {
  random: { temperature: 1e6, lr: 0 },
  chemotaxis: { temperature: 0.1, lr: 0 },
  mushroom: { temperature: 0.1, lr: 0.1, rule: "difference" },
};

/** One fly: a position, a history, and (for tier 2) its learned synapses. */
export class Fly {
  pos: number;
  path: number[];
  best: number;
  learner: MushroomBodyLearner | null;
  lastDelta = 0;

  constructor(public L: Landscape, public mode: Mode, start: number, public params: AgentParams, learner?: MushroomBodyLearner) {
    this.pos = start;
    this.path = [start];
    this.best = L.fitness[start];
    this.learner = mode === "mushroom" ? (learner ?? new MushroomBodyLearner(L.nKc, params.lr, params.rule)) : null;
  }

  step(rng: Rng): number {
    const { L, params } = this;
    const next =
      this.mode === "mushroom"
        ? this.learner!.step(L, this.pos, params.temperature, rng)
        : chemotaxisStep(L, this.pos, params.temperature, rng);
    if (this.learner) this.lastDelta = this.learner.learn(L, this.pos, next);
    this.pos = next;
    this.path.push(next);
    if (L.fitness[next] > this.best) this.best = L.fitness[next];
    return next;
  }
}
