import type { Landscape } from "./landscape";
import { type Rng, softmaxSample } from "./rng";

/**
 * Tier 2: a mushroom-body-inspired learner with random PN->KC wiring (FlyHash codes).
 *
 * The fly does NOT see neighbor fitness. It only gets each neighbor's Kenyon cell
 * code (its "odor") and an output neuron that reads that code through plastic
 * KC->MBON weights w: value(v) = sum of w[c] over the cells c active for v.
 * Moves are sampled by softmax over the neighbors' values.
 *
 * After a move cur -> next the fly gets a dopamine-like signal, the reward
 * prediction error
 *     delta = (fitness[next] - fitness[cur]) - (value(next) - value(cur))
 * and only synapses of cells whose activity changed are updated:
 *     w[c] += lr * delta * (active_next[c] - active_cur[c])
 * which is gradient descent on delta^2 for a linear readout. The step is divided by the
 * number of cells involved (normalized LMS), so lr in (0, 2) is stable regardless of code size.
 */
export type LearningRule = "difference" | "absolute";

export class MushroomBodyLearner {
  w: Float32Array;
  constructor(public nKc: number, public lr = 0.05, public rule: LearningRule = "difference") {
    this.w = new Float32Array(nKc);
  }

  value(code: Int32Array): number {
    let s = 0;
    for (let i = 0; i < code.length; i++) s += this.w[code[i]];
    return s;
  }

  /** Pick the next variant among cur's neighbors using only their codes. */
  step(L: Landscape, cur: number, temperature: number, rng: Rng): number {
    const vals = new Float64Array(L.k);
    for (let j = 0; j < L.k; j++) vals[j] = this.value(L.codes[L.nbrs[cur * L.k + j]]);
    return L.nbrs[cur * L.k + softmaxSample(vals, temperature, rng)];
  }

  /** Reward-gated plasticity. Returns delta (the dopamine signal). */
  learn(L: Landscape, cur: number, next: number): number {
    if (this.rule === "absolute") {
      // dopamine = fitness of the arrival variant minus what the output neuron expected
      const delta = L.fitness[next] - this.value(L.codes[next]);
      const a = (this.lr * delta) / L.codes[next].length;
      for (const c of L.codes[next]) this.w[c] += a;
      return delta;
    }
    const reward = L.fitness[next] - L.fitness[cur];
    const delta = reward - (this.value(L.codes[next]) - this.value(L.codes[cur]));
    const a = (this.lr * delta) / (L.codes[next].length + L.codes[cur].length);
    for (const c of L.codes[next]) this.w[c] += a;
    for (const c of L.codes[cur]) this.w[c] -= a;
    return delta;
  }
}
