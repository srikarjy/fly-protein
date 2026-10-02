import type { Landscape } from "./landscape";
import { type Rng, softmaxSample } from "./rng";

/**
 * Tier 1 (baseline, no brain model): look at the true fitness of every neighbor
 * and step to one with probability softmax(fitness / temperature).
 * Low temperature is greedy, high temperature is a random walk.
 */
export function chemotaxisStep(L: Landscape, cur: number, temperature: number, rng: Rng): number {
  const vals = new Float64Array(L.k);
  for (let j = 0; j < L.k; j++) vals[j] = L.fitness[L.nbrs[cur * L.k + j]];
  return L.nbrs[cur * L.k + softmaxSample(vals, temperature, rng)];
}
