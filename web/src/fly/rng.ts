/** Small seeded PRNG (mulberry32) so runs are reproducible and shareable. */
export type Rng = () => number;

export function mulberry32(seed: number): Rng {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Sample an index with probability proportional to softmax(values / temperature). */
export function softmaxSample(values: ArrayLike<number>, temperature: number, rng: Rng): number {
  const n = values.length;
  let max = -Infinity;
  for (let i = 0; i < n; i++) if (values[i] > max) max = values[i];
  const w = new Float64Array(n);
  let sum = 0;
  for (let i = 0; i < n; i++) {
    w[i] = Math.exp((values[i] - max) / temperature);
    sum += w[i];
  }
  let r = rng() * sum;
  for (let i = 0; i < n; i++) {
    r -= w[i];
    if (r <= 0) return i;
  }
  return n - 1;
}
