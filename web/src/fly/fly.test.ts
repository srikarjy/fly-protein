import { describe, expect, it } from "vitest";
import { Fly, DEFAULTS } from "./agent";
import { chemotaxisStep } from "./chemotaxis";
import { type Landscape, applyMutations, hemiView, parseLandscape, quantile } from "./landscape";
import { MushroomBodyLearner } from "./learner";
import { mulberry32, softmaxSample } from "./rng";

/** A ring of 20 variants: fitness peaks at index 10; neighbors are i-1 and i+1 (k=2). */
function ring(): Landscape {
  const n = 20;
  const variants = Array.from({ length: n }, (_, i) => ({
    m: `A${i + 1}G`,
    f: 1 - Math.abs(i - 10) / 10,
    x: i / n,
    y: 0,
    n: [(i + n - 1) % n, (i + 1) % n],
    c: [i, (i + 7) % 40, (i + 13) % 40],
  }));
  return parseLandscape({ meta: { k: 2, n_kc: 40, wt: "A".repeat(n), assay: "ring", embedding: "none" }, variants });
}

describe("softmaxSample", () => {
  it("is greedy at low temperature and uniform at high temperature", () => {
    const rng = mulberry32(1);
    const low = Array.from({ length: 200 }, () => softmaxSample([0, 1, 0.5], 0.001, rng));
    expect(low.every((i) => i === 1)).toBe(true);
    const high = Array.from({ length: 3000 }, () => softmaxSample([0, 1, 0.5], 1e6, rng));
    for (const i of [0, 1, 2]) expect(high.filter((x) => x === i).length).toBeGreaterThan(800);
  });
});

describe("chemotaxisStep", () => {
  it("moves to the fitter neighbor at low temperature", () => {
    const L = ring();
    expect(chemotaxisStep(L, 3, 0.001, mulberry32(2))).toBe(4);
    expect(chemotaxisStep(L, 15, 0.001, mulberry32(2))).toBe(14);
  });
});

describe("MushroomBodyLearner", () => {
  it("raises weights on cells active for a gain and lowers them for a loss", () => {
    const L = ring();
    const mb = new MushroomBodyLearner(40, 0.5);
    const delta = mb.learn(L, 3, 4); // fitness goes up by 0.1
    expect(delta).toBeCloseTo(0.1);
    for (const c of L.codes[4]) expect(mb.w[c]).toBeGreaterThan(0);
    for (const c of L.codes[3]) expect(mb.w[c]).toBeLessThan(0);
  });

  it("shrinks its own prediction error when the same transition repeats", () => {
    const L = ring();
    const mb = new MushroomBodyLearner(40, 0.3);
    const first = Math.abs(mb.learn(L, 3, 4));
    let last = first;
    for (let i = 0; i < 50; i++) last = Math.abs(mb.learn(L, 3, 4));
    expect(last).toBeLessThan(first * 0.1);
  });

  it("after training on a transition, scores the destination above the source", () => {
    const L = ring();
    const mb = new MushroomBodyLearner(40, 0.3);
    for (let i = 0; i < 100; i++) mb.learn(L, 3, 4);
    expect(mb.value(L.codes[4])).toBeGreaterThan(mb.value(L.codes[3]));
  });

  it("does not read fitness when choosing a step (weights alone decide)", () => {
    const L = ring();
    const mb = new MushroomBodyLearner(40, 0.3);
    for (let i = 0; i < 100; i++) mb.learn(L, 3, 4);
    // untrained weights are zero: all neighbors tie, so both moves occur
    const fresh = new MushroomBodyLearner(40);
    const rng = mulberry32(5);
    const moves = new Set(Array.from({ length: 50 }, () => fresh.step(L, 3, 0.1, rng)));
    expect(moves.size).toBe(2);
  });
});

describe("Fly", () => {
  it("chemotaxis climbs a smooth landscape", () => {
    const L = ring();
    const rng = mulberry32(3);
    const fly = new Fly(L, "chemotaxis", 0, { ...DEFAULTS.chemotaxis, temperature: 0.01 });
    for (let t = 0; t < 30; t++) fly.step(rng);
    expect(fly.best).toBeCloseTo(1, 5);
  });
  it("tracks path and best", () => {
    const L = ring();
    const fly = new Fly(L, "random", 5, DEFAULTS.random);
    fly.step(mulberry32(4));
    expect(fly.path.length).toBe(2);
    expect(fly.best).toBeGreaterThanOrEqual(L.fitness[5]);
  });
});

describe("landscape helpers", () => {
  it("quantile interpolates", () => expect(quantile([0, 10], 0.5)).toBe(5));
  it("applies ProteinGym mutations 1-indexed", () => {
    expect(applyMutations("ABCDE", "B2X")).toBe("AXCDE");
    expect(applyMutations("ABCDE", "A1Z:E5Y")).toBe("ZBCDY");
  });
});

describe("hemibrain mode", () => {
  function ringHemi(): Landscape {
    const n = 20;
    const variants = Array.from({ length: n }, (_, i) => ({
      m: `A${i + 1}G`,
      f: 1 - Math.abs(i - 10) / 10,
      x: i / n,
      y: 0,
      n: [(i + n - 1) % n, (i + 1) % n],
      c: [i, (i + 7) % 40],
      h: [i % 9, 9 + (i % 5)],
    }));
    return parseLandscape({ meta: { k: 2, n_kc: 40, wt: "A".repeat(n), assay: "ring", embedding: "none", hemi: { n_kc: 14 } }, variants });
  }

  it("hemiView swaps codes and nKc and keeps fitness", () => {
    const L = ringHemi();
    const V = hemiView(L);
    expect(V.nKc).toBe(14);
    expect(V.codes[3]).toEqual(Int32Array.from([3, 12]));
    expect(V.fitness).toBe(L.fitness);
  });

  it("throws a clear error without hemibrain codes", () => {
    expect(() => hemiView(ring())).toThrow(/hemi-only/);
  });

  it("a hemibrain fly learns on the 14-cell codes", () => {
    const L = ringHemi();
    const fly = new Fly(L, "hemibrain", 3, DEFAULTS.hemibrain);
    expect(fly.learner!.w.length).toBe(14);
    const rng = mulberry32(5);
    for (let i = 0; i < 30; i++) fly.step(rng);
    expect(fly.learner!.w.some((x) => x !== 0)).toBe(true);
  });
});
