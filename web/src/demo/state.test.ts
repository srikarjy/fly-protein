import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import type { Circuit, Landscape, Trace } from "../lib/types";
import { SETTLED, flyPose, revealAt, specOf, stepDuration } from "./choreo";
import { boModel, candidateModel, kcSummary, metricModel, pipelineModel } from "./models";
import { createEngine, circuitShown, type StepView } from "./state";
import { ReplayStore, deepLinkFor, parseDeepLink } from "./store";

const dir = new URL("../../public/data/", import.meta.url);
const read = <T>(f: string): T => JSON.parse(readFileSync(new URL(f, dir), "utf8"));
const trace = read<Trace>("demo_trace.json");
const land = read<Landscape>("demo_landscape.json");
const circ = read<Circuit>("demo_circuit.json");
const B = trace.budget;
const ev = trace.fly.events;

/** Independent oracle straight from the raw recorded events (does not use the engine's own bookkeeping). */
const lastEventAt = (k: number) => ev.filter((e) => e.q <= k).at(-1)!;
const flyAtOracle = (k: number) => { if (k === 0) return trace.start; const e = lastEventAt(k); return e.type === "move" ? e.nxt! : e.pos; };

const fresh = () => createEngine(trace, land, circ);

describe("replay step -> recorded protein variant", () => {
  const engine = fresh();
  it("the fly sits on the recorded variant at every measurement, and both methods' k-th measurement is the k-th logged query", () => {
    for (let k = 0; k <= B; k++) {
      const v = engine.view(k);
      expect(v.fly.at).toBe(flyAtOracle(k));
      expect(v.k).toBe(k);
      expect(v.remaining).toBe(B - k);
      if (k > 0) {
        expect(v.bo.chosen).toBe(trace.gpbo.events[k - 1].chosen);
        expect(v.fly.queried.length).toBe(k);
        expect(v.bo.queried.length).toBe(k);
        expect(v.fly.at).toBe(lastEventAt(k).type === "move" ? lastEventAt(k).nxt : lastEventAt(k).pos);
      }
    }
  });
  it("the recorded path of a step is a chain of recorded variants that ends where the fly settles", () => {
    for (let k = 1; k <= B; k++) {
      const v = engine.view(k).fly;
      let at = v.prevAt;
      for (const h of v.hops) { expect(h.from).toBe(at); at = h.to; }
      expect(at).toBe(v.at);
      expect(v.hops[0].newMeasurement).toBe(true); // a step is one budget-spending measurement
      expect(v.hops.slice(1).every((h) => h.kind === "move" && !h.newMeasurement)).toBe(true);
    }
    expect(engine.view(0).fly.at).toBe(trace.start);
    expect(engine.view(1).fly.prevAt).toBe(trace.start);
  });
});

describe("replay step -> recorded Kenyon-cell activity", () => {
  const engine = fresh();
  it("the shown circuit has exactly the recorded active set, 100 of 2000 (5%), with the recorded drive", () => {
    for (let k = 0; k <= B; k++) {
      const v = engine.view(k);
      const c = circuitShown(v);
      expect(c.active.length).toBe(trace.ka);
      expect([...c.active]).toEqual(trace.fly.codes[String(c.variant)]);
      expect([...c.drive]).toEqual(circ.drive[String(c.variant)]);
      expect(c.variant).toBe(v.fly.decision ? v.fly.from : v.fly.at);
      expect(kcSummary(engine, v).text).toBe("100 / 2,000 KCs active (5%)");
      expect(Math.min(...c.drive)).toBeCloseTo(c.cutoff[0], 4);
      expect(c.cutoff[0]).toBeGreaterThanOrEqual(c.cutoff[1]);
    }
  });
});

describe("replay step -> recorded learner state", () => {
  const engine = fresh();
  it("scores, probabilities, reward, prediction error, step size and weight statistics equal the recorded event", () => {
    let checked = 0;
    for (let k = 1; k <= B; k++) {
      const v = engine.view(k);
      const e = lastEventAt(k);
      expect(v.fly.nz).toBe(e.nz);
      expect(v.fly.wabs).toBe(e.wabs);
      const d = v.fly.decision;
      if (e.type !== "move") { expect(d).toBeNull(); continue; }
      expect(d).not.toBeNull();
      expect([...d!.scores]).toEqual(e.scores);
      expect(d!.reward).toBe(e.reward);
      expect(d!.delta).toBe(e.delta);
      expect(d!.step).toBe(e.a);
      expect(d!.chosen).toBe(e.nxt);
      expect(d!.candidates[d!.chosenIndex]).toBe(e.nxt);
      expect(d!.predictedGain).toBeCloseTo(e.reward! - e.delta!, 12);
      expect(d!.probs.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 4);
      expect(d!.fitnessAfter - d!.fitnessBefore).toBeCloseTo(land.fitness[e.nxt!] - land.fitness[e.pos], 12);
      checked++;
    }
    expect(checked).toBeGreaterThan(150); // most steps end in a learner decision; the rest are the initial placement and restarts
  });
  it("the update the page animates is exactly the recorded one: +a on the cells only the new code has, -a on those only the old code has", () => {
    for (let k = 1; k <= B; k++) {
      const v = engine.view(k).fly;
      const d = v.decision;
      const sumAbs = v.weightsAfter.reduce((s, x) => s + Math.abs(x), 0);
      expect(Math.abs(sumAbs - v.wabs)).toBeLessThan(1e-9 * Math.max(1, v.wabs));
      expect(v.weightsAfter.filter((x) => x !== 0).length).toBe(v.nz);
      if (!d) continue;
      const diff = v.weightsAfter.map((x, i) => x - v.weightsBefore[i]);
      for (const c of d.raised) expect(diff[c]).toBeCloseTo(d.step, 12);
      for (const c of d.lowered) expect(diff[c]).toBeCloseTo(-d.step, 12);
      const touched = new Set([...d.raised, ...d.lowered]);
      diff.forEach((x, i) => { if (!touched.has(i)) expect(Math.abs(x)).toBeLessThan(1e-12); });
    }
  });
});

describe("one authoritative step: controls, panels and the URL hash stay in sync", () => {
  it("scrubbing updates every subscribed panel to the same step and the same view object", () => {
    const store = new ReplayStore(fresh());
    const panels = ["landscape", "fly", "circuit", "learner", "bo", "metrics", "timeline"].map((name) => ({ name, k: -1, view: null as StepView | null }));
    panels.forEach((p) => store.subscribe((c) => { p.k = c.k; p.view = c.view; }));
    for (const k of [90, 17, 200, 3, 0, 150]) {
      store.setStep(k, "scrub");
      for (const p of panels) { expect(p.k).toBe(k); expect(p.view).toBe(store.engine.view(k)); expect(p.view).toBe(store.view); }
      // every view model derived from the step reports the same step
      const v = store.view;
      expect(metricModel(store.engine, v).measured).toBe(`${k} / ${B}`);
      expect(boModel(store.engine, v).last?.name ?? null).toBe(k ? land.mutants[trace.gpbo.events[k - 1].chosen] : null);
      expect(v.fly.queried.length).toBe(k);
      expect(v.bo.queried.length).toBe(k);
    }
  });

  it("#demo-N loads that step, clamps to the budget and ignores other hashes", () => {
    expect(parseDeepLink("#demo-90", B)).toBe(90);
    expect(parseDeepLink("#demo-999", B)).toBe(B);
    expect(parseDeepLink("#demo-0", B)).toBe(0);
    expect(parseDeepLink("#demo", B)).toBeNull();
    expect(parseDeepLink("#results", B)).toBeNull();
    expect(parseDeepLink("#demo-9x", B)).toBeNull();
    expect(deepLinkFor(90)).toBe("#demo-90");
    expect(parseDeepLink(deepLinkFor(143), B)).toBe(143);
    const store = new ReplayStore(fresh());
    store.setStep(parseDeepLink("#demo-90", B)!, "deeplink");
    expect(store.k).toBe(90);
    expect(store.view.fly.at).toBe(flyAtOracle(90));
    expect(store.view.fly.queried.length).toBe(90);
  });

  it("Next/Previous never desynchronize: a long walk ends in exactly the state a fresh engine gives for that step", () => {
    const store = new ReplayStore(fresh());
    let want = 0;
    const seen: number[] = [];
    store.subscribe((c) => seen.push(c.k));
    let seed = 7;
    const rnd = () => ((seed = (seed * 1103515245 + 12345) & 0x7fffffff) / 0x7fffffff);
    for (let i = 0; i < 400; i++) {
      const forward = rnd() < 0.6;
      const moved = forward ? store.next() : store.prev();
      const target = Math.max(0, Math.min(B, want + (forward ? 1 : -1)));
      expect(moved).toBe(target !== want);
      want = target;
      expect(store.k).toBe(want);
      expect(store.view).toBe(store.engine.view(want));
    }
    expect(seen.length).toBeGreaterThan(300);
    for (const k of [0, 1, 2, want, 90, B]) {
      store.setStep(k, "scrub");
      expect(JSON.stringify(store.view)).toBe(JSON.stringify(fresh().view(k))); // no state left over from the walk (weights, hops, trail)
    }
  });

  it("Reset returns to step 0 and the untouched start state", () => {
    const store = new ReplayStore(fresh());
    store.setStep(77, "scrub");
    let src = "";
    store.subscribe((c) => (src = c.source));
    expect(store.reset()).toBe(true);
    expect(src).toBe("reset");
    const v = store.view;
    expect(v.k).toBe(0);
    expect(v.fly.kind).toBe("start");
    expect(v.fly.at).toBe(trace.start);
    expect(v.fly.queried).toEqual([]);
    expect(v.bo.queried).toEqual([]);
    expect(v.fly.nz).toBe(0);
    expect(v.fly.weightsAfter.every((x) => x === 0)).toBe(true);
    expect(v.fly.decision).toBeNull();
    expect(store.reset()).toBe(false); // already at 0
  });
});

describe("animation state never replaces recorded scientific values", () => {
  const engine = fresh();
  const times = (v: StepView) => { const spec = specOf(v); const T = stepDuration(spec); return Array.from({ length: 41 }, (_, i) => (T * i) / 40); };

  it("views are immutable and sweeping every transition's animation leaves them byte-for-byte unchanged", () => {
    for (const k of [1, 2, 3, 12, 43, 90, 133, 170, B]) {
      const v = engine.view(k);
      expect(Object.isFrozen(v) && Object.isFrozen(v.fly) && Object.isFrozen(v.fly.hops)).toBe(true);
      expect(() => { (v.fly as { at: number }).at = -1; }).toThrow(TypeError);
      const before = JSON.stringify(v);
      for (const t of times(v)) { const r = revealAt(t, specOf(v)); flyPose(v, r); pipelineModel(engine, v, r); }
      expect(JSON.stringify(v)).toBe(before);
    }
  });

  it("reveals only ever grow from 0 to 1, in stage order, and settle exactly on the recorded state", () => {
    for (const k of [1, 2, 44, 90, B]) {
      const v = engine.view(k);
      const spec = specOf(v);
      let prev = revealAt(0, spec);
      for (const t of times(v)) {
        const r = revealAt(t, spec);
        for (const key of ["arrive", "input", "sparse", "readout", "move", "measure", "update", "advance"] as const) {
          expect(r[key]).toBeGreaterThanOrEqual(prev[key] - 1e-12);
          expect(r[key]).toBeGreaterThanOrEqual(0);
          expect(r[key]).toBeLessThanOrEqual(1);
        }
        if (r.sparse > 0) expect(r.input).toBe(1);
        if (r.move > 0) expect(r.readout).toBe(1);
        if (r.update > 0) expect(r.measure).toBe(1);
        prev = r;
      }
      expect(revealAt(stepDuration(spec), spec)).toBe(SETTLED);
      const pose = flyPose(v, SETTLED);
      expect(pose.to).toBe(v.fly.at); // exactly the recorded variant, no interpolation residue
      expect(pose.u).toBe(1);
    }
  });

  it("the fly's travelled path only uses recorded variants, starting where the previous step settled", () => {
    for (const k of [1, 44, 90, 140, B]) {
      const v = engine.view(k);
      const spec = specOf(v);
      const allowed = new Set([v.fly.prevAt, ...v.fly.hops.map((h) => h.to)]);
      const first = flyPose(v, revealAt(0, spec));
      expect(first.from).toBe(v.fly.prevAt);
      expect(first.u).toBe(0);
      for (const t of times(v)) {
        const p = flyPose(v, revealAt(t, spec));
        expect(allowed.has(p.from) && allowed.has(p.to)).toBe(true);
        expect(p.u).toBeGreaterThanOrEqual(0);
        expect(p.u).toBeLessThanOrEqual(1);
      }
    }
  });

  it("the pipeline shows a stage's values only once that stage is reached, and everything once settled", () => {
    const v = engine.view(90);
    const spec = specOf(v);
    const early = pipelineModel(engine, v, revealAt(0, spec));
    expect(early.every((s) => s.state === "pending" && s.value === null)).toBe(true);
    const mid = pipelineModel(engine, v, revealAt(stepDuration(spec) * 0.5, spec));
    expect(mid.some((s) => s.state === "done") && mid.some((s) => s.state === "pending")).toBe(true);
    const done = pipelineModel(engine, v, SETTLED);
    expect(done.every((s) => s.state === "done" && s.value !== null)).toBe(true);
    expect(done[1].value).toBe("100 / 2,000 KCs active (5%)");
    const d = v.fly.decision!;
    expect(done[5].value).toContain(d.reward.toFixed(3));
    expect(candidateModel(engine, v)!.rows.filter((r) => r.chosen).length).toBe(1);
  });
});
