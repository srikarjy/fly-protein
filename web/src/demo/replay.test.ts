import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import type { Landscape, Trace } from "../lib/types";
import { BoReplay, FlyReplay, sha256Int32, softmax, verifyFlyReplay } from "./replay";

const dir = new URL("../../public/data/", import.meta.url);
const trace: Trace = JSON.parse(readFileSync(new URL("demo_trace.json", dir), "utf8"));
const land: Landscape = JSON.parse(readFileSync(new URL("demo_landscape.json", dir), "utf8"));

describe("recorded demo trace", () => {
  it("replaying the logged weight increments reproduces the logged weight statistics at every measurement", () => {
    const v = verifyFlyReplay(trace);
    expect(v.checked).toBe(trace.budget);
    expect(v.ok).toBe(true);
  });

  it("the k-th measurement is the k-th entry of the query log, and the logs hash to the benchmark's hashes", async () => {
    const fly = new FlyReplay(trace);
    const bo = new BoReplay(trace.gpbo.events);
    expect(fly.queryOrder.length).toBe(trace.budget);
    expect(bo.queryOrder.length).toBe(trace.budget);
    expect(new Set(fly.queryOrder).size).toBe(trace.budget); // every measurement is of a new variant
    expect(new Set(bo.queryOrder).size).toBe(trace.budget);
    expect(fly.queryOrder[0]).toBe(trace.start);
    expect(bo.queryOrder[0]).toBe(trace.start); // same starting variant for both methods
    expect(await sha256Int32(fly.queryOrder)).toBe(trace.verified.fly_log_sha256);
    expect(await sha256Int32(bo.queryOrder)).toBe(trace.verified.gpbo_log_sha256);
  });

  it("recorded fitness matches the landscape and chosen candidates are among the recorded neighbours", () => {
    for (const e of trace.fly.events) {
      const v = e.type === "move" ? e.nxt! : e.pos;
      expect(Math.abs(land.fitness[v] - e.f)).toBeLessThan(5e-4);
      if (e.type === "move") {
        expect(trace.fly.nbrs[String(e.pos)][e.j!]).toBe(e.nxt);
        expect(e.scores!.length).toBe(15);
      }
    }
    for (const e of trace.gpbo.events) expect(Math.abs(land.fitness[e.chosen] - e.f)).toBeLessThan(5e-4);
  });

  it("seeking backwards gives the same state as seeking forwards", () => {
    const r = new FlyReplay(trace);
    r.seek(120);
    const w120 = Float64Array.from(r.w);
    r.seek(40);
    r.seek(120);
    expect([...r.w]).toEqual([...w120]);
  });

  it("softmax sums to one and prefers the larger score", () => {
    const p = softmax([0, 1, 0.5], 0.1);
    expect(p.reduce((a, b) => a + b)).toBeCloseTo(1);
    expect(p[1]).toBeGreaterThan(p[2]);
  });
});
