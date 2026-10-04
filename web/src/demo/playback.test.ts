import { readFileSync } from "node:fs";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { Circuit, Landscape, Trace } from "../lib/types";
import { Playback, type Scheduler } from "./playback";
import { createEngine } from "./state";
import { ReplayStore } from "./store";

const dir = new URL("../../public/data/", import.meta.url);
const read = <T>(f: string): T => JSON.parse(readFileSync(new URL(f, dir), "utf8"));
const engine = createEngine(read<Trace>("demo_trace.json"), read<Landscape>("demo_landscape.json"), read<Circuit>("demo_circuit.json"));

const sched: Scheduler = { setTimeout: (fn, ms) => setTimeout(fn, ms) as unknown as number, clearTimeout: (id) => clearTimeout(id) };

describe("playback", () => {
  beforeEach(() => vi.useFakeTimers());
  afterEach(() => vi.useRealTimers());

  it("advances one step per settle signal and never on its own", () => {
    const store = new ReplayStore(engine);
    const p = new Playback(store, () => {}, sched);
    p.play();
    expect(store.k).toBe(1); // first step immediately
    vi.advanceTimersByTime(10_000);
    expect(store.k).toBe(1); // waits for the page to report the step settled
    p.settled(300);
    vi.advanceTimersByTime(299);
    expect(store.k).toBe(1);
    vi.advanceTimersByTime(2);
    expect(store.k).toBe(2);
  });

  it("pause cancels a pending step and stays paused", () => {
    const store = new ReplayStore(engine);
    const p = new Playback(store, () => {}, sched);
    p.play();
    p.settled(500);
    p.pause();
    vi.advanceTimersByTime(5000);
    expect(store.k).toBe(1);
    expect(p.playing).toBe(false);
    p.settled(10); // late settle signal after a pause is ignored
    vi.advanceTimersByTime(50);
    expect(store.k).toBe(1);
  });

  it("stops at the end of the budget, and Start from the end restarts at step 0", () => {
    const store = new ReplayStore(engine);
    store.setStep(engine.budget - 1, "scrub");
    const p = new Playback(store, () => {}, sched);
    p.play();
    expect(store.k).toBe(engine.budget);
    p.settled(10);
    expect(p.playing).toBe(false);
    p.play();
    expect(store.k).toBe(1); // reset to 0, then the first step
    expect(p.playing).toBe(true);
  });
});
