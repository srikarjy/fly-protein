/** Pure replay of recorded search state. Nothing here simulates a search: it only re-applies the exact
 *  weight increments and replays the exact choices that the Python implementation logged. */
import type { BoEvent, FlyEvent, Trace } from "../lib/types";

export interface FlyState {
  tick: number;
  last: FlyEvent;
  lastMove: FlyEvent | null;
  moves: number;
  revisits: number;
  revisitsSinceNew: number;
  episode: number;
  queried: number[];
  recent: FlyEvent[];
}

export class FlyReplay {
  w: Float64Array;
  private applied = 0;
  private tick = 0;
  /** the variant measured by the k-th query, k = 1..; derived from the event list */
  readonly queryOrder: number[] = [];
  constructor(readonly trace: Trace) {
    this.w = new Float64Array(trace.n_kc);
    for (const e of trace.fly.events) {
      if (e.type !== "move") this.queryOrder.push(e.pos);
      else if (e.new) this.queryOrder.push(e.nxt!);
    }
  }

  private reset() {
    this.w.fill(0);
    this.applied = 0;
    this.tick = 0;
  }

  private apply(e: FlyEvent) {
    if (e.type !== "move") {
      if (e.reset) this.w.fill(0);
      return;
    }
    const a = e.a ?? 0;
    if (a === 0) return;
    const codes = this.trace.fly.codes;
    for (const c of codes[String(e.nxt)]) this.w[c] += a;
    for (const c of codes[String(e.pos)]) this.w[c] -= a;
  }

  /** State after k measurements (all moves up to and including revisit moves that follow the k-th measurement). */
  seek(k: number): FlyState {
    const ev = this.trace.fly.events;
    if (k < this.tick) this.reset();
    while (this.applied < ev.length && ev[this.applied].q <= k) this.apply(ev[this.applied++]);
    this.tick = k;
    const done = ev.slice(0, this.applied);
    const moves = done.filter((e) => e.type === "move");
    let since = 0;
    for (let i = done.length - 1; i >= 0 && done[i].type === "move" && !done[i].new; i--) since++;
    return {
      tick: k,
      last: done[done.length - 1],
      lastMove: moves.length ? moves[moves.length - 1] : null,
      moves: moves.length,
      revisits: moves.filter((e) => !e.new).length,
      revisitsSinceNew: since,
      episode: done.filter((e) => e.type !== "move").length,
      queried: this.queryOrder.slice(0, k),
      recent: moves.slice(-24),
    };
  }
}

export class BoReplay {
  readonly queryOrder: number[];
  constructor(readonly events: BoEvent[]) {
    this.queryOrder = events.map((e) => e.chosen);
  }
  seek(k: number) {
    const done = this.events.slice(0, k);
    return { tick: k, last: done[done.length - 1], queried: this.queryOrder.slice(0, k) };
  }
}

/** Replays every recorded event and checks the replayed weights against the weight statistics Python recorded. */
export function verifyFlyReplay(trace: Trace): { ok: boolean; checked: number; maxRelErr: number } {
  const r = new FlyReplay(trace);
  let maxRelErr = 0;
  let checked = 0;
  // apply events one tick at a time: after the last event of each tick the weights must match the recorded stats
  const ev = trace.fly.events;
  for (let k = 1; k <= trace.budget; k++) {
    r.seek(k);
    let i = ev.length - 1;
    while (i >= 0 && ev[i].q > k) i--;
    if (i < 0) continue;
    const abs = r.w.reduce((s, v) => s + Math.abs(v), 0);
    let nz = 0;
    for (const v of r.w) if (v !== 0) nz++;
    maxRelErr = Math.max(maxRelErr, Math.abs(abs - ev[i].wabs) / Math.max(1e-12, Math.abs(ev[i].wabs)));
    if (nz !== ev[i].nz) maxRelErr = Infinity;
    checked++;
  }
  return { ok: maxRelErr < 1e-9, checked, maxRelErr };
}

export async function sha256Int32(values: number[]): Promise<string | null> {
  try {
    const buf = new Int32Array(values).buffer;
    const h = await crypto.subtle.digest("SHA-256", buf);
    return [...new Uint8Array(h)].map((b) => b.toString(16).padStart(2, "0")).join("");
  } catch {
    return null; // insecure context or unsupported
  }
}

export function softmax(scores: number[], T: number): number[] {
  const m = Math.max(...scores);
  const e = scores.map((s) => Math.exp((s - m) / T));
  const z = e.reduce((a, b) => a + b, 0);
  return e.map((v) => v / z);
}
