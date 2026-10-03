/** One authoritative current replay step. Every control (Start, Pause, Next, Previous, scrubber, step input, Reset, the URL hash)
 *  goes through setStep, and every panel subscribes here, so no panel can hold a different step from another. */
import type { Engine, StepView } from "./state";

export type StepSource = "play" | "next" | "prev" | "scrub" | "input" | "reset" | "deeplink";

export interface StepChange {
  k: number;
  prevK: number;
  source: StepSource;
  view: StepView;
}

export class ReplayStore {
  private current = 0;
  private listeners = new Set<(c: StepChange) => void>();

  constructor(readonly engine: Engine) {}

  get k() { return this.current; }
  get view(): StepView { return this.engine.view(this.current); }

  /** Moves to measurement k (clamped to 0..budget). Returns false if nothing changed. */
  setStep(k: number, source: StepSource): boolean {
    const next = Math.max(0, Math.min(this.engine.budget, Number.isFinite(k) ? Math.round(k) : 0));
    if (next === this.current) return false;
    const prevK = this.current;
    this.current = next;
    const change: StepChange = { k: next, prevK, source, view: this.engine.view(next) };
    for (const fn of [...this.listeners]) fn(change);
    return true;
  }

  next(source: StepSource = "next") { return this.setStep(this.current + 1, source); }
  prev(source: StepSource = "prev") { return this.setStep(this.current - 1, source); }
  reset() { return this.setStep(0, "reset"); }

  subscribe(fn: (c: StepChange) => void): () => void {
    this.listeners.add(fn);
    return () => this.listeners.delete(fn);
  }
}

/** `#demo-90` -> 90 (clamped to the budget); anything else -> null. */
export function parseDeepLink(hash: string, budget: number): number | null {
  const m = /^#demo-(\d+)$/.exec(hash);
  return m ? Math.min(budget, +m[1]) : null;
}

export const deepLinkFor = (k: number) => (k > 0 ? `#demo-${k}` : "#demo");
