/** Staging of one replay transition. This is presentation timing only: it maps elapsed time to how far each stage of the
 *  recorded step has been revealed (0..1). It carries no scientific values; renderers combine it with the immutable StepView.
 *  At elapsed >= duration (and always when animation is skipped) every reveal is 1 and the view is exactly the recorded state. */
import type { StepView } from "./state";

export type PhaseId = "arrive" | "input" | "sparse" | "readout" | "move" | "measure" | "update" | "advance";

export interface Reveal {
  /** the stage currently running; "advance" once settled */
  phase: PhaseId;
  /** free revisit moves before the step's decision (or, without a decision, the whole path) have been travelled */
  arrive: number;
  /** PN input values shown */
  input: number;
  /** activity propagated PN -> KC; the sparse winners light up */
  sparse: number;
  /** candidate scores and the chosen mutation shown */
  readout: number;
  /** the fly has travelled the final move */
  move: number;
  /** measured fitness revealed */
  measure: number;
  /** learner/memory update applied (weights, counts) */
  update: number;
  /** best-so-far, path and curves advanced */
  advance: number;
  settled: boolean;
}

export const ORDER: PhaseId[] = ["arrive", "input", "sparse", "readout", "move", "measure", "update", "advance"];

export const SETTLED: Reveal = Object.freeze({ phase: "advance", arrive: 1, input: 1, sparse: 1, readout: 1, move: 1, measure: 1, update: 1, advance: 1, settled: true });

export interface StepSpec {
  /** free hops before the final recorded move of the step */
  hopsBeforeFinal: number;
  /** the final event is a learner decision (a move); false for the initial placement and restarts */
  hasDecision: boolean;
}

export const specOf = (v: StepView): StepSpec => ({ hopsBeforeFinal: Math.max(0, v.fly.hops.length - 1), hasDecision: v.fly.decision !== null });

/** milliseconds per phase at 1x */
export function phaseDurations(spec: StepSpec): Record<PhaseId, number> {
  const travel = spec.hasDecision ? Math.min(1100, 90 * spec.hopsBeforeFinal) : Math.min(1100, 300 + 90 * spec.hopsBeforeFinal);
  return { arrive: travel, input: 450, sparse: 750, readout: spec.hasDecision ? 500 : 0, move: spec.hasDecision ? 750 : 0, measure: 400, update: 500, advance: 250 };
}

export const stepDuration = (spec: StepSpec) => ORDER.reduce((s, p) => s + phaseDurations(spec)[p], 0);

const smooth = (x: number) => { const t = Math.max(0, Math.min(1, x)); return t * t * (3 - 2 * t); };

export function revealAt(elapsedMs: number, spec: StepSpec): Reveal {
  const d = phaseDurations(spec);
  if (elapsedMs >= stepDuration(spec)) return SETTLED;
  const out: Record<string, number | string | boolean> = { settled: false };
  let t = Math.max(0, elapsedMs);
  let phase: PhaseId = "arrive";
  let found = false;
  for (const p of ORDER) {
    const dur = d[p];
    if (dur === 0) { out[p] = 1; continue; }
    out[p] = smooth(t / dur);
    if (!found && t < dur) { phase = p; found = true; }
    t -= dur;
  }
  out.phase = phase;
  return out as unknown as Reveal;
}

/** "Stage animation" switched off (or very fast playback): only the fly's travel is animated, every panel shows the settled recorded state at once. */
export function glideReveal(g: number, spec: StepSpec): Reveal {
  if (g >= 1) return SETTLED;
  const free = spec.hopsBeforeFinal > 0;
  const arrive = spec.hasDecision ? (free ? smooth(g / 0.45) : 1) : smooth(g);
  const move = spec.hasDecision ? smooth((g - (free ? 0.35 : 0)) / (free ? 0.65 : 1)) : 1;
  return { phase: spec.hasDecision && arrive >= 1 ? "move" : "arrive", arrive, input: 1, sparse: 1, readout: 1, move, measure: 1, update: 1, advance: 1, settled: false };
}

export interface Pose {
  /** the two recorded variants the fly is between, and how far along (0..1); u === 1 means exactly at `to` */
  from: number;
  to: number;
  u: number;
}

/** Where the fly is between recorded variants. The path is the recorded hops of the step: prevAt -> hop 1 -> ... -> at. */
export function flyPose(v: StepView, r: Reveal): Pose {
  const nodes = [v.fly.prevAt, ...v.fly.hops.map((h) => h.to)];
  const segs = nodes.length - 1;
  if (segs <= 0) return { from: v.fly.at, to: v.fly.at, u: 1 };
  const hasDecision = v.fly.decision !== null;
  // s counts segments travelled: the free hops during "arrive", then the final move during "move" (or all of it during "arrive" without a decision)
  const s = hasDecision ? r.arrive * (segs - 1) + r.move : r.arrive * segs;
  if (s >= segs) return { from: nodes[segs - 1], to: nodes[segs], u: 1 };
  const i = Math.min(segs - 1, Math.floor(s));
  return { from: nodes[i], to: nodes[i + 1], u: s - i };
}
