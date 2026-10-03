export interface FlyEvent {
  type: "init" | "restart" | "move";
  pos: number;
  q: number;
  f: number;
  reset?: boolean;
  j?: number;
  nxt?: number;
  scores?: number[];
  new?: boolean;
  reward?: number;
  delta?: number;
  a?: number;
  wabs: number;
  nz: number;
}
export interface BoCandidate { i: number; ei: number; mu: number; sd: number }
export interface BoEvent { type: "init" | "bo"; chosen: number; q: number; f: number; ell?: number; s2?: number; n_obs?: number; top?: BoCandidate[] }
export interface Trace {
  assay: string;
  seed: number;
  budget: number;
  start: number;
  n_kc: number;
  ka: number;
  embedding: string;
  pn_scale: number;
  fly: { events: FlyEvent[]; codes: Record<string, number[]>; nbrs: Record<string, number[]>; pn: Record<string, number[]> };
  gpbo: { events: BoEvent[] };
  verified: { fly_log_sha256: string; gpbo_log_sha256: string; note: string };
}
export interface Landscape { assay: string; n: number; mutants: string[]; x: number[]; y: number[]; fitness: number[]; pct: number[] }

/** Additive display serialization (scripts/export_demo_circuit.py): what the trace does not carry about the PN->KC circuit. */
export interface Circuit {
  assay: string;
  seed: number;
  trace_sha256: string;
  meta: { n_pn: number; n_kc: number; per_kc: number; k_active: number; temperature: number; wiring: string; pn: string };
  /** little-endian uint16, row-major (n_kc x per_kc): the PN indices each Kenyon cell samples */
  wiring_b64: string;
  /** embedding-unit scale of each measured variant's 8-bit PN values (value = q / 127 * pn_max) */
  pn_max: Record<string, number>;
  /** exact input drive of each active Kenyon cell (same order as trace.fly.codes) */
  drive: Record<string, number[]>;
  /** [weakest active drive, strongest inactive drive] */
  cutoff: Record<string, [number, number]>;
  /** the learner's move probabilities per recorded event (null for non-move events) */
  probs: (number[] | null)[];
}
