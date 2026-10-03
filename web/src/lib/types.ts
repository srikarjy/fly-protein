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
