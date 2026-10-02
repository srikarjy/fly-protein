/** The protein landscape as typed arrays, built from prep/build_data.py's data.json. */
export interface Landscape {
  n: number;
  k: number;
  fitness: Float32Array;
  x: Float32Array;
  y: Float32Array;
  /** n*k neighbor indices, row-major; neighbors of i are nbrs[i*k .. i*k+k) */
  nbrs: Int32Array;
  /** active Kenyon cell indices per variant */
  codes: Int32Array[];
  nKc: number;
  mutants: string[];
  wt: string;
  assay: string;
  embedding: string;
}

interface RawVariant {
  m: string;
  f: number;
  x: number;
  y: number;
  n: number[];
  c: number[];
}

export function parseLandscape(raw: {
  meta: { k: number; n_kc: number; wt: string; assay: string; embedding: string };
  variants: RawVariant[];
}): Landscape {
  const { meta, variants } = raw;
  const n = variants.length;
  const k = meta.k;
  const L: Landscape = {
    n,
    k,
    fitness: new Float32Array(n),
    x: new Float32Array(n),
    y: new Float32Array(n),
    nbrs: new Int32Array(n * k),
    codes: new Array(n),
    nKc: meta.n_kc,
    mutants: new Array(n),
    wt: meta.wt,
    assay: meta.assay,
    embedding: meta.embedding,
  };
  variants.forEach((v, i) => {
    L.fitness[i] = v.f;
    L.x[i] = v.x;
    L.y[i] = v.y;
    L.nbrs.set(v.n, i * k);
    L.codes[i] = Int32Array.from(v.c);
    L.mutants[i] = v.m;
  });
  return L;
}

export function quantile(values: ArrayLike<number>, q: number): number {
  const s = Array.from(values).sort((a, b) => a - b);
  const pos = (s.length - 1) * q;
  const lo = Math.floor(pos);
  const hi = Math.ceil(pos);
  return s[lo] + (s[hi] - s[lo]) * (pos - lo);
}

/** Apply a ProteinGym mutation string ("A23G" or "A23G:C45D", 1-indexed) to the wild type. */
export function applyMutations(wt: string, mutant: string): string {
  const chars = wt.split("");
  for (const m of mutant.split(":")) {
    const pos = parseInt(m.slice(1, -1), 10) - 1;
    chars[pos] = m[m.length - 1];
  }
  return chars.join("");
}
