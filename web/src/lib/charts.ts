/** Tiny SVG chart toolkit. Everything draws from the data it is given; colours are CSS variables so charts follow the theme. */
export const M_COLOR: Record<string, string> = {
  random_search: "var(--c-random)", random_walk: "var(--c-walk)", greedy_local: "var(--c-greedy)", regevo: "var(--c-evo)",
  gpbo_ei: "var(--c-bo)", adalead_graph: "var(--c-ada)", fly: "var(--c-fly)", fly_memory: "var(--c-fly)", fly_best_tested: "var(--c-fly)",
  zs_wt_8M: "var(--c-zs)", zs_wt_150M: "var(--c-zs)", zs_wt_650M: "var(--c-zs)", zs_mm_8M: "var(--c-zs)", zs_mm_150M: "var(--c-zs)",
};

export const esc = (s: string | number) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]!);
export const f1 = (x: number) => x.toFixed(1);
export const ci = (t: number[], d = 1) => `${t[0].toFixed(d)} [${t[1].toFixed(d)}, ${t[2].toFixed(d)}]`;

export type Scale = ((v: number) => number) & { d0: number; d1: number };
export function linear(d0: number, d1: number, r0: number, r1: number): Scale {
  const f = ((v: number) => r0 + ((v - d0) / (d1 - d0)) * (r1 - r0)) as Scale;
  f.d0 = d0;
  f.d1 = d1;
  return f;
}
export function ticks(lo: number, hi: number, n = 5): number[] {
  const span = hi - lo;
  const step0 = span / n;
  const mag = 10 ** Math.floor(Math.log10(step0));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => s >= step0) ?? 10 * mag;
  const out: number[] = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-9; v += step) out.push(+v.toFixed(10));
  return out;
}

export interface Frame { w: number; h: number; l: number; r: number; t: number; b: number }
export function axes(fr: Frame, x: Scale, y: Scale, o: { xTicks?: number[]; yTicks?: number[]; xLabel?: string; yLabel?: string; xFmt?: (v: number) => string; yFmt?: (v: number) => string; grid?: boolean }) {
  const xt = o.xTicks ?? ticks(x.d0, x.d1, 6);
  const yt = o.yTicks ?? ticks(y.d0, y.d1, 5);
  const xf = o.xFmt ?? String;
  const yf = o.yFmt ?? String;
  let s = "";
  for (const v of yt) s += `<line class="grid" x1="${fr.l}" x2="${fr.w - fr.r}" y1="${y(v)}" y2="${y(v)}"/><text class="tick" x="${fr.l - 8}" y="${y(v) + 4}" text-anchor="end">${yf(v)}</text>`;
  for (const v of xt) s += `<line class="tickline" x1="${x(v)}" x2="${x(v)}" y1="${fr.h - fr.b}" y2="${fr.h - fr.b + 5}"/><text class="tick" x="${x(v)}" y="${fr.h - fr.b + 19}" text-anchor="middle">${xf(v)}</text>`;
  s += `<line class="axis" x1="${fr.l}" x2="${fr.w - fr.r}" y1="${fr.h - fr.b}" y2="${fr.h - fr.b}"/>`;
  if (o.xLabel) s += `<text class="axlabel" x="${(fr.l + fr.w - fr.r) / 2}" y="${fr.h - 6}" text-anchor="middle">${esc(o.xLabel)}</text>`;
  if (o.yLabel) s += `<text class="axlabel" transform="translate(14 ${(fr.t + fr.h - fr.b) / 2}) rotate(-90)" text-anchor="middle">${esc(o.yLabel)}</text>`;
  return s;
}

export interface Line { name: string; color: string; ys: number[]; lo?: number[]; hi?: number[]; dash?: string; width?: number }
export function linePath(xs: number[], ys: number[], x: Scale, y: Scale): string {
  return xs.map((v, i) => `${i ? "L" : "M"}${x(v).toFixed(1)},${y(ys[i]).toFixed(1)}`).join("");
}
export function bandPath(xs: number[], lo: number[], hi: number[], x: Scale, y: Scale): string {
  const up = xs.map((v, i) => `${i ? "L" : "M"}${x(v).toFixed(1)},${y(hi[i]).toFixed(1)}`).join("");
  const dn = xs.map((v, i) => `L${x(xs[xs.length - 1 - i]).toFixed(1)},${y(lo[xs.length - 1 - i]).toFixed(1)}`).join("");
  return up + dn + "Z";
}

/** Spread right-edge labels so they don't overlap. */
export function spreadLabels(ys: number[], min: number, lo: number, hi: number): number[] {
  const idx = ys.map((_, i) => i).sort((a, b) => ys[a] - ys[b]);
  const out = [...ys];
  for (let k = 1; k < idx.length; k++) out[idx[k]] = Math.max(out[idx[k]], out[idx[k - 1]] + min);
  const over = out[idx[idx.length - 1]] - hi;
  if (over > 0) for (let k = idx.length - 1; k >= 0; k--) out[idx[k]] = Math.max(lo, out[idx[k]] - over);
  return out;
}

/** Horizontal dot plot with intervals: rows of {label, v, lo, hi}. */
export function dotPlot(rows: { label: string; v: number; lo: number; hi: number; color?: string; note?: string }[], o: { x: [number, number]; xLabel: string; zero?: boolean; w?: number; rowH?: number; left?: number; fmt?: (v: number) => string }) {
  const rowH = o.rowH ?? 28;
  const left = o.left ?? 210;
  const w = o.w ?? 720;
  const fr: Frame = { w, h: rows.length * rowH + 54, l: left, r: 24, t: 10, b: 44 };
  const x = linear(o.x[0], o.x[1], fr.l, w - fr.r);
  const fmt = o.fmt ?? ((v: number) => v.toFixed(1));
  let s = "";
  for (const v of ticks(o.x[0], o.x[1], 6)) s += `<line class="grid" x1="${x(v)}" x2="${x(v)}" y1="${fr.t}" y2="${fr.h - fr.b}"/><text class="tick" x="${x(v)}" y="${fr.h - fr.b + 17}" text-anchor="middle">${fmt(v)}</text>`;
  if (o.zero !== false && o.x[0] < 0 && o.x[1] > 0) s += `<line class="zero" x1="${x(0)}" x2="${x(0)}" y1="${fr.t}" y2="${fr.h - fr.b}"/>`;
  rows.forEach((r, i) => {
    const cy = fr.t + i * rowH + rowH / 2;
    const c = r.color ?? "var(--accent)";
    s += `<g><title>${esc(r.label)}: ${fmt(r.v)} [${fmt(r.lo)}, ${fmt(r.hi)}]${r.note ? " · " + esc(r.note) : ""}</title>`;
    s += `<text class="rowlabel" x="${fr.l - 10}" y="${cy + 4}" text-anchor="end">${esc(r.label)}</text>`;
    s += `<line stroke="${c}" stroke-width="2.4" stroke-linecap="round" x1="${x(Math.max(o.x[0], r.lo))}" x2="${x(Math.min(o.x[1], r.hi))}" y1="${cy}" y2="${cy}"/>`;
    s += `<circle cx="${x(r.v)}" cy="${cy}" r="5" fill="${c}"/></g>`;
  });
  s += `<text class="axlabel" x="${(fr.l + w - fr.r) / 2}" y="${fr.h - 6}" text-anchor="middle">${esc(o.xLabel)}</text>`;
  return `<svg class="chart" viewBox="0 0 ${w} ${fr.h}" role="img" aria-label="${esc(o.xLabel)}">${s}</svg>`;
}

/** Vertical grouped bars with error bars. */
export function groupedBars(groups: { label: string; bars: { name: string; v: number; err?: number; color: string }[] }[], o: { y: [number, number]; yLabel: string; w?: number; h?: number; fmt?: (v: number) => string }) {
  const w = o.w ?? 720;
  const h = o.h ?? 340;
  const fr: Frame = { w, h, l: 56, r: 16, t: 14, b: 54 };
  const y = linear(o.y[0], o.y[1], h - fr.b, fr.t);
  const fmt = o.fmt ?? ((v: number) => v.toFixed(2));
  let s = "";
  for (const v of ticks(o.y[0], o.y[1], 5)) s += `<line class="grid" x1="${fr.l}" x2="${w - fr.r}" y1="${y(v)}" y2="${y(v)}"/><text class="tick" x="${fr.l - 8}" y="${y(v) + 4}" text-anchor="end">${fmt(v)}</text>`;
  const gw = (w - fr.l - fr.r) / groups.length;
  groups.forEach((g, gi) => {
    const bw = Math.min(34, (gw * 0.78) / g.bars.length);
    const x0 = fr.l + gi * gw + (gw - bw * g.bars.length) / 2;
    g.bars.forEach((b, bi) => {
      const bx = x0 + bi * bw;
      const top = y(Math.max(b.v, o.y[0]));
      s += `<g><title>${esc(g.label)} · ${esc(b.name)}: ${fmt(b.v)}${b.err != null ? " ± " + fmt(b.err) : ""}</title><rect x="${bx + 1}" y="${Math.min(top, y(0 < o.y[0] ? o.y[0] : 0))}" width="${bw - 2}" height="${Math.abs(y(Math.max(o.y[0], 0)) - top)}" rx="2" fill="${b.color}"/>`;
      if (b.err != null) s += `<line class="err" x1="${bx + bw / 2}" x2="${bx + bw / 2}" y1="${y(b.v - b.err)}" y2="${y(b.v + b.err)}"/>`;
      s += `</g>`;
    });
    s += `<text class="rowlabel" x="${fr.l + gi * gw + gw / 2}" y="${h - fr.b + 20}" text-anchor="middle">${esc(g.label)}</text>`;
  });
  s += `<line class="axis" x1="${fr.l}" x2="${w - fr.r}" y1="${y(Math.max(0, o.y[0]))}" y2="${y(Math.max(0, o.y[0]))}"/>`;
  s += `<text class="axlabel" transform="translate(14 ${(fr.t + h - fr.b) / 2}) rotate(-90)" text-anchor="middle">${esc(o.yLabel)}</text>`;
  return `<svg class="chart" viewBox="0 0 ${w} ${h}" role="img" aria-label="${esc(o.yLabel)}">${s}</svg>`;
}

/** Small multi-line chart (few points per line). */
export function smallLines(lines: { name: string; color: string; pts: [number, number][] }[], o: { x: [number, number]; y: [number, number]; xLabel: string; yLabel: string; xTicks?: number[]; xFmt?: (v: number) => string; log?: boolean; w?: number; h?: number }) {
  const w = o.w ?? 340;
  const h = o.h ?? 250;
  const fr: Frame = { w, h, l: 52, r: 26, t: 12, b: 46 };
  const lx = (v: number) => (o.log ? Math.log2(v) : v);
  const x = linear(lx(o.x[0]), lx(o.x[1]), fr.l, w - fr.r);
  const y = linear(o.y[0], o.y[1], h - fr.b, fr.t);
  const xs = o.xTicks ?? ticks(o.x[0], o.x[1], 4);
  let s = "";
  for (const v of ticks(o.y[0], o.y[1], 4)) s += `<line class="grid" x1="${fr.l}" x2="${w - fr.r}" y1="${y(v)}" y2="${y(v)}"/><text class="tick" x="${fr.l - 6}" y="${y(v) + 4}" text-anchor="end">${v.toFixed(2)}</text>`;
  for (const v of xs) s += `<text class="tick" x="${x(lx(v))}" y="${h - fr.b + 17}" text-anchor="middle">${(o.xFmt ?? String)(v)}</text>`;
  s += `<line class="axis" x1="${fr.l}" x2="${w - fr.r}" y1="${h - fr.b}" y2="${h - fr.b}"/>`;
  for (const l of lines) {
    s += `<path d="${l.pts.map(([a, b], i) => `${i ? "L" : "M"}${x(lx(a)).toFixed(1)},${y(b).toFixed(1)}`).join("")}" fill="none" stroke="${l.color}" stroke-width="2.2"/>`;
    for (const [a, b] of l.pts) s += `<circle cx="${x(lx(a))}" cy="${y(b)}" r="3.2" fill="${l.color}"><title>${esc(l.name)}: ${a} → ${b.toFixed(3)}</title></circle>`;
  }
  s += `<text class="axlabel" x="${(fr.l + w - fr.r) / 2}" y="${h - 6}" text-anchor="middle">${esc(o.xLabel)}</text>`;
  s += `<text class="axlabel" transform="translate(13 ${(fr.t + h - fr.b) / 2}) rotate(-90)" text-anchor="middle">${esc(o.yLabel)}</text>`;
  return `<svg class="chart" viewBox="0 0 ${w} ${h}" role="img" aria-label="${esc(o.yLabel)} vs ${esc(o.xLabel)}">${s}</svg>`;
}
