/** Best-percentile-found vs measurements for both methods (recorded runs). SVG, redrawn per step; shares nothing with the renderers. */
import { M_COLOR, linePath, linear, spreadLabels, ticks } from "../lib/charts";
import type { Engine } from "./state";

export function renderTrend(el: HTMLElement, engine: Engine, k: number) {
  const B = engine.budget;
  const best = (order: number[]) => { let b = 0; return order.map((i) => (b = Math.max(b, engine.land.pct[i]))); };
  const flyBest = best(engine.queryOrder.fly);
  const boBest = best(engine.queryOrder.bo);
  const narrow = el.clientWidth < 640;
  const W = narrow ? 460 : 900, H = narrow ? 260 : 230, l = narrow ? 40 : 54, r = narrow ? 12 : 150, t = 12, b = 40;
  const lo = Math.floor(Math.min(flyBest[0], boBest[0]) / 5) * 5;
  const x = linear(0, B, l, W - r), y = linear(lo, 100.2, H - b, t);
  const xs = Array.from({ length: B }, (_, i) => i + 1);
  let s = "";
  for (const v of ticks(lo, 100, 4)) s += `<line class="grid" x1="${l}" x2="${W - r}" y1="${y(v)}" y2="${y(v)}"/><text class="tick" x="${l - 8}" y="${y(v) + 4}" text-anchor="end">${v}</text>`;
  for (const v of [0, 50, 100, 150, 200]) s += `<text class="tick" x="${x(v)}" y="${H - b + 17}" text-anchor="middle">${v}</text>`;
  s += `<line class="axis" x1="${l}" x2="${W - r}" y1="${H - b}" y2="${H - b}"/><text class="axlabel" x="${(l + W - r) / 2}" y="${H - 6}" text-anchor="middle">fitness measurements</text>`;
  s += `<line class="grid" x1="${l}" x2="${W - r}" y1="${y(99)}" y2="${y(99)}" stroke-dasharray="3 3"/><text class="tick" x="${x(150)}" y="${y(99) + 17}" text-anchor="middle">assay top 1% (dashed line)</text>`;
  const labels: { y: number; col: string; name: string }[] = [];
  for (const [bst, col, name] of [[flyBest, M_COLOR.fly, "Fly learner"], [boBest, M_COLOR.gpbo_ei, "GP-BO"]] as [number[], string, string][]) {
    s += `<path d="${linePath(xs, bst, x, y)}" fill="none" stroke="${col}" stroke-width="1.4" opacity="0.28"/>`;
    if (k > 0) s += `<path d="${linePath(xs.slice(0, k), bst.slice(0, k), x, y)}" fill="none" stroke="${col}" stroke-width="2.8"/><circle cx="${x(k)}" cy="${y(bst[k - 1])}" r="4.5" fill="${col}"/>`;
    labels.push({ y: y(bst[B - 1]), col, name });
  }
  const sp = spreadLabels(labels.map((q) => q.y), 18, t + 12, H - b - 6);
  if (!narrow) labels.forEach((q, i) => { s += `<text class="mlabel" x="${W - r + 8}" y="${sp[i] + 4}" fill="${q.col}">${q.name}</text>`; });
  else s += `<text class="mlabel" x="${l + 8}" y="${t + 14}" fill="${M_COLOR.gpbo_ei}">GP-BO</text><text class="mlabel" x="${l + 70}" y="${t + 14}" fill="${M_COLOR.fly}">Fly learner</text>`;
  el.innerHTML = `<svg class="chart" viewBox="0 0 ${W} ${H}" role="img" aria-label="Best percentile found versus measurements for both methods, at measurement ${k} of ${B}">${s}</svg>`;
}
