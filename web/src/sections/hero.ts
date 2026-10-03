import { M_COLOR, axes, bandPath, ci, esc, linePath, linear, spreadLabels, type Frame } from "../lib/charts";
import { $ } from "../lib/dom";

interface Card { id: string; value: string; unit: string; ci: number[] | null; label: string; sub: string }
export interface Hero {
  cards: Card[];
  figure: { budgets: number[]; names: Record<string, string>; curves: Record<string, Record<string, number[]>> };
  table: { top1: Record<string, Record<string, number[]>>; pct: Record<string, Record<string, number[]>> };
  meta: { n_assays: number; n_seeds: number; assays: string[] };
}

export function renderCards(h: Hero) {
  $("#cards").innerHTML = h.cards
    .map((c) => `<article class="card" data-card="${esc(c.id)}"><div class="big">${esc(c.value)}${c.unit ? `<span class="unit">${esc(c.unit)}</span>` : ""}</div>${c.ci ? `<div class="cinterval">95% CI [${c.ci[0]}, ${c.ci[1]}]</div>` : ""}<p class="clabel">${esc(c.label)}</p><p class="csub">${esc(c.sub)}</p></article>`)
    .join("");
}

const ORDER = ["random_search", "fly_memory", "zs_wt_650M", "gpbo_ei", "adalead_graph"];
const DASH: Record<string, string> = { random_search: "5 4" };

export function renderFigure(h: Hero) {
  const host = $("#fig-main");
  let metric: "top1" | "pct" = "top1";
  let xmax = 300;
  const names = h.figure.names;
  function draw() {
    const narrow = host.clientWidth > 0 && host.clientWidth < 640;
    const W = narrow ? 460 : 900, H = narrow ? 470 : 440;
    const fr: Frame = narrow ? { w: W, h: H, l: 42, r: 12, t: 18, b: 56 } : { w: W, h: H, l: 58, r: 200, t: 18, b: 52 };
    const xs = Array.from({ length: xmax }, (_, i) => i + 1);
    const yDom: [number, number] = metric === "top1" ? [0, 100] : [90, 100];
    const x = linear(0, xmax, fr.l, W - fr.r);
    const y = linear(yDom[0], yDom[1], H - fr.b, fr.t);
    const pick = (m: string, k: string) => (h.figure.curves[m][metric === "top1" ? k : "pct" + k.slice(4)] as number[]);
    const get = (m: string) => ({ mean: pick(m, "top1").slice(0, xmax), lo: pick(m, "top1_lo").slice(0, xmax), hi: pick(m, "top1_hi").slice(0, xmax) });
    let svg = `<rect class="band" x="${x(50)}" y="${fr.t}" width="${x(200) - x(50)}" height="${H - fr.b - fr.t}"/><text class="bandlabel" x="${(x(50) + x(200)) / 2}" y="${fr.t + 14}" text-anchor="middle">${narrow ? "50–200: informative" : "50–200 measurements: the informative range"}</text>`;
    svg += axes(fr, x, y, { xLabel: narrow ? "fitness measurements" : "fitness measurements (variants whose fitness has been measured)", yLabel: metric === "top1" ? (narrow ? "runs at assay top 1% (%)" : "runs that have reached the assay's top 1% (%)") : "best fitness percentile found", xTicks: xmax === 300 ? [0, 50, 100, 150, 200, 250, 300] : [0, 100, 200, 300, 400, 500], yFmt: (v) => (metric === "top1" ? `${v}` : `${v}`) });
    const ends: number[] = [];
    for (const m of ORDER) {
      const g = get(m);
      ends.push(y(g.mean[xmax - 1]));
    }
    const lab = spreadLabels(ends, 17, fr.t + 8, H - fr.b - 4);
    ORDER.forEach((m, i) => {
      const g = get(m);
      const col = M_COLOR[m];
      svg += `<path d="${bandPath(xs, g.lo, g.hi, x, y)}" fill="${col}" opacity="${m.startsWith("zs") ? 0.06 : 0.13}"/>`;
      svg += `<path d="${linePath(xs, g.mean, x, y)}" fill="none" stroke="${col}" stroke-width="${m === "fly_memory" ? 3.2 : 2.2}" ${DASH[m] ? `stroke-dasharray="${DASH[m]}"` : ""} stroke-linejoin="round"/>`;
      if (!narrow) svg += `<text class="mlabel" x="${W - fr.r + 10}" y="${lab[i] + 4}" fill="${col}">${esc(shortName(names[m]))}</text>`;
    });
    svg += `<line id="fig-cursor" class="cursor" y1="${fr.t}" y2="${H - fr.b}" x1="-10" x2="-10"/>`;
    host.innerHTML = `<svg class="chart" viewBox="0 0 ${W} ${H}" role="img" aria-label="Search efficiency of five methods versus number of fitness measurements">${svg}</svg>${narrow ? `<div class="legend">${ORDER.map((m) => `<span><i style="background:${M_COLOR[m]}"></i>${esc(shortName(names[m]))}</span>`).join("")}</div>` : ""}<div id="fig-tip" class="tip" hidden></div>`;
    const el = host.querySelector("svg")!;
    const tip = $("#fig-tip");
    const cur = el.querySelector("#fig-cursor") as SVGLineElement;
    el.addEventListener("pointermove", (ev: PointerEvent) => {
      const r = el.getBoundingClientRect();
      const px = ((ev.clientX - r.left) / r.width) * W;
      const q = Math.round(((px - fr.l) / (W - fr.r - fr.l)) * xmax);
      if (q < 1 || q > xmax) { tip.hidden = true; cur.setAttribute("x1", "-10"); cur.setAttribute("x2", "-10"); return; }
      cur.setAttribute("x1", String(x(q))); cur.setAttribute("x2", String(x(q)));
      const rows = [...ORDER].sort((a, b) => get(b).mean[q - 1] - get(a).mean[q - 1]).map((m) => {
        const g = get(m);
        return `<div><i style="background:${M_COLOR[m]}"></i>${esc(shortName(names[m]))}<b>${g.mean[q - 1].toFixed(metric === "top1" ? 1 : 2)}</b><span>[${g.lo[q - 1].toFixed(metric === "top1" ? 1 : 2)}, ${g.hi[q - 1].toFixed(metric === "top1" ? 1 : 2)}]</span></div>`;
      }).join("");
      tip.innerHTML = `<strong>${q} measurements</strong>${rows}`;
      tip.hidden = false;
      const tx = Math.min(Math.max(8, ((x(q) / W) * r.width) + 14), r.width - 250);
      tip.style.left = `${tx}px`;
    });
    el.addEventListener("pointerleave", () => { tip.hidden = true; cur.setAttribute("x1", "-10"); cur.setAttribute("x2", "-10"); });
  }

  const ctl = $("#fig-controls");
  ctl.innerHTML = `<div class="seg" role="group" aria-label="Metric"><button data-m="top1" aria-pressed="true">Reached assay top 1%</button><button data-m="pct" aria-pressed="false">Best percentile found</button></div>
  <div class="seg" role="group" aria-label="Range"><button data-x="300" aria-pressed="true">0–300</button><button data-x="500" aria-pressed="false">0–500</button></div>`;
  ctl.addEventListener("click", (e) => {
    const b = (e.target as HTMLElement).closest("button");
    if (!b) return;
    if (b.dataset.m) { metric = b.dataset.m as "top1" | "pct"; ctl.querySelectorAll("[data-m]").forEach((x) => x.setAttribute("aria-pressed", String(x === b))); }
    if (b.dataset.x) { xmax = +b.dataset.x; ctl.querySelectorAll("[data-x]").forEach((x) => x.setAttribute("aria-pressed", String(x === b))); }
    draw();
  });
  draw();
  let rz: number | undefined;
  let lastNarrow = host.clientWidth < 640;
  window.addEventListener("resize", () => { window.clearTimeout(rz); rz = window.setTimeout(() => { const n = host.clientWidth < 640; if (n !== lastNarrow) { lastNarrow = n; draw(); } }, 150); });

  const rows = ORDER.map((m) => `<tr><th scope="row"><i style="background:${M_COLOR[m]}"></i>${esc(names[m])}</th>${h.figure.budgets.map((b) => `<td>${ci(h.table.top1[m][String(b)])}</td>`).join("")}</tr>`).join("");
  $("#fig-table").innerHTML = `<table><caption>Share of runs that reached the assay's top 1%, in % (mean over ${h.meta.n_assays} assays; [95% bootstrap CI over assays])</caption><thead><tr><th></th>${h.figure.budgets.map((b) => `<th>${b} measured</th>`).join("")}</tr></thead><tbody>${rows}</tbody></table>`;
}

function shortName(n: string) {
  return n.replace("Fly learner + memory", "Fly learner (+memory)").replace(" zero-shot (WT-marginal)", " zero-shot").replace("GP-BO (Matern-5/2, EI)", "GP-BO");
}
