/** Canvas 2D fallbacks used when WebGL is unavailable (or the context is lost): same StepView, simpler drawing, no 3D fly. */
import { flyPose, type Reveal } from "../choreo";
import { circuitShown, type Engine, type StepView } from "../state";
import { viridisRGB, type InspectHit, type LandscapeHooks, type LandscapeRenderer } from "./common";
import { readTheme, type RGB, type Theme } from "./theme";

const PAD = 16;
const css = (c: RGB, a = 1) => `rgba(${Math.round(c[0] * 255)},${Math.round(c[1] * 255)},${Math.round(c[2] * 255)},${a})`;

function fit(c: HTMLCanvasElement) {
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const w = c.clientWidth, h = c.clientHeight;
  if (c.width !== Math.round(w * dpr) || c.height !== Math.round(h * dpr)) { c.width = Math.round(w * dpr); c.height = Math.round(h * dpr); }
  const g = c.getContext("2d")!;
  g.setTransform(dpr, 0, 0, dpr, 0, 0);
  return { g, w, h };
}

export class Landscape2D implements LandscapeRenderer {
  private canvas = document.createElement("canvas");
  private base: HTMLCanvasElement | null = null;
  private theme: Theme = readTheme();
  private w = 0;
  private h = 0;
  private cleanup: (() => void)[] = [];

  constructor(private host: HTMLElement, private engine: Engine, private mode: "fly" | "bo", hooks: LandscapeHooks) {
    this.canvas.setAttribute("role", "img");
    this.canvas.tabIndex = 0;
    host.append(this.canvas);
    const onMove = (ev: PointerEvent) => {
      const r = this.canvas.getBoundingClientRect();
      const x = ev.clientX - r.left, y = ev.clientY - r.top;
      const L = this.engine.land;
      let best = -1, bd = 14 * 14;
      for (let i = 0; i < L.n; i++) { const d = (this.X(i) - x) ** 2 + (this.Y(i) - y) ** 2; if (d < bd) { bd = d; best = i; } }
      hooks.onInspect(best < 0 ? null : ({ variant: best, clientX: ev.clientX, clientY: ev.clientY } satisfies InspectHit));
    };
    const onLeave = () => hooks.onInspect(null);
    this.canvas.addEventListener("pointermove", onMove);
    this.canvas.addEventListener("pointerleave", onLeave);
    this.cleanup.push(() => { this.canvas.removeEventListener("pointermove", onMove); this.canvas.removeEventListener("pointerleave", onLeave); });
    this.resize();
  }

  private X = (i: number) => PAD + this.engine.land.x[i] * (this.w - 2 * PAD);
  private Y = (i: number) => this.h - PAD - this.engine.land.y[i] * (this.h - 2 * PAD);

  setTheme() { this.theme = readTheme(); this.base = null; }
  resize() { this.w = this.host.clientWidth; this.h = this.host.clientHeight; this.base = null; }

  private baseLayer() {
    if (this.base) return this.base;
    const L = this.engine.land;
    const b = document.createElement("canvas");
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    b.width = Math.round(this.w * dpr); b.height = Math.round(this.h * dpr);
    const g = b.getContext("2d")!;
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    for (let i = 0; i < L.n; i++) { g.fillStyle = css(viridisRGB(L.pct[i] / 100), 0.62); g.beginPath(); g.arc(this.X(i), this.Y(i), 2.2, 0, 6.2832); g.fill(); }
    g.strokeStyle = css(this.theme.top1); g.lineWidth = 1.1;
    for (let i = 0; i < L.n; i++) if (L.pct[i] >= 99) { g.beginPath(); g.arc(this.X(i), this.Y(i), 3.8, 0, 6.2832); g.stroke(); }
    return (this.base = b);
  }

  render(view: StepView, reveal: Reveal): boolean {
    if (!this.w) return false;
    const { g, w, h } = fit(this.canvas);
    const T = this.theme;
    const fly = this.mode === "fly";
    g.fillStyle = css(T.soft); g.fillRect(0, 0, w, h);
    g.drawImage(this.baseLayer(), 0, 0, w, h);
    const circle = (i: number, r: number, stroke?: RGB, fill?: RGB, lw = 1.5, a = 1) => {
      g.beginPath(); g.arc(this.X(i), this.Y(i), r, 0, 6.2832);
      if (fill) { g.fillStyle = css(fill, a); g.fill(); }
      if (stroke) { g.lineWidth = lw; g.strokeStyle = css(stroke, a); g.stroke(); }
    };
    const q = fly ? view.fly.queried : view.bo.queried;
    q.forEach((v, i) => circle(v, 3.6, !fly && i < 10 ? T.muted : fly ? T.fly : T.bo, T.panel, 1.6));
    const best = fly ? view.fly.bestVariant : view.bo.bestVariant;
    if (best !== null) { circle(best, 9, T.ink, undefined, 1.4); circle(best, 5.5, T.ink, undefined, 1.4); }
    if (fly) {
      g.lineWidth = 1.3; g.strokeStyle = css(T.fly, 0.45); g.beginPath();
      view.fly.trail.forEach(([a, b], i) => { if (i === 0) g.moveTo(this.X(a), this.Y(a)); g.lineTo(this.X(b), this.Y(b)); });
      g.stroke();
      const d = view.fly.decision;
      if (d) {
        d.candidates.forEach((c, j) => circle(c, 3 + 12 * d.probs[j], T.ink, undefined, 1, j === d.chosenIndex ? 0.95 : 0.45));
        g.beginPath(); g.moveTo(this.X(view.fly.from), this.Y(view.fly.from)); g.lineTo(this.X(d.chosen), this.Y(d.chosen)); g.lineWidth = 2.4; g.strokeStyle = css(T.fly); g.stroke();
        circle(view.fly.from, 7, T.muted, undefined, 2);
      }
      const p = flyPose(view, reveal);
      const x = p.u >= 1 ? this.X(p.to) : this.X(p.from) + (this.X(p.to) - this.X(p.from)) * p.u;
      const y = p.u >= 1 ? this.Y(p.to) : this.Y(p.from) + (this.Y(p.to) - this.Y(p.from)) * p.u;
      g.beginPath(); g.arc(x, y, 7.5, 0, 6.2832); g.fillStyle = css(T.fly); g.fill(); g.lineWidth = 2.4; g.strokeStyle = css(T.bg); g.stroke();
    } else {
      view.bo.top.forEach((t, r) => { circle(t.i, 9, T.bo, undefined, r === 0 ? 2.6 : 1.4, r === 0 ? 1 : 0.7); g.fillStyle = css(T.ink); g.font = "600 10px ui-sans-serif, system-ui"; g.textAlign = "center"; g.fillText(String(r + 1), this.X(t.i), this.Y(t.i) - 12); });
      circle(view.bo.chosen ?? view.start, 7, T.bg, view.bo.chosen === null ? T.muted : T.bo, 2.4);
    }
    return false;
  }

  dispose() { for (const c of this.cleanup) c(); this.canvas.remove(); }
}

/** Flat version of the circuit panel: PN channels as a bar strip and the 2,000 Kenyon cells as a grid (no wiring edges). */
export class Circuit2D {
  private pn = document.createElement("canvas");
  private kc = document.createElement("canvas");
  private theme: Theme = readTheme();

  constructor(host: HTMLElement, private engine: Engine) {
    this.pn.className = "strip";
    this.kc.className = "kcgrid";
    this.pn.setAttribute("role", "img");
    this.kc.setAttribute("role", "img");
    host.classList.add("circuit-flat");
    host.append(this.pn, this.kc);
  }

  setTheme() { this.theme = readTheme(); }
  resize() { /* canvases are sized by CSS and re-fitted on every render */ }

  render(view: StepView, reveal: Reveal): boolean {
    const T = this.theme;
    const c = circuitShown(view);
    {
      const { g, w, h } = fit(this.pn);
      g.clearRect(0, 0, w, h);
      const bw = w / c.pn.length;
      c.pn.forEach((v, i) => { const m = (Math.abs(v) / 127) * reveal.input; g.fillStyle = css(v >= 0 ? T.fly : T.bo, 0.9); g.fillRect(i * bw, h / 2 - (v >= 0 ? m * (h / 2) : 0), Math.max(1, bw - 0.4), m * (h / 2)); });
      g.fillStyle = css(T.line); g.fillRect(0, h / 2, w, 1);
    }
    {
      const { g, w, h } = fit(this.kc);
      g.clearRect(0, 0, w, h);
      const cols = 125, rows = Math.ceil(this.engine.circ.meta.n_kc / cols);
      const cw = w / cols, ch = h / rows;
      const lit = new Set(c.active);
      const wmax = Math.max(1e-12, ...view.fly.weightsAfter.map(Math.abs));
      for (let i = 0; i < this.engine.circ.meta.n_kc; i++) {
        const on = lit.has(i) && reveal.sparse > 0.7;
        const wi = view.fly.weightsBefore[i] + (view.fly.weightsAfter[i] - view.fly.weightsBefore[i]) * reveal.update;
        const t = Math.tanh((2.2 * wi) / wmax);
        g.fillStyle = on ? css(wi === 0 ? T.ink : t > 0 ? T.fly : T.bo, wi === 0 ? 0.55 : 0.4 + 0.6 * Math.abs(t)) : css(T.kcOff);
        g.fillRect((i % cols) * cw + 0.3, Math.floor(i / cols) * ch + 0.3, cw - 0.6, ch - 0.6);
      }
      const d = view.fly.decision;
      if (d && reveal.update > 0) {
        g.lineWidth = 1;
        for (const [set, col] of [[d.raised, T.accent], [d.lowered, T.muted]] as const) { g.strokeStyle = css(col, reveal.update); for (const i of set) g.strokeRect((i % cols) * cw + 0.2, Math.floor(i / cols) * ch + 0.2, cw - 0.4, ch - 0.4); }
      }
    }
    return false;
  }

  dispose() { this.pn.remove(); this.kc.remove(); }
}
