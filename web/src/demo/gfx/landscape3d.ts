/** WebGL protein-landscape map for one method. Marks are instanced (one draw call per layer), in CSS-pixel units under an orthographic camera,
 *  so the fly stands on the same pixel as the recorded variant it occupies. Everything drawn is read from the StepView; the only
 *  thing computed here is presentation: positions between two recorded variants, heading and wing amplitude. */
import { OrthographicCamera, Scene, type WebGLRenderer } from "three";
import { flyPose, type Reveal } from "../choreo";
import { signed } from "../models";
import type { Engine, StepView } from "../state";
import { createFly, type FlyHandle } from "./fly3d";
import { Dots } from "./dots";
import { viridisRGB, type LandscapeHooks, type LandscapeRenderer } from "./common";
import { lights, makeRenderer } from "./gl";
import { Segments } from "./segments";
import { readTheme, type RGB, type Theme } from "./theme";

const PAD = 16;
const TRAIL = 24;
const clamp01 = (x: number) => Math.max(0, Math.min(1, x));
const wrapAngle = (a: number) => Math.atan2(Math.sin(a), Math.cos(a));

export class LandscapeGL implements LandscapeRenderer {
  private canvas = document.createElement("canvas");
  private overlay = document.createElement("div");
  private ring = document.createElement("div");
  private renderer: WebGLRenderer;
  private scene = new Scene();
  private camera = new OrthographicCamera(0, 1, 1, 0, -100, 100);
  private theme: Theme = readTheme();
  private w = 0;
  private h = 0;
  private px: Float32Array;
  private py: Float32Array;
  private base: Dots;
  private top1: Dots;
  private vFill: Dots;
  private vRing: Dots;
  private cand: Dots;
  private fx: Dots; // halo + shadow under the data layers
  private marks: Dots; // best-so-far, departing ring, pulse, anchor, shadow, query marker
  private trail: Segments;
  private link: Segments;
  private fly: FlyHandle | null = null;
  private heading = 0;
  private headingSet = false;
  private intensity = 0;
  private labels: { best: HTMLElement; delta: HTMLElement; ranks: HTMLElement[] };
  private lost = false;
  private cleanup: (() => void)[] = [];

  constructor(private host: HTMLElement, private engine: Engine, private mode: "fly" | "bo", private hooks: LandscapeHooks) {
    const land = engine.land;
    this.px = new Float32Array(land.n);
    this.py = new Float32Array(land.n);
    this.renderer = makeRenderer(this.canvas);
    this.canvas.setAttribute("role", "img");
    this.canvas.tabIndex = 0;
    this.overlay.className = "map-overlay";
    this.ring.className = "hover-ring";
    this.ring.hidden = true;
    host.append(this.canvas, this.overlay);
    this.overlay.append(this.ring);

    const n = land.n;
    this.base = new Dots(n, 0);
    this.top1 = new Dots(n, 1);
    this.vFill = new Dots(engine.budget, 3);
    this.vRing = new Dots(engine.budget, 4);
    this.cand = new Dots(16, 5);
    this.fx = new Dots(2, 2);
    this.marks = new Dots(16, 6);
    this.trail = new Segments(TRAIL + 1, 2);
    this.link = new Segments(2, 7);
    for (const o of [this.base, this.top1, this.fx, this.vFill, this.vRing, this.cand, this.marks, this.trail, this.link]) this.scene.add(o.mesh);
    for (let i = 0; i < n; i++) {
      const [r, g, b] = viridisRGB(land.pct[i] / 100);
      this.base.set(i, 0, 0, 0, 4.4, r, g, b, 0.62);
    }
    this.base.count(n);
    let t1 = 0;
    for (let i = 0; i < n; i++) if (land.pct[i] >= 99) t1++;
    this.top1.count(t1);

    if (mode === "fly") {
      this.fly = createFly();
      for (const l of lights()) this.scene.add(l);
      this.scene.add(this.fly.group);
    }
    const mk = (cls: string) => { const e = document.createElement("div"); e.className = cls; e.hidden = true; this.overlay.append(e); return e; };
    this.labels = { best: mk("map-label"), delta: mk("map-delta"), ranks: Array.from({ length: 5 }, () => mk("map-rank")) };

    const onMove = (ev: PointerEvent) => this.inspectAt(ev);
    const onLeave = () => { this.ring.hidden = true; hooks.onInspect(null); };
    this.canvas.addEventListener("pointermove", onMove);
    this.canvas.addEventListener("pointerdown", onMove);
    this.canvas.addEventListener("pointerleave", onLeave);
    const onLost = (e: Event) => { e.preventDefault(); this.lost = true; hooks.onLost(); };
    this.canvas.addEventListener("webglcontextlost", onLost);
    this.cleanup.push(() => { this.canvas.removeEventListener("pointermove", onMove); this.canvas.removeEventListener("pointerdown", onMove); this.canvas.removeEventListener("pointerleave", onLeave); this.canvas.removeEventListener("webglcontextlost", onLost); });
    this.resize();
  }

  setTheme() { this.theme = readTheme(); this.resize(); }

  resize() {
    const w = this.host.clientWidth;
    const h = this.host.clientHeight;
    if (!w || !h) return;
    this.w = w; this.h = h;
    this.renderer.setSize(w, h, false);
    this.camera.left = 0; this.camera.right = w; this.camera.bottom = 0; this.camera.top = h;
    this.camera.updateProjectionMatrix();
    const land = this.engine.land;
    for (let i = 0; i < land.n; i++) { this.px[i] = PAD + land.x[i] * (w - 2 * PAD); this.py[i] = PAD + land.y[i] * (h - 2 * PAD); }
    for (let i = 0; i < land.n; i++) this.base.setPos(i, this.px[i], this.py[i]);
    this.base.flush();
    let j = 0;
    for (let i = 0; i < land.n; i++) if (land.pct[i] >= 99) { const [r, g, b] = this.theme.top1; this.top1.set(j++, this.px[i], this.py[i], 0, 8, r, g, b, 0.95, 0.24); }
    this.top1.flush();
  }

  /** nearest variant under the pointer, for the hover inspector */
  private inspectAt(ev: PointerEvent) {
    const r = this.canvas.getBoundingClientRect();
    const x = ev.clientX - r.left;
    const y = this.h - (ev.clientY - r.top);
    let best = -1, bd = 14 * 14;
    for (let i = 0; i < this.px.length; i++) {
      const d = (this.px[i] - x) ** 2 + (this.py[i] - y) ** 2;
      if (d < bd) { bd = d; best = i; }
    }
    if (best < 0) { this.ring.hidden = true; this.hooks.onInspect(null); return; }
    this.ring.hidden = false;
    this.ring.style.transform = `translate(${this.px[best] - 9}px, ${this.h - this.py[best] - 9}px)`;
    this.hooks.onInspect({ variant: best, clientX: ev.clientX, clientY: ev.clientY });
  }

  private X = (i: number) => this.px[i];
  private Y = (i: number) => this.py[i];

  render(view: StepView, reveal: Reveal, now: number, dt: number): boolean {
    if (this.lost || !this.w) return false;
    const T = this.theme;
    const f = view.fly;
    const [bgr, bgg, bgb] = T.soft;
    this.renderer.setClearColor((Math.round(bgr * 255) << 16) | (Math.round(bgg * 255) << 8) | Math.round(bgb * 255), 1);
    const fly = this.mode === "fly";
    const mcol = fly ? T.fly : T.bo;
    const set = (d: Dots, i: number, x: number, y: number, size: number, c: RGB, a: number, ring = 0) => d.set(i, x, y, 0, size, c[0], c[1], c[2], a, ring);

    // ---- measured variants
    const q = fly ? f.queried : view.bo.queried;
    const newest = q.length - 1;
    q.forEach((v, i) => {
      const grow = i === newest ? 0.35 + 0.65 * Math.max(reveal.measure, view.k === 1 ? 1 : 0) : 1;
      const early = !fly && i < 10;
      set(this.vFill, i, this.X(v), this.Y(v), 6.8 * grow, early ? T.bg : T.panel, 1);
      set(this.vRing, i, this.X(v), this.Y(v), 7.6 * grow, early ? T.muted : mcol, 1, 0.3);
    });
    this.vFill.count(q.length); this.vRing.count(q.length);
    this.vFill.flush(); this.vRing.flush();

    let nMarks = 0;
    const mark = (x: number, y: number, size: number, c: RGB, a: number, ring = 0) => { if (nMarks < 16) set(this.marks, nMarks++, x, y, size, c, a, ring); };

    // ---- best so far
    let best: number | null = view.k > 0 ? (fly ? f.bestVariant : view.bo.bestVariant) : null;
    if (best !== null) { mark(this.X(best), this.Y(best), 21, T.ink, 0.9, 0.09); mark(this.X(best), this.Y(best), 12.5, T.ink, 0.9, 0.12); }

    // ---- candidates and the selected move
    let candN = 0;
    const cand = (x: number, y: number, size: number, c: RGB, a: number, ring: number) => set(this.cand, candN++, x, y, size, c, a, ring);
    const d = f.decision;
    let linkN = 0;
    if (fly && d) {
      d.candidates.forEach((c, j) => cand(this.X(c), this.Y(c), 6 + 26 * d.probs[j], T.ink, (j === d.chosenIndex ? 0.95 : 0.45) * reveal.readout, j === d.chosenIndex ? 0.2 : 0.14));
      mark(this.X(f.from), this.Y(f.from), 14, T.muted, 0.95 * reveal.readout, 0.16); // where the learner decided from
      const ax = this.X(f.from), ay = this.Y(f.from), bx = this.X(d.chosen), by = this.Y(d.chosen);
      const m = reveal.move;
      this.link.set(linkN++, ax, ay, ax + (bx - ax) * m, ay + (by - ay) * m, 2.6, T.fly[0], T.fly[1], T.fly[2], 0.95);
    }
    if (!fly && view.bo.top.length) {
      const maxEi = Math.max(1e-12, ...view.bo.top.map((t) => t.ei));
      view.bo.top.forEach((t, r) => cand(this.X(t.i), this.Y(t.i), 13 + 14 * (t.ei / maxEi), T.bo, (r === 0 ? 1 : 0.7) * reveal.readout, r === 0 ? 0.2 : 0.12));
    }
    this.cand.count(candN); this.cand.flush();
    this.link.count(linkN); this.link.flush();

    // ---- trail (the fly's recorded moves; the newest segments are drawn as the fly travels them)
    if (fly) {
      const tr = f.trail;
      const Hm = f.hops.filter((h) => h.kind === "move").length;
      tr.forEach(([a, b], i) => {
        const e = tr.length - 1 - i;
        let frac = 1;
        if (d) frac = e === 0 ? reveal.move : e < Hm ? clamp01(reveal.arrive * (Hm - 1) - (Hm - 1 - e)) : 1;
        const ax = this.X(a), ay = this.Y(a), bx = this.X(b), by = this.Y(b);
        this.trail.set(i, ax, ay, ax + (bx - ax) * frac, ay + (by - ay) * frac, 1.5, T.fly[0], T.fly[1], T.fly[2], 0.2 + 0.4 * ((i + 1) / tr.length));
      });
      this.trail.count(tr.length); this.trail.flush();
    } else {
      this.trail.count(0);
    }

    // ---- the fly (or GP-BO's query marker)
    let busy = false;
    let cx = 0, cy = 0;
    if (fly && this.fly) {
      const p = flyPose(view, reveal);
      const x0 = this.X(p.from), y0 = this.Y(p.from), x1 = this.X(p.to), y1 = this.Y(p.to);
      cx = p.u >= 1 ? x1 : x0 + (x1 - x0) * p.u;
      cy = p.u >= 1 ? y1 : y0 + (y1 - y0) * p.u;
      const dx = x1 - x0, dy = y1 - y0;
      let target = this.heading;
      if (Math.hypot(dx, dy) > 1) target = Math.atan2(dy, dx);
      else if (!this.headingSet && f.from !== f.at) target = Math.atan2(this.Y(f.at) - this.Y(f.from), this.X(f.at) - this.X(f.from));
      if (!this.headingSet) { this.heading = target; this.headingSet = true; }
      const k = 1 - Math.exp(-Math.min(dt, 64) / 110);
      const err = wrapAngle(target - this.heading);
      this.heading += err * k;
      const moving = !reveal.settled && (reveal.phase === "arrive" || reveal.phase === "move");
      const wantI = reveal.settled ? 0 : moving ? 1 : 0.3;
      this.intensity += (wantI - this.intensity) * (1 - Math.exp(-Math.min(dt, 64) / 140));
      const size = Math.max(36, Math.min(60, 0.112 * Math.min(this.w, this.h)));
      const bank = Math.max(-0.35, Math.min(0.35, err * 0.5)) * this.intensity;
      this.fly.update({ x: cx, y: cy, heading: this.heading, intensity: this.intensity, bank, timeMs: now }, size);
      set(this.fx, 0, cx, cy, size * 1.25, T.panel, 0.6, -1); // light halo so the fly reads against dark points
      set(this.fx, 1, cx + size * 0.12, cy - size * 0.16, size * 0.9, [0, 0, 0], 0.14, -1); // soft contact shadow
      this.fx.count(2); this.fx.flush();
      mark(cx, cy, 17, T.fly, 0.95, 0.12); // anchor ring on the exact recorded coordinate of the fly
      busy = this.intensity > 0.02 || Math.abs(err) > 0.02;
      if (d && reveal.measure > 0 && reveal.measure < 1) mark(this.X(f.at), this.Y(f.at), 16 + 44 * reveal.measure, d.fitnessDelta >= 0 ? T.accent : T.muted, 0.7 * (1 - reveal.measure), 0.1);
    } else if (!fly) {
      const chosen = view.bo.chosen ?? view.start;
      cx = this.X(chosen); cy = this.Y(chosen);
      mark(cx, cy, 13 * (view.k === 0 ? 1 : 0.5 + 0.5 * Math.max(reveal.move, reveal.measure)), view.k === 0 ? T.muted : T.bo, 1);
      mark(cx, cy, 15, T.bg, 1, 0.16);
      if (view.k > 0 && reveal.measure > 0 && reveal.measure < 1) mark(cx, cy, 16 + 44 * reveal.measure, T.bo, 0.7 * (1 - reveal.measure), 0.1);
    }
    this.marks.count(nMarks); this.marks.flush();

    this.renderer.render(this.scene, this.camera);

    // ---- text overlays (DOM, updated only when their content changes)
    this.updateLabels(view, reveal, best, cx, cy);
    return busy || !reveal.settled;
  }

  private updateLabels(view: StepView, reveal: Reveal, best: number | null, cx: number, cy: number) {
    const L = this.labels;
    const place = (el: HTMLElement, x: number, y: number, text: string, opacity: number) => {
      el.hidden = opacity <= 0.01;
      if (el.hidden) return;
      if (el.textContent !== text) el.textContent = text;
      el.style.transform = `translate(${Math.round(x)}px, ${Math.round(this.h - y)}px)`;
      el.style.opacity = String(opacity);
    };
    if (best !== null) place(L.best, this.X(best) + 12, this.Y(best) - 14, "best so far", 1); else L.best.hidden = true;
    const d = view.fly.decision;
    if (this.mode === "fly" && d && reveal.measure > 0) {
      place(L.delta, cx + 26, cy + 34, `${signed(d.fitnessDelta)} fitness`, reveal.measure);
      L.delta.dataset.dir = d.fitnessDelta >= 0 ? "up" : "down";
    } else L.delta.hidden = true;
    L.ranks.forEach((el, r) => {
      const t = this.mode === "bo" ? view.bo.top[r] : undefined;
      if (t) place(el, this.X(t.i) - 3, this.Y(t.i) + 20 + 6, String(r + 1), reveal.readout);
      else el.hidden = true;
    });
  }

  dispose() {
    for (const c of this.cleanup) c();
    for (const o of [this.base, this.top1, this.fx, this.vFill, this.vRing, this.cand, this.marks, this.trail, this.link]) o.dispose();
    this.fly?.dispose();
    this.renderer.dispose();
    this.canvas.remove();
    this.overlay.remove();
  }
}
