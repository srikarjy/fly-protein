/** WebGL view of the recorded PN -> KC circuit. All 320 input channels and all 2,000 Kenyon cells are drawn as instanced dots; edges are
 *  the real wiring (circuit.wiring) from every input channel to each *active* Kenyon cell only, so the picture shows input activity ->
 *  sparse response without drawing the other ~60k potential connections. Values, active set, drive and weights all come from the StepView. */
import { BufferAttribute, BufferGeometry, DynamicDrawUsage, Group, LineBasicMaterial, LineSegments, PerspectiveCamera, Scene, ShaderMaterial, Vector3, type WebGLRenderer } from "three";
import type { Reveal } from "../choreo";
import { circuitShown, type Engine, type StepView } from "../state";
import { Dots } from "./dots";
import { makeRenderer } from "./gl";
import { mix, readTheme, type RGB, type Theme } from "./theme";

const N_PN = 320, PN_COLS = 16, PN_ROWS = 20, PN_PITCH = 0.27;
const N_KC = 2000, KC_COLS = 40, KC_ROWS = 50, KC_PITCH = 0.152;
const PN_X = -5.3, KC_X = 3.7, YAW = 0.3;
const VIEW_W = 17;

export interface CircuitHit { kind: "pn" | "kc"; index: number; clientX: number; clientY: number }
export interface CircuitHooks { onInspect(hit: CircuitHit | null): void; onLost(): void }

const edgeVert = /* glsl */ `
attribute float aT; attribute float aS; attribute float aSign;
uniform vec3 uPos; uniform vec3 uNeg; uniform float uProg; uniform float uAlpha;
varying float vA; varying vec3 vC;
void main() {
  vC = aSign > 0.0 ? uPos : uNeg;
  float visible = 1.0 - smoothstep(uProg - 0.12, uProg, aT);
  float head = exp(-pow((aT - (uProg - 0.07)) / 0.045, 2.0)) * (1.0 - step(1.12, uProg));
  vA = uAlpha * (visible * (0.01 + 0.3 * pow(aS, 1.9)) + head * 0.5 * pow(aS, 1.5));
  gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}`;
const edgeFrag = /* glsl */ `
varying float vA; varying vec3 vC;
void main() { if (vA < 0.004) discard; gl_FragColor = vec4(vC, vA); }`;

export class CircuitGL {
  private canvas = document.createElement("canvas");
  private overlay = document.createElement("div");
  private renderer: WebGLRenderer;
  private scene = new Scene();
  private camera = new PerspectiveCamera(30, 2, 0.1, 200);
  private theme: Theme = readTheme();
  private pnGroup = new Group();
  private kcGroup = new Group();
  private pnSlots = new Dots(N_PN, 1);
  private pnVals = new Dots(N_PN, 2);
  private kc = new Dots(N_KC, 3);
  private kcMarks = new Dots(256, 4);
  private pnWorld = new Float32Array(N_PN * 3);
  private kcWorld = new Float32Array(N_KC * 3);
  private edges: LineSegments;
  private edgeMat: ShaderMaterial;
  private edgeGeo = new BufferGeometry();
  private hoverEdges: LineSegments;
  private hoverGeo = new BufferGeometry();
  private edgeAttr!: { pos: BufferAttribute; t: BufferAttribute; s: BufferAttribute; sg: BufferAttribute };
  private hoverPos!: BufferAttribute;
  private edgeVariant = -1;
  private w = 0;
  private h = 0;
  private dist = 18;
  private pointer = { x: 0, y: 0, cx: 0, cy: 0 };
  private lost = false;
  private cleanup: (() => void)[] = [];
  private hover: { kind: "pn" | "kc"; index: number } | null = null;
  private lastView: StepView | null = null;

  constructor(private host: HTMLElement, private engine: Engine, private hooks: CircuitHooks, private reduced: boolean) {
    this.renderer = makeRenderer(this.canvas);
    this.canvas.setAttribute("role", "img");
    this.canvas.tabIndex = 0;
    this.overlay.className = "map-overlay";
    host.append(this.canvas, this.overlay);
    for (const [txt, x, cls] of [["PN input · 320 ESM-2 channels", (PN_X + VIEW_W / 2) / VIEW_W, "pn"], ["Kenyon cells · 2,000", (KC_X + VIEW_W / 2) / VIEW_W, "kc"], ["PN → KC · random-sparse wiring, 32 inputs per cell", 0.5, "mid"]] as const) {
      const e = document.createElement("div");
      e.className = `circuit-label ${cls}`;
      e.textContent = txt;
      e.style.left = `${100 * x}%`;
      this.overlay.append(e);
    }

    this.pnGroup.position.set(PN_X, 0, 0); this.pnGroup.rotation.y = YAW;
    this.kcGroup.position.set(KC_X, 0, 0); this.kcGroup.rotation.y = -YAW;
    this.pnGroup.add(this.pnSlots.mesh, this.pnVals.mesh);
    this.kcGroup.add(this.kc.mesh, this.kcMarks.mesh);
    this.pnGroup.updateMatrixWorld(true); this.kcGroup.updateMatrixWorld(true);
    const v = new Vector3();
    for (let i = 0; i < N_PN; i++) {
      const x = (i % PN_COLS - (PN_COLS - 1) / 2) * PN_PITCH, y = ((PN_ROWS - 1) / 2 - Math.floor(i / PN_COLS)) * PN_PITCH;
      this.pnSlots.set(i, x, y, 0, 0.24, 0, 0, 0, 0, 0);
      v.set(x, y, 0).applyMatrix4(this.pnGroup.matrixWorld);
      this.pnWorld.set([v.x, v.y, v.z], 3 * i);
    }
    for (let i = 0; i < N_KC; i++) {
      const x = (i % KC_COLS - (KC_COLS - 1) / 2) * KC_PITCH, y = ((KC_ROWS - 1) / 2 - Math.floor(i / KC_COLS)) * KC_PITCH;
      this.kc.set(i, x, y, 0, 0.1, 0, 0, 0, 1, 0);
      v.set(x, y, 0).applyMatrix4(this.kcGroup.matrixWorld);
      this.kcWorld.set([v.x, v.y, v.z], 3 * i);
    }
    this.pnSlots.count(N_PN); this.pnVals.count(N_PN); this.kc.count(N_KC);
    this.applyThemeStatic();

    // GPU buffers are allocated once (every step has exactly k_active x per_kc edges) and rewritten in place: replacing attributes would leak a buffer per step
    const nEdgeV = engine.trace.ka * engine.circ.meta.per_kc * 2;
    const dyn = (n: number, size: number) => { const a = new BufferAttribute(new Float32Array(n * size), size); a.setUsage(DynamicDrawUsage); return a; };
    this.edgeAttr = { pos: dyn(nEdgeV, 3), t: dyn(nEdgeV, 1), s: dyn(nEdgeV, 1), sg: dyn(nEdgeV, 1) };
    this.edgeGeo.setAttribute("position", this.edgeAttr.pos); this.edgeGeo.setAttribute("aT", this.edgeAttr.t); this.edgeGeo.setAttribute("aS", this.edgeAttr.s); this.edgeGeo.setAttribute("aSign", this.edgeAttr.sg);
    this.edgeGeo.setDrawRange(0, 0);
    this.hoverPos = dyn(nEdgeV, 3);
    this.hoverGeo.setAttribute("position", this.hoverPos);
    this.hoverGeo.setDrawRange(0, 0);
    this.edgeMat = new ShaderMaterial({ vertexShader: edgeVert, fragmentShader: edgeFrag, transparent: true, depthTest: false, depthWrite: false, uniforms: { uPos: { value: this.theme.fly }, uNeg: { value: this.theme.bo }, uProg: { value: 0 }, uAlpha: { value: 1 } } });
    this.edges = new LineSegments(this.edgeGeo, this.edgeMat);
    this.edges.frustumCulled = false; this.edges.renderOrder = 0;
    this.hoverEdges = new LineSegments(this.hoverGeo, new LineBasicMaterial({ color: 0x000000, transparent: true, opacity: 0.8, depthTest: false }));
    this.hoverEdges.frustumCulled = false; this.hoverEdges.renderOrder = 5;
    this.scene.add(this.edges, this.pnGroup, this.kcGroup, this.hoverEdges);

    const onMove = (ev: PointerEvent) => {
      const r = this.canvas.getBoundingClientRect();
      this.pointer.x = ((ev.clientX - r.left) / r.width) * 2 - 1;
      this.pointer.y = -(((ev.clientY - r.top) / r.height) * 2 - 1);
      this.inspect(ev);
    };
    const onLeave = () => { this.pointer.x = this.pointer.y = 0; this.setHover(null); hooks.onInspect(null); };
    const onLost = (e: Event) => { e.preventDefault(); this.lost = true; hooks.onLost(); };
    this.canvas.addEventListener("pointermove", onMove);
    this.canvas.addEventListener("pointerdown", onMove);
    this.canvas.addEventListener("pointerleave", onLeave);
    this.canvas.addEventListener("webglcontextlost", onLost);
    this.cleanup.push(() => { this.canvas.removeEventListener("pointermove", onMove); this.canvas.removeEventListener("pointerdown", onMove); this.canvas.removeEventListener("pointerleave", onLeave); this.canvas.removeEventListener("webglcontextlost", onLost); });
    this.resize();
  }

  private applyThemeStatic() {
    const T = this.theme;
    for (let i = 0; i < N_PN; i++) this.pnSlots.setStyle(i, 0.24, T.line[0], T.line[1], T.line[2], 1, 0.09);
    this.pnSlots.flush();
  }

  setTheme() {
    this.theme = readTheme();
    this.applyThemeStatic();
    this.edgeMat.uniforms.uPos.value = this.theme.fly;
    this.edgeMat.uniforms.uNeg.value = this.theme.bo;
    (this.hoverEdges.material as LineBasicMaterial).color.setRGB(...this.theme.ink);
  }

  resize() {
    const w = this.host.clientWidth, h = this.host.clientHeight;
    if (!w || !h) return;
    this.w = w; this.h = h;
    this.renderer.setSize(w, h, false);
    this.camera.aspect = w / h;
    const tan = Math.tan((this.camera.fov * Math.PI) / 360);
    this.dist = Math.max(VIEW_W / 2 / (tan * this.camera.aspect), 4.45 / tan);
    this.camera.updateProjectionMatrix();
  }

  /** rebuilds the edge batch for one variant: real wiring into every active cell, strength = |PN value| / its largest */
  private buildEdges(view: StepView) {
    const c = circuitShown(view);
    const per = this.engine.circ.meta.per_kc;
    const wiring = this.engine.wiring;
    const { pos, t, s, sg } = this.edgeAttr;
    let o = 0;
    for (const kc of c.active) {
      for (let k = 0; k < per; k++) {
        const pn = wiring[kc * per + k];
        const val = c.pnValues[pn];
        pos.array.set(this.pnWorld.subarray(3 * pn, 3 * pn + 3), 3 * o); t.array[o] = 0;
        pos.array.set(this.kcWorld.subarray(3 * kc, 3 * kc + 3), 3 * (o + 1)); t.array[o + 1] = 1;
        s.array[o] = s.array[o + 1] = Math.min(1, Math.abs(c.pn[pn]) / 127);
        sg.array[o] = sg.array[o + 1] = val >= 0 ? 1 : -1;
        o += 2;
      }
    }
    for (const a of [pos, t, s, sg]) a.needsUpdate = true;
    this.edgeGeo.setDrawRange(0, o);
    this.edgeVariant = c.variant;
  }

  private screenOf(world: Float32Array, i: number): { x: number; y: number } {
    const v = new Vector3(world[3 * i], world[3 * i + 1], world[3 * i + 2]).project(this.camera);
    return { x: ((v.x + 1) / 2) * this.w, y: ((1 - v.y) / 2) * this.h };
  }

  private inspect(ev: PointerEvent) {
    const r = this.canvas.getBoundingClientRect();
    const x = ev.clientX - r.left, y = ev.clientY - r.top;
    let best: { kind: "pn" | "kc"; index: number } | null = null, bd = 11 * 11;
    for (let i = 0; i < N_PN; i++) { const p = this.screenOf(this.pnWorld, i); const d = (p.x - x) ** 2 + (p.y - y) ** 2; if (d < bd) { bd = d; best = { kind: "pn", index: i }; } }
    bd = Math.min(bd, 7 * 7);
    for (let i = 0; i < N_KC; i++) { const p = this.screenOf(this.kcWorld, i); const d = (p.x - x) ** 2 + (p.y - y) ** 2; if (d < bd) { bd = d; best = { kind: "kc", index: i }; } }
    this.setHover(best);
    this.hooks.onInspect(best ? { ...best, clientX: ev.clientX, clientY: ev.clientY } : null);
  }

  /** highlights the wiring of the hovered unit: all 32 inputs of a Kenyon cell, or the active cells an input channel feeds */
  private setHover(h: { kind: "pn" | "kc"; index: number } | null) {
    this.hover = h;
    const view = this.lastView;
    if (!h || !view) { this.hoverGeo.setDrawRange(0, 0); return; }
    const per = this.engine.circ.meta.per_kc;
    const wiring = this.engine.wiring;
    const arr = this.hoverPos.array as Float32Array;
    let n = 0;
    const add = (pn: number, kc: number) => { if (n + 2 > arr.length / 3) return; arr.set(this.pnWorld.subarray(3 * pn, 3 * pn + 3), 3 * n); arr.set(this.kcWorld.subarray(3 * kc, 3 * kc + 3), 3 * (n + 1)); n += 2; };
    if (h.kind === "kc") for (let k = 0; k < per; k++) add(wiring[h.index * per + k], h.index);
    else for (const kc of circuitShown(view).active) for (let k = 0; k < per; k++) if (wiring[kc * per + k] === h.index) add(h.index, kc);
    this.hoverPos.needsUpdate = true;
    this.hoverGeo.setDrawRange(0, n);
  }

  render(view: StepView, reveal: Reveal, now: number, dt: number): boolean {
    void now;
    if (this.lost || !this.w) return false;
    const T = this.theme;
    const sc = T.soft;
    this.renderer.setClearColor((Math.round(sc[0] * 255) << 16) | (Math.round(sc[1] * 255) << 8) | Math.round(sc[2] * 255), 1);
    const c = circuitShown(view);
    if (c.variant !== this.edgeVariant) this.buildEdges(view);
    const stepChanged = this.lastView?.k !== view.k;
    this.lastView = view;
    if (stepChanged && this.hover) this.setHover(this.hover);
    const f = view.fly;
    const d = f.decision;
    const smooth = (a: number, b: number, x: number) => { const t = Math.max(0, Math.min(1, (x - a) / (b - a))); return t * t * (3 - 2 * t); };

    // ---- PN: the recorded input values, size = |value| relative to the variant's largest channel
    const input = reveal.input;
    for (let i = 0; i < N_PN; i++) {
      const q = c.pn[i] / 127;
      const col = q >= 0 ? T.fly : T.bo;
      this.pnVals.set(i, (i % PN_COLS - (PN_COLS - 1) / 2) * PN_PITCH, ((PN_ROWS - 1) / 2 - Math.floor(i / PN_COLS)) * PN_PITCH, 0, (0.03 + 0.2 * Math.abs(q)) * input, col[0], col[1], col[2], 0.95 * Math.min(1, input * 1.5), 0);
    }
    this.pnVals.flush();

    // ---- KC: sparse winners light up as the wave arrives; colour = learned weight (before -> after the update)
    const lit = smooth(0.72, 1, reveal.sparse);
    const wb = f.weightsBefore, wa = f.weightsAfter;
    let wmax = 1e-12;
    for (let i = 0; i < N_KC; i++) wmax = Math.max(wmax, Math.abs(wa[i]), Math.abs(wb[i]));
    const on = new Int32Array(N_KC).fill(-1);
    c.active.forEach((kc, j) => (on[kc] = j));
    const dmin = c.cutoff[0], dmax = Math.max(...c.drive);
    for (let i = 0; i < N_KC; i++) {
      const x = (i % KC_COLS - (KC_COLS - 1) / 2) * KC_PITCH, y = ((KC_ROWS - 1) / 2 - Math.floor(i / KC_COLS)) * KC_PITCH;
      if (on[i] < 0) { this.kc.set(i, x, y, 0, 0.1, T.line[0], T.line[1], T.line[2], 0.9, 0); continue; }
      const w = wb[i] + (wa[i] - wb[i]) * reveal.update;
      const t = Math.tanh((2.2 * w) / wmax);
      const base: RGB = Math.abs(w) < 1e-15 ? T.ink : t > 0 ? T.fly : T.bo;
      const col = mix(T.line, base, lit);
      const a = Math.abs(w) < 1e-15 ? 0.55 : 0.4 + 0.6 * Math.abs(t);
      const rel = dmax > dmin ? (c.drive[on[i]] - dmin) / (dmax - dmin) : 0.5;
      this.kc.set(i, x, y, 0, 0.1 + lit * (0.07 + 0.06 * rel), col[0], col[1], col[2], 1 + (a - 1) * lit, 0);
    }
    this.kc.flush();

    // ---- the recorded update: +a on the cells only the new code has (raised), -a on the cells only the old code has (lowered)
    let m = 0;
    if (d && reveal.update > 0) {
      for (const kc of d.raised) if (m < 256) this.kcMarks.set(m++, (kc % KC_COLS - (KC_COLS - 1) / 2) * KC_PITCH, ((KC_ROWS - 1) / 2 - Math.floor(kc / KC_COLS)) * KC_PITCH, 0, 0.27, T.accent[0], T.accent[1], T.accent[2], reveal.update, 0.14);
      for (const kc of d.lowered) if (m < 256) this.kcMarks.set(m++, (kc % KC_COLS - (KC_COLS - 1) / 2) * KC_PITCH, ((KC_ROWS - 1) / 2 - Math.floor(kc / KC_COLS)) * KC_PITCH, 0, 0.27, T.muted[0], T.muted[1], T.muted[2], reveal.update, 0.14);
    }
    this.kcMarks.count(m); this.kcMarks.flush();

    this.edgeMat.uniforms.uProg.value = reveal.sparse * 1.15;
    this.edgeMat.uniforms.uAlpha.value = Math.min(1, reveal.input * 2);

    // ---- camera: fixed, with a slight pointer parallax for depth (off under reduced motion)
    const k = 1 - Math.exp(-Math.min(dt, 64) / 220);
    if (!this.reduced) { this.pointer.cx += (this.pointer.x - this.pointer.cx) * k; this.pointer.cy += (this.pointer.y - this.pointer.cy) * k; }
    this.camera.position.set(this.pointer.cx * 0.9, this.pointer.cy * 0.5, this.dist);
    this.camera.lookAt(0, 0, 0);
    this.renderer.render(this.scene, this.camera);
    return !reveal.settled || Math.abs(this.pointer.x - this.pointer.cx) + Math.abs(this.pointer.y - this.pointer.cy) > 0.003;
  }

  dispose() {
    for (const c of this.cleanup) c();
    for (const o of [this.pnSlots, this.pnVals, this.kc, this.kcMarks]) o.dispose();
    this.edgeGeo.dispose(); this.hoverGeo.dispose(); this.edgeMat.dispose();
    this.renderer.dispose();
    this.canvas.remove(); this.overlay.remove();
  }
}
