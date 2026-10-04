/** A small procedural Drosophila, used as the navigation marker on the landscape. No external asset: low-poly primitives only
 *  (thorax, head, banded abdomen, red eyes, two veined translucent wings, six folded legs). Length along +x is 1 unit; the
 *  landscape scales it to a few tens of CSS pixels. It is a marker for where the learner is, not a result. */
import {
  BufferGeometry, Color, DoubleSide, Float32BufferAttribute, Group, LineBasicMaterial, LineSegments, Mesh, MeshStandardMaterial, Shape, ShapeGeometry, SphereGeometry,
} from "three";

export interface FlyPose {
  x: number;
  y: number;
  /** radians, 0 = facing +x (map right), counter-clockwise */
  heading: number;
  /** 0 = wings folded and still, 1 = full wingbeat */
  intensity: number;
  /** roll about the body axis, radians */
  bank: number;
  timeMs: number;
}

export interface FlyHandle {
  group: Group;
  update(p: FlyPose, sizePx: number): void;
  dispose(): void;
}

const BODY = new Color("#c2975d");
const BAND = new Color("#5d4630");
const HEAD = new Color("#a67c48");
const EYE = new Color("#b83a42");

function ellipsoid(rx: number, ry: number, rz: number, color: Color, bands = false) {
  const g = new SphereGeometry(1, 22, 14);
  g.scale(rx, ry, rz);
  const pos = g.getAttribute("position");
  const col = new Float32Array(pos.count * 3);
  for (let i = 0; i < pos.count; i++) {
    const t = (pos.getX(i) / rx + 1) / 2;
    const c = bands && Math.floor(t * 6) % 2 === 1 && t > 0.12 ? BAND : color;
    col[3 * i] = c.r; col[3 * i + 1] = c.g; col[3 * i + 2] = c.b;
  }
  g.setAttribute("color", new Float32BufferAttribute(col, 3));
  return g;
}

export function createFly(): FlyHandle {
  const root = new Group();
  const rig = new Group(); // rolls about the body axis
  root.add(rig);
  const disposables: { dispose(): void }[] = [];
  const body = new MeshStandardMaterial({ vertexColors: true, roughness: 0.55, metalness: 0.04, transparent: true });
  const eyeMat = new MeshStandardMaterial({ color: EYE, roughness: 0.25, metalness: 0.1, transparent: true });
  const wingMat = new MeshStandardMaterial({ color: "#e6eef4", transparent: true, opacity: 0.5, side: DoubleSide, depthWrite: false, roughness: 0.2 });
  const veinMat = new LineBasicMaterial({ color: "#4a5560", transparent: true, opacity: 0.7, depthWrite: false });
  const legMat = new LineBasicMaterial({ color: "#2d241b", transparent: true });
  disposables.push(body, eyeMat, wingMat, veinMat, legMat);

  const part = (g: BufferGeometry, m: MeshStandardMaterial, x: number, y = 0, z = 0) => {
    const mesh = new Mesh(g, m);
    mesh.position.set(x, y, z);
    mesh.renderOrder = 20;
    rig.add(mesh);
    disposables.push(g);
    return mesh;
  };
  part(ellipsoid(0.19, 0.15, 0.14, BODY), body, 0.02); // thorax
  part(ellipsoid(0.115, 0.13, 0.115, HEAD), body, 0.25, 0, 0.005); // head
  part(ellipsoid(0.2, 0.135, 0.12, BODY, true), body, -0.3, 0, -0.01).scale.set(1.28, 1, 1); // abdomen with dark bands
  for (const s of [1, -1]) part(new SphereGeometry(0.07, 14, 10), eyeMat, 0.285, s * 0.095, 0.03);

  // legs: hip -> knee -> foot, three pairs, folded close to the body
  const legPts: number[] = [];
  const leg = (s: number, hx: number, kx: number, ky: number, fx: number, fy: number) => {
    legPts.push(hx, s * 0.05, -0.06, kx, s * ky, -0.09, kx, s * ky, -0.09, fx, s * fy, -0.12);
  };
  for (const s of [1, -1]) { leg(s, 0.15, 0.26, 0.17, 0.33, 0.24); leg(s, 0.03, 0.05, 0.22, 0.02, 0.31); leg(s, -0.1, -0.2, 0.19, -0.3, 0.25); }
  const legGeo = new BufferGeometry();
  legGeo.setAttribute("position", new Float32BufferAttribute(legPts, 3));
  const legs = new LineSegments(legGeo, legMat);
  legs.renderOrder = 19;
  rig.add(legs);
  disposables.push(legGeo);

  // wings: a teardrop from the hinge, swept back (-x); veins as thin lines
  const shape = new Shape();
  shape.moveTo(0, 0);
  shape.bezierCurveTo(-0.12, 0.11, -0.52, 0.21, -0.66, 0.07);
  shape.bezierCurveTo(-0.7, -0.01, -0.5, -0.12, -0.18, -0.07);
  shape.lineTo(0, 0);
  const wingGeo = new ShapeGeometry(shape, 14);
  const veinGeo = new BufferGeometry();
  veinGeo.setAttribute("position", new Float32BufferAttribute([0, 0, 0, -0.62, 0.075, 0, -0.1, 0.02, 0, -0.5, 0.13, 0, -0.12, -0.01, 0, -0.52, -0.04, 0, -0.24, 0.05, 0, -0.3, -0.06, 0], 3));
  disposables.push(wingGeo, veinGeo);
  const wings = [1, -1].map((s) => {
    const pivot = new Group();
    pivot.position.set(0.03, s * 0.11, 0.15);
    const holder = new Group();
    holder.scale.y = s; // mirror the right wing
    const m = new Mesh(wingGeo, wingMat);
    m.renderOrder = 22;
    const v = new LineSegments(veinGeo, veinMat);
    v.renderOrder = 23;
    holder.add(m, v);
    pivot.add(holder);
    rig.add(pivot);
    return { pivot, s };
  });

  return {
    group: root,
    update(p, sizePx) {
      root.position.set(p.x, p.y, 20);
      root.scale.setScalar(sizePx);
      root.rotation.z = p.heading;
      rig.rotation.x = p.bank;
      const w = p.timeMs * 2 * Math.PI * 0.0085; // stylised ~8.5 Hz so the beat is visible at 60 fps
      const beat = Math.sin(w) * p.intensity;
      const spread = 0.2 + 0.78 * p.intensity + 0.1 * beat;
      for (const { pivot, s } of wings) {
        pivot.rotation.z = -s * spread;
        pivot.rotation.x = s * 0.75 * beat;
      }
      rig.scale.setScalar(1 + 0.012 * Math.sin(p.timeMs * 0.006) * p.intensity);
    },
    dispose() { for (const d of disposables) d.dispose(); },
  };
}
