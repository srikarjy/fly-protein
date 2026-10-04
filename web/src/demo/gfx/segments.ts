/** Instanced anti-aliased line segments with a width in the parent's local units (CSS px in the map scenes). */
import { DynamicDrawUsage, Float32BufferAttribute, InstancedBufferAttribute, InstancedBufferGeometry, Mesh, ShaderMaterial, Uint16BufferAttribute } from "three";

const vertexShader = /* glsl */ `
attribute vec3 aA; attribute vec3 aB; attribute float aWidth; attribute vec4 aColor;
varying float vY; varying vec4 vColor;
void main() {
  vec2 d = aB.xy - aA.xy;
  float len = max(length(d), 1e-5);
  vec2 dir = d / len;
  vec2 n = vec2(-dir.y, dir.x);
  vec2 p = mix(aA.xy, aB.xy, position.x + 0.5) + n * position.y * (aWidth + 1.0) + dir * position.x * 1.0;
  vY = position.y * 2.0; vColor = aColor;
  gl_Position = projectionMatrix * modelViewMatrix * vec4(p, mix(aA.z, aB.z, position.x + 0.5), 1.0);
}`;

const fragmentShader = /* glsl */ `
varying float vY; varying vec4 vColor;
void main() {
  float fw = max(fwidth(vY), 1e-4);
  float a = 1.0 - smoothstep(1.0 - fw, 1.0, abs(vY));
  float alpha = vColor.a * a;
  if (alpha < 0.004) discard;
  gl_FragColor = vec4(vColor.rgb, alpha);
}`;

export class Segments {
  readonly mesh: Mesh;
  private geo = new InstancedBufferGeometry();
  private a: Float32Array; private b: Float32Array; private w: Float32Array; private col: Float32Array;
  private attrs: InstancedBufferAttribute[] = [];

  constructor(readonly capacity: number, renderOrder = 0) {
    this.geo.setAttribute("position", new Float32BufferAttribute([-0.5, -0.5, 0, 0.5, -0.5, 0, 0.5, 0.5, 0, -0.5, 0.5, 0], 3));
    this.geo.setIndex(new Uint16BufferAttribute([0, 1, 2, 0, 2, 3], 1));
    const mk = (arr: Float32Array, n: number, name: string) => {
      const at = new InstancedBufferAttribute(arr, n);
      at.setUsage(DynamicDrawUsage);
      this.geo.setAttribute(name, at);
      this.attrs.push(at);
      return arr;
    };
    this.a = mk(new Float32Array(capacity * 3), 3, "aA");
    this.b = mk(new Float32Array(capacity * 3), 3, "aB");
    this.w = mk(new Float32Array(capacity), 1, "aWidth");
    this.col = mk(new Float32Array(capacity * 4), 4, "aColor");
    this.geo.instanceCount = 0;
    this.mesh = new Mesh(this.geo, new ShaderMaterial({ vertexShader, fragmentShader, transparent: true, depthTest: false, depthWrite: false }));
    this.mesh.frustumCulled = false;
    this.mesh.renderOrder = renderOrder;
  }

  set(i: number, ax: number, ay: number, bx: number, by: number, width: number, r: number, g: number, b: number, a: number, z = 0) {
    this.a[3 * i] = ax; this.a[3 * i + 1] = ay; this.a[3 * i + 2] = z;
    this.b[3 * i] = bx; this.b[3 * i + 1] = by; this.b[3 * i + 2] = z;
    this.w[i] = width;
    this.col[4 * i] = r; this.col[4 * i + 1] = g; this.col[4 * i + 2] = b; this.col[4 * i + 3] = a;
  }

  count(n: number) { this.geo.instanceCount = Math.min(n, this.capacity); }
  flush() { for (const a of this.attrs) a.needsUpdate = true; }
  dispose() { this.geo.dispose(); (this.mesh.material as ShaderMaterial).dispose(); }
}
