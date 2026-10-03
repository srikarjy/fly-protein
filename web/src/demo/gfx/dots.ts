/** Instanced, anti-aliased dots and rings: thousands of marks in one draw call, sized in the parent's local units (CSS px in the map scenes). */
import { DynamicDrawUsage, Float32BufferAttribute, InstancedBufferAttribute, InstancedBufferGeometry, Mesh, ShaderMaterial, Uint16BufferAttribute } from "three";

const vertexShader = /* glsl */ `
attribute vec3 aOffset; attribute float aSize; attribute vec4 aColor; attribute float aRing;
varying vec2 vUv; varying vec4 vColor; varying float vRing;
void main() {
  vUv = position.xy * 2.0; vColor = aColor; vRing = aRing;
  gl_Position = projectionMatrix * modelViewMatrix * vec4(aOffset.xy + position.xy * aSize, aOffset.z, 1.0);
}`;

// aRing: 0 = filled disc, (0,1] = ring whose width is that fraction of the radius, -1 = soft radial falloff (shadow / glow)
const fragmentShader = /* glsl */ `
varying vec2 vUv; varying vec4 vColor; varying float vRing;
void main() {
  float d = length(vUv);
  float fw = max(fwidth(d), 1e-4);
  float a = 1.0 - smoothstep(1.0 - fw, 1.0, d);
  if (vRing > 0.0) a *= smoothstep(1.0 - vRing - fw, 1.0 - vRing, d);
  else if (vRing < 0.0) a = pow(max(0.0, 1.0 - d), 2.0);
  float alpha = vColor.a * a;
  if (alpha < 0.004) discard;
  gl_FragColor = vec4(vColor.rgb, alpha);
}`;

export class Dots {
  readonly mesh: Mesh;
  private geo = new InstancedBufferGeometry();
  private off: Float32Array; private size: Float32Array; private col: Float32Array; private ring: Float32Array;
  private attrs: InstancedBufferAttribute[] = [];

  constructor(readonly capacity: number, renderOrder = 0) {
    this.geo.setAttribute("position", new Float32BufferAttribute([-0.5, -0.5, 0, 0.5, -0.5, 0, 0.5, 0.5, 0, -0.5, 0.5, 0], 3));
    this.geo.setIndex(new Uint16BufferAttribute([0, 1, 2, 0, 2, 3], 1));
    const mk = (arr: Float32Array, n: number, name: string) => {
      const a = new InstancedBufferAttribute(arr, n);
      a.setUsage(DynamicDrawUsage);
      this.geo.setAttribute(name, a);
      this.attrs.push(a);
      return arr;
    };
    this.off = mk(new Float32Array(capacity * 3), 3, "aOffset");
    this.size = mk(new Float32Array(capacity), 1, "aSize");
    this.col = mk(new Float32Array(capacity * 4), 4, "aColor");
    this.ring = mk(new Float32Array(capacity), 1, "aRing");
    this.geo.instanceCount = 0;
    this.mesh = new Mesh(this.geo, new ShaderMaterial({ vertexShader, fragmentShader, transparent: true, depthTest: false, depthWrite: false }));
    this.mesh.frustumCulled = false;
    this.mesh.renderOrder = renderOrder;
  }

  set(i: number, x: number, y: number, z: number, size: number, r: number, g: number, b: number, a: number, ring = 0) {
    this.off[3 * i] = x; this.off[3 * i + 1] = y; this.off[3 * i + 2] = z;
    this.size[i] = size;
    this.col[4 * i] = r; this.col[4 * i + 1] = g; this.col[4 * i + 2] = b; this.col[4 * i + 3] = a;
    this.ring[i] = ring;
  }

  setPos(i: number, x: number, y: number, z = 0) { this.off[3 * i] = x; this.off[3 * i + 1] = y; this.off[3 * i + 2] = z; }
  setStyle(i: number, size: number, r: number, g: number, b: number, a: number, ring = 0) {
    this.size[i] = size;
    this.col[4 * i] = r; this.col[4 * i + 1] = g; this.col[4 * i + 2] = b; this.col[4 * i + 3] = a;
    this.ring[i] = ring;
  }

  /** draw only the first n instances */
  count(n: number) { this.geo.instanceCount = Math.min(n, this.capacity); }
  flush() { for (const a of this.attrs) a.needsUpdate = true; }
  dispose() { this.geo.dispose(); (this.mesh.material as ShaderMaterial).dispose(); }
}
