/** Shared three.js setup. Colours are used as plain sRGB numbers everywhere (no colour-space conversion), so the scenes match the CSS tokens. */
import { AmbientLight, ColorManagement, DirectionalLight, LinearSRGBColorSpace, WebGLRenderer } from "three";

export function makeRenderer(canvas: HTMLCanvasElement): WebGLRenderer {
  ColorManagement.enabled = false;
  const r = new WebGLRenderer({ canvas, antialias: true, alpha: false });
  r.outputColorSpace = LinearSRGBColorSpace;
  r.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  return r;
}

export function lights() {
  const amb = new AmbientLight(0xffffff, 1.15);
  const key = new DirectionalLight(0xffffff, 2.4);
  key.position.set(-0.6, 0.8, 1.2);
  return [amb, key];
}
