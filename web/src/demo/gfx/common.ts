/** Renderer-independent graphics helpers and interfaces (no three.js import, so the Canvas 2D fallback stays small). */
import { interpolateViridis } from "d3-scale-chromatic";
import type { Reveal } from "../choreo";
import type { StepView } from "../state";
import { parseColor, type RGB } from "./theme";

export const viridisRGB = (t: number): RGB => parseColor(interpolateViridis(Math.max(0, Math.min(1, t))));

/** true if a WebGL2 context can be created here (checked once, on a throwaway canvas) */
export function webglAvailable(): boolean {
  try {
    const c = document.createElement("canvas");
    const gl = c.getContext("webgl2");
    const ok = !!gl;
    gl?.getExtension("WEBGL_lose_context")?.loseContext();
    return ok;
  } catch {
    return false;
  }
}

export interface LandscapeRenderer {
  resize(): void;
  /** draws the frame; returns true while it still needs animation frames */
  render(view: StepView, reveal: Reveal, nowMs: number, dtMs: number): boolean;
  setTheme(): void;
  dispose(): void;
}

export interface InspectHit { variant: number; clientX: number; clientY: number }
export interface LandscapeHooks { onInspect(hit: InspectHit | null): void; onLost(): void }
