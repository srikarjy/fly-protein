/** Reads the page's CSS colour tokens as linear-agnostic RGB triples (0..1) so WebGL and Canvas renderers follow the light/dark theme. */
export type RGB = readonly [number, number, number];

export interface Theme {
  ink: RGB; bg: RGB; panel: RGB; muted: RGB; line: RGB; soft: RGB; accent: RGB; fly: RGB; bo: RGB; top1: RGB; kcOff: RGB; zs: RGB;
}

export function parseColor(s: string): RGB {
  s = s.trim();
  const hex = /^#([0-9a-f]{3}|[0-9a-f]{6})$/i.exec(s);
  if (hex) {
    let h = hex[1];
    if (h.length === 3) h = [...h].map((c) => c + c).join("");
    return [parseInt(h.slice(0, 2), 16) / 255, parseInt(h.slice(2, 4), 16) / 255, parseInt(h.slice(4, 6), 16) / 255];
  }
  const rgb = /rgba?\(\s*([\d.]+)[ ,]+([\d.]+)[ ,]+([\d.]+)/.exec(s);
  if (rgb) return [+rgb[1] / 255, +rgb[2] / 255, +rgb[3] / 255];
  return [0.5, 0.5, 0.5];
}

export function readTheme(): Theme {
  const cs = getComputedStyle(document.documentElement);
  const c = (n: string) => parseColor(cs.getPropertyValue(n));
  return { ink: c("--ink"), bg: c("--bg"), panel: c("--panel"), muted: c("--muted"), line: c("--line"), soft: c("--soft"), accent: c("--accent"), fly: c("--c-fly"), bo: c("--c-bo"), top1: c("--c-top1"), kcOff: c("--kc-off"), zs: c("--c-zs") };
}

export function onThemeChange(fn: () => void): () => void {
  const mq = window.matchMedia("(prefers-color-scheme: dark)");
  mq.addEventListener("change", fn);
  return () => mq.removeEventListener("change", fn);
}

export const mix = (a: RGB, b: RGB, t: number): RGB => [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t];
