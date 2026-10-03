export const $ = <T extends HTMLElement>(sel: string, root: ParentNode = document) => root.querySelector(sel) as T;
export async function getJSON<T>(name: string): Promise<T> {
  const r = await fetch(`${import.meta.env.BASE_URL}data/${name}`);
  if (!r.ok) throw new Error(`${name}: ${r.status}`);
  return r.json() as Promise<T>;
}
export function onVisible(el: Element, fn: () => void, margin = "500px") {
  if (!("IntersectionObserver" in window)) return fn();
  const io = new IntersectionObserver((es) => { if (es.some((e) => e.isIntersecting)) { io.disconnect(); fn(); } }, { rootMargin: margin });
  io.observe(el);
}
export const reducedMotion = () => window.matchMedia("(prefers-reduced-motion: reduce)").matches;
