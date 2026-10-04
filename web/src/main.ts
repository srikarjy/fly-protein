import "./style.css";
import { $, getJSON, onVisible } from "./lib/dom";
import { renderCards, renderFigure, type Hero } from "./sections/hero";

const fail = (id: string, e: unknown) => {
  const el = document.querySelector(id);
  if (el) el.innerHTML = `<p class="error">Could not load this section (${String(e)}). Reload the page; the figures are generated from static data files.</p>`;
};

getJSON<Hero>("hero.json").then((h) => { renderCards(h); renderFigure(h); }).catch((e) => { fail("#cards", e); fail("#fig-main", e); });

const loadDemo = () => import("./demo/demo").then((m) => m.initDemo()).catch((e) => fail("#demo-body", e));
if (/^#demo-\d+$/.test(location.hash)) loadDemo(); // a deep link opens the replay at once, not when it is scrolled into view
else onVisible($("#demo"), loadDemo, "300px");

let results: Promise<void> | null = null;
const loadResults = () => (results ??= Promise.all([getJSON("results.json"), import("./sections/results")]).then(([R, m]) => m.renderResults(R)).catch((e) => { fail("#bench-body", e); fail("#find-list", e); }));
onVisible($("#results"), loadResults, "800px");
onVisible($("#found"), loadResults, "800px");
onVisible($("#reproduce"), () => Promise.all([getJSON("manifest.json"), import("./sections/repro")]).then(([m, r]) => r.renderRepro(m)).catch((e) => fail("#repro", e)), "600px");
