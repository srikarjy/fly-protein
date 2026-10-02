// Evaluation table on the real landscape: npm run eval
// 50 random starts x 200 steps per row, averaged over 5 seeds (each seed = different starts).
import { readFileSync } from "node:fs";
import { DEFAULTS, type Mode } from "../src/fly/agent";
import { evaluate } from "../src/fly/evaluate";
import { parseLandscape, quantile } from "../src/fly/landscape";

const L = parseLandscape(JSON.parse(readFileSync(new URL("../public/data.json", import.meta.url), "utf8")));
console.log(`${L.assay}: n=${L.n}  map median=${quantile(L.fitness, 0.5).toFixed(3)}  top10% >= ${quantile(L.fitness, 0.9).toFixed(3)}  top1% >= ${quantile(L.fitness, 0.99).toFixed(3)}`);

const rows: [Mode, string, boolean][] = [
  ["random", "random walk", false],
  ["chemotaxis", "chemotaxis (sees neighbor fitness)", false],
  ["mushroom", "mushroom body, fresh each start", false],
  ["mushroom", "mushroom body, weights carried over", true],
];
const seeds = [1, 2, 3, 4, 5];
const mean = (a: number[]) => a.reduce((x, y) => x + y, 0) / a.length;
console.log("row".padEnd(38), "median best", " top10% reached / median steps", " top1% reached / median steps");
for (const [mode, label, carryOver] of rows) {
  const at = (topFrac: number) => seeds.map((seed) => evaluate(L, mode, label, DEFAULTS[mode], { seed, topFrac, carryOver }));
  const a = at(0.1);
  const b = at(0.01);
  const steps = (r: ReturnType<typeof evaluate>[]) => {
    const v = r.map((x) => x.medianStepsToTop).filter((x): x is number => x !== null);
    return v.length === r.length ? mean(v).toFixed(0) : "-";
  };
  console.log(
    label.padEnd(38),
    mean(a.map((r) => r.medianBest)).toFixed(3).padStart(11),
    `${(mean(a.map((r) => r.reachRate)) * 100).toFixed(0)}%`.padStart(12), steps(a).padStart(6),
    `${(mean(b.map((r) => r.reachRate)) * 100).toFixed(0)}%`.padStart(16), steps(b).padStart(6),
  );
}
