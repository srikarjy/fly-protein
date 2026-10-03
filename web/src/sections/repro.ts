import { esc } from "../lib/charts";
import { $ } from "../lib/dom";

/* eslint-disable @typescript-eslint/no-explicit-any */
export function renderRepro(m: any) {
  const short = (h: string) => `<code title="${esc(h)}">${esc(h.slice(0, 12))}…</code>`;
  const assays = m.assays.map((a: any) => `<tr><th scope="row"><code>${esc(a.id)}</code></th><td>${a.n_variants.toLocaleString()}</td><td>${a.wt_length}</td><td>${short(a.embedding_sha256)}</td></tr>`).join("");
  const models = Object.entries(m.models).map(([k, v]: [string, any]) => `<tr><th scope="row"><code>${esc(v.repo)}</code></th><td>${v.revision ? short(v.revision) : "n/a"}</td></tr>`).join("");
  const cfgs = Object.entries(m.configs).map(([k, v]: [string, any]) => `<tr><th scope="row"><code>${esc(k)}</code></th><td>${short(v.sha256)}</td><td class="cfg">${esc(JSON.stringify(v.config))}</td></tr>`).join("");
  const arts = Object.entries(m.artifacts).map(([k, v]: [string, any]) => `<tr><th scope="row"><code>${esc(k)}</code></th><td>${short(v.sha256)}</td><td>${(v.bytes / 1e3).toFixed(0)} KB</td></tr>`).join("");
  $("#repro").innerHTML = `
  <div class="two">
    <div><h4>Design</h4><ul class="plain"><li><b>Dataset:</b> ${esc(m.dataset.name)} (<code>${esc(m.dataset.repo)}</code>, revision ${m.dataset.revision ? short(m.dataset.revision) : "n/a"}).</li><li><b>Seeds:</b> ${m.seeds[0]}–${m.seeds[m.seeds.length - 1]} (${m.seeds.length}). <b>Budgets:</b> ${m.budgets.join(", ")} measured variants.</li><li><b>Start and RNG:</b> <code>${esc(m.start_rule)}</code></li><li><b>Environment:</b> Python ${esc(m.environment.python)}; ${Object.entries(m.environment.packages).filter(([, v]) => v).map(([k, v]) => `${esc(k)} ${esc(String(v))}`).join(", ")}.</li></ul></div>
    <div><h4>Model revisions</h4><div class="tablewrap"><table><thead><tr><th>checkpoint</th><th>revision</th></tr></thead><tbody>${models}</tbody></table></div></div>
  </div>
  <h4>Assays (embedding hash = SHA-256 of the cached ESM-2 8M embedding matrix used by the search methods)</h4>
  <div class="tablewrap"><table><thead><tr><th>ProteinGym assay id</th><th>variants</th><th>WT length</th><th>embedding SHA-256</th></tr></thead><tbody>${assays}</tbody></table></div>
  <details><summary>Method configurations and their hashes</summary><div class="tablewrap"><table><thead><tr><th>method</th><th>config SHA-256</th><th>config</th></tr></thead><tbody>${cfgs}</tbody></table></div></details>
  <details><summary>Result artifacts (SHA-256)</summary><div class="tablewrap"><table><thead><tr><th>file</th><th>SHA-256</th><th>size</th></tr></thead><tbody>${arts}</tbody></table></div></details>
  <h4>Reproduce</h4>
  <pre class="code"><code>python -m venv .venv &amp;&amp; source .venv/bin/activate
pip install -e ".[esm,connectome,analysis,dev]"
# ProteinGym parquet shards -&gt; data/proteingym/DMS_substitutions/
python scripts/embed_assays.py                 # ESM-2 8M embeddings (CPU)
python scripts/prepare_benchmark_data.py
python scripts/zeroshot_scores.py --model facebook/esm2_t33_650M_UR50D --mode wt
python scripts/run_benchmark.py --seeds 20 --budget 500
python scripts/analyze_benchmark.py            # tables, plots, summary.json
python scripts/verify_replay.py                # deterministic replay vs queries.npz
pytest &amp;&amp; (cd web &amp;&amp; npm ci &amp;&amp; npm test &amp;&amp; npm run build)</code></pre>
  <p class="foot">Full manifest: <code>results/MANIFEST.json</code>. Deterministic replay: <code>${esc(m.replay)}</code>.</p>`;
}
