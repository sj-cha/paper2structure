# Synthetic input fixtures

**These are hand-written test inputs, not data extracted from real publications.** Each scenario
stands in for the outputs of Agent 1 (summarizer) and Agent 2 (characterization), so that Agents
3-5 can run without a PDF. No PDF is behind them: `source_pdf` is `synthetic`, the titles are
prefixed `[SYNTHETIC FIXTURE]`, the paper summaries and the characterization reports say so, and
every observation is a statement of the (synthetic) text. The values are chosen to be chemically
plausible, but they must not be cited as experimental results.

Each scenario folder has the files a PDF run writes to `paper/paper_graph.json` and
`characterization/`, in the same formats:

| file | format | content |
|---|---|---|
| `paper_graph.json` | `PaperGraph` (Agent 1) | metadata, the material systems (`S1`, ...), text evidence items (`E1`, `E2`, ...) tagged by technique, and the knowledge-graph nodes and edges; no figures |
| `characterization/<TECHNIQUE>.json` | `CharacterizationReport` (Agent 2) | one per technique: observations (`O1`, ...) citing the evidence IDs, findings, structural constraints and caveats, including what the paper does not provide |

Because they are hand-written, the reports may cover techniques that have no characterization
agent (`LEED`) or only a stub spec (`XPS`). The planner cites them as evidence IDs (`E2`) and
technique-qualified observation IDs (`FTIR:O1`).

The campaigns `config/<scenario>.yml` load a scenario through `inputs.paper_graph` and
`inputs.characterization`, so Agents 1-2 do not run; Agents 3-5 run as usual:

```bash
python main.py --config config/pt111_co.yml --random_seed 0
```

| scenario | what it exercises |
|---|---|
| `pt111_co/` | metal slab; lattice constant given by "XRD"; atop vs bridge CO as competing hypotheses |
| `cspbbr3_phosphonate/` | halide-perovskite nanocubes; slab vs nanocrystal model choice; phosphonate replacing surface Br |
| `anatase_formate/` | oxide with **no reported lattice constant** (literature fallback); bidentate bridging formate on Ti5c pairs |
| `cdse_zns_coreshell/` | a model **no toolkit builder covers**: coherent core/shell quantum dot (CdSe core, ~2 ML ZnS shell) with carboxylate adsorbates; tests the agent's own code |
