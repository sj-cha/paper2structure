# Synthetic input fixtures

**These are hand-written test inputs, not data extracted from real publications.** Each scenario
stands in for the outputs of the Summarizer and the characterization agents, so that the Planner,
the Constructor and the Reviewer can run without a PDF. No PDF is behind them: `source_pdf` is `synthetic`, the titles are
prefixed `[SYNTHETIC FIXTURE]`, the paper summaries and the characterization reports say so, and
every observation is a statement of the (synthetic) text. The values are chosen to be chemically
plausible, but they must not be cited as experimental results.

Each scenario folder has the files a PDF run writes to `paper/paper_graph.json` and
`characterization/`, in the same formats:

| file | format | content |
|---|---|---|
| `paper_graph.json` | `PaperGraph` (the Summarizer's output) | metadata, the material systems (`S1`, ...) and text evidence items (`E1`, `E2`, ...) tagged by technique; no figures. Older `nodes`/`edges` fields are ignored on loading |
| `characterization/<TECHNIQUE>.json` | `CharacterizationReport` (a characterization agent's output) | one per technique: observations (`O1`, ...) citing the evidence IDs, findings, structural constraints and caveats, including what the paper does not provide |

Because they are hand-written, the reports may cover techniques that have no characterization
agent (`LEED`). The planner cites them as evidence IDs (`E2`) and
technique-qualified observation IDs (`FTIR:O1`).

The campaigns `config/<scenario>.yml` load a scenario through `inputs.paper_graph` and
`inputs.characterization`, so the Summarizer and the characterization agents do not run; the Planner,
the Constructor and the Reviewer run as usual:

```bash
python main.py --config config/pt111_co.yml --random_seed 0
```

| scenario | what it exercises |
|---|---|
| `pt111_co/` | metal slab; lattice constant given by "XRD"; atop vs bridge CO as competing hypotheses |
| `cspbbr3_phosphonate/` | halide-perovskite nanocubes; slab vs nanocrystal model choice; phosphonate replacing surface Br |
| `anatase_formate/` | oxide with **no reported lattice constant** (literature fallback); bidentate bridging formate on Ti5c pairs |
| `cdse_zns_coreshell/` | a model **no toolkit builder covers**: coherent core/shell quantum dot (CdSe core, ~2 ML ZnS shell) with carboxylate adsorbates; tests the agent's own code |
