# Synthetic input fixtures

**These are hand-written test inputs, not data extracted from real publications.** They stand in
for the outputs of Agent 1 (paper summary) and Agent 2 (experimental evidence) until those agents
exist. Every file sets `"synthetic": true`, and the titles are prefixed `[SYNTHETIC FIXTURE]`. The
values are chosen to be chemically plausible, but they must not be cited as experimental results.

| scenario | what it exercises |
|---|---|
| `pt111_co/` | metal slab; lattice constant given by "XRD"; atop vs bridge CO as competing hypotheses |
| `cspbbr3_phosphonate/` | halide-perovskite nanocubes; slab vs nanocrystal model choice; phosphonate replacing surface Br |
| `anatase_formate/` | oxide with **no reported lattice constant** (literature fallback); bidentate bridging formate on Ti5c pairs |
| `cdse_zns_coreshell/` | a model **no toolkit builder covers**: coherent core/shell quantum dot (CdSe core, ~2 ML ZnS shell) with carboxylate adsorbates; tests the agent's own code |
