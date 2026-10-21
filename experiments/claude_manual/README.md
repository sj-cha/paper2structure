# Claude walkthrough: the summarizer flow by hand (no Pydantic, no API)

The goal of this folder is to **understand the flow** before trusting the automated version.
Claude (in Claude Code) played the role of each LLM step by hand on one paper:

> A. Shahee, D. Kumar, N. P. Lalla, *Structural studies on Multiferroic La1-xBixCrO3 Perovskites*,
> AIP Conf. Proc. 1349, 1243 (2011), [arXiv:1403.3526](https://arxiv.org/abs/1403.3526).
> PDF stored outside the repo at `hackathon/papers/La1-xBixCrO3_arXiv1403.3526.pdf`.

It is a 2-page paper, but it covers what the pipeline has to handle: a composition series
(5 systems), XRD + Rietveld refinement + TEM + electron diffraction, and key numbers that appear
**only in a figure**.

## The flow, one step at a time

```
PDF ─► [0] read PDF ─► [1] find systems ─► [2] text evidence (per system) ─► [3] figures (per figure) ─► [4] build graph
         no LLM           LLM                LLM ×5                            vision LLM ×4                no LLM
```

### Step 0: read the PDF (plain code, no LLM)
`pdf_reader.py` extracts the text of each page and cuts out figure regions as PNGs.
What we learned from this paper:
- Fig. 1 and Fig. 2 are **vector drawings**, not embedded images. Their axis tick labels show up
  as junk text ("2 0  4 0  6 0 …"), so the reader drops text that lies inside a figure region.
- A sentence like "Fig.1 (a) shows the XRD patterns…" looks like a caption but is body text.
  The reader skips those so each figure gets its real caption.

Result: 4 figure images (`output/figures/`), each with its caption.

### Step 1: how many material systems? → [`step1_find_systems.json`](step1_find_systems.json)
Question asked: *"List every distinct material that was evaluated."*
Answer: **5 systems**, one per composition: LaCrO3 (x=0), La0.8Bi0.2CrO3, La0.75Bi0.25CrO3,
La0.7Bi0.3CrO3, La0.65Bi0.35CrO3. They get IDs S1-S5, which every later step uses.

Judgement calls an LLM must get right:
- La2O3, Cr2O3, Bi2O3 are **precursors**, not systems.
- Bi2O3 found on the pellet surface is an **impurity**, not a system.
- "La1-xBixCrO3" is a family. Each x is its own system, written as an explicit formula.

### Step 2: text evidence, once per system → [`step2_text_evidence.json`](step2_text_evidence.json)
Question for each system: *"From the text only, what synthesis, structure and per-technique
evidence applies to S_n?"*
- Synthesis (the same for all): solid-state reaction, 99.99% oxides, 5% excess Bi2O3, 550 °C/2 h
  then 1250 °C/5-8 h.
- XRD (all): Cu-Kα, 2θ 10-110°, bulk is free of Bi2O3.
- Rietveld (all): FullProf, Pnma, Bi on the La site, non-monotonic lattice-parameter trends.
- TEM and SAED (**S4 only**): ~1 µm grains; SAED gives a=5.48, b=7.76, c=5.49 Å along the [010]
  and [101] zones.

Things to notice:
- Series-wide statements are repeated for every system. The final step merges items that have
  the same technique and quote (15 items remain, not 3×5 + extras).
- **The text gives no lattice-parameter values per composition, only trends.** Text-only
  extraction would leave S1, S2, S3 and S5 without numbers.

### Step 3: figures, once per figure → [`step3_figures.json`](step3_figures.json)
Question for each image: *"Which technique, which systems, what can you read off it?"*
- Fig. 1a: stacked XRD patterns for all 5 compositions, the same reflections each time, no impurity peaks.
- Fig. 1b: Rietveld fit for x=0.30 with a flat difference curve. No R-factors are reported anywhere.
- **Fig. 2 (a-d): a, b, c and V versus x.** This is where the per-composition numbers are.
  Read by eye, so they are marked `approximate: true`:

  | x | a (Å) | b (Å) | c (Å) | V (Å³) |
  |---|---|---|---|---|
  | 0 | 5.480 | 7.761 | 5.515 | 234.37 |
  | 0.20 | 5.485 | 7.757 | 5.500 | 234.01 |
  | 0.25 | 5.488 | 7.760 | 5.497 | 234.12 |
  | 0.30 | 5.492 | 7.763 | 5.495 | 234.26 |
  | 0.35 | 5.489 | 7.755 | 5.483 | 233.37 |

  Sanity check: a·b·c for x=0.35 = 5.489 × 7.755 × 5.483 ≈ 233.4 Å³, which matches panel (d).
- Fig. 3: TEM micrograph (a) and SAED patterns along <010> (b) and <101> (c) for x=0.30.

**Main lesson: the figure step is not optional.** For structure generation, the most important
numbers in this paper exist only in Fig. 2.

### Step 4: build the graph and save (plain code)
`replay_into_graph.py` feeds the three JSON files into the **real** pydantic graph through a
fake "replay" model. They pass through the same schemas, validators and output code as an
OpenAI run, so [`output/`](output/) has exactly the format the GPT version will produce:

```
output/
  paper_graph.json                 # everything (load with load_graph)
  summary.md                       # readable report
  graph.dot                        # picture of paper → systems → techniques
  figures/*.png
  evidence_by_technique/
    XRD.json  Rietveld_refinement.json  TEM.json  SAED.json
```

`evidence_by_technique/Rietveld_refinement.json` is what a later "diffraction-analysis agent"
would receive: the techniques, the systems they cover (with synthesis and structure claims), and
every evidence item with its numbers, conditions, quote and figure image path.

## Using this as a reference answer

Once the OpenAI version has run on the same PDF, compare it against this folder:

```bash
cd hackathon/paper2structure
.venv/bin/paper-summarizer ../papers/La1-xBixCrO3_arXiv1403.3526.pdf --out outputs/la_bi_cro3_gpt
diff <(jq '.systems[].name' experiments/claude_manual/output/paper_graph.json) \
     <(jq '.systems[].name' outputs/la_bi_cro3_gpt/paper_graph.json)
```

Questions to ask when comparing:
- Did it find exactly 5 systems, without listing precursors?
- Did it put TEM/SAED on S4 only?
- Did it read the Fig. 2 values within about ±0.003 Å?
- Did it invent numbers that are not in the paper, such as R-factors?

## Re-running

```bash
cd experiments/claude_manual
../../.venv/bin/python replay_into_graph.py ../../../papers/La1-xBixCrO3_arXiv1403.3526.pdf
```
