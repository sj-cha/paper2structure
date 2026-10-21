"""Web page for uploading a paper and browsing what was extracted.

Run from the repo root:
    .venv/bin/streamlit run app/streamlit_app.py
then open the URL it prints (VS Code forwards the port automatically on a remote machine).
"""

from __future__ import annotations

import io
import os
import time
import zipfile
from pathlib import Path

import streamlit as st

from paper2structure.agents.summarizer.graph import summarize_paper_sync
from paper2structure.agents.summarizer.knowledge_graph import load_graph, to_dot, technique_packets
from paper2structure.structure import export, generate_structure
from paper2structure.config import DEFAULT_MODEL, MODEL_CHOICES, get_model
from structure_viewer import structure_viewer

OUTPUTS = Path("outputs")

st.set_page_config(page_title="Paper → Structure summarizer", page_icon="🧪", layout="wide")
st.title("Paper → Structure summarizer")
st.write(
    "Upload a materials-science paper (PDF). The assistant finds every material system studied, "
    "collects what the text and the figures say about each one, and files the evidence by "
    "experimental technique (XRD, TEM, XPS, ...)."
)

with st.sidebar:
    st.header("Settings")
    default_model = os.getenv("P2S_MODEL", DEFAULT_MODEL)
    choices = list(MODEL_CHOICES) + ["other"]
    choice = st.selectbox(
        "Model",
        choices,
        index=choices.index(default_model) if default_model in choices else len(choices) - 1,
        format_func=lambda m: MODEL_CHOICES.get(m, "Other (type a model name)"),
        help="USD per 1M tokens (input / output). Luna price varies between sources. GPT-5 nano is cheap per token "
        "but slow and thinks a lot, so it costs more per paper than GPT-5.4 nano. 'Dry run' tests the page without an API key.",
    )
    model_name = st.text_input("Model name", default_model, help="Any pydantic-ai model name, e.g. openai:gpt-5.4-mini") if choice == "other" else choice
    if not os.getenv("OPENAI_API_KEY") and model_name.startswith("openai:"):
        key = st.text_input("OpenAI API key", type="password", help="Only kept for this browser session. Better: put it in the .env file.")
        if key:
            os.environ["OPENAI_API_KEY"] = key
    analyze_figures = st.checkbox("Analyse figures (slower, costs more)", value=True)
    st.divider()
    previous = sorted([p.parent for p in OUTPUTS.glob("*/paper_graph.json")], key=lambda p: p.stat().st_mtime, reverse=True)
    chosen = st.selectbox("…or open a previous result", ["—"] + [p.name for p in previous])

uploaded = st.file_uploader("Paper PDF", type="pdf")
if uploaded and st.button("Summarize this paper", type="primary"):
    run_dir = OUTPUTS / f"{Path(uploaded.name).stem}_{time.strftime('%Y%m%d-%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = run_dir / "paper.pdf"
    pdf_path.write_bytes(uploaded.getvalue())
    try:
        model = get_model(model_name)
    except RuntimeError as e:
        st.error(str(e))
        st.stop()
    with st.status("Working on it…", expanded=True) as status:
        summarize_paper_sync(pdf_path, run_dir, model, on_progress=status.write, analyze_figures=analyze_figures)
        status.update(label="Finished", state="complete")
    st.session_state["result_dir"] = str(run_dir)
elif chosen != "—":
    st.session_state["result_dir"] = str(OUTPUTS / chosen)

if "result_dir" not in st.session_state:
    st.stop()

result_dir = Path(st.session_state["result_dir"])
graph = load_graph(result_dir / "paper_graph.json")
m = graph.metadata

st.header(m.title)
st.caption(f"{', '.join(m.authors)} · DOI: {m.doi or '–'} · model: {graph.model}")
st.write(m.summary)
if (result_dir / "errors.txt").exists():
    st.warning("Some steps had problems:\n\n" + (result_dir / "errors.txt").read_text())

tab_sys, tab_ev, tab_graph, tab_struct, tab_dl = st.tabs(
    [f"Material systems ({len(graph.systems)})", f"Evidence ({len(graph.evidence)})", "Knowledge graph", "Structure (mock-up)", "Download"]
)

with tab_sys:
    st.dataframe(
        [
            {
                "ID": s.system_id,
                "Name": s.name,
                "Variables": ", ".join(f"{k}={v}" for k, v in s.composition_variables.items()),
                "Form": s.form,
                "Crystal system": s.structure.crystal_system,
                "Space group": s.structure.space_group,
                "Lattice parameters": ", ".join(f"{x.quantity}={x.value}{x.unit or ''}" for x in s.structure.lattice_parameters),
                "Synthesis": s.synthesis.method,
            }
            for s in graph.systems
        ],
        hide_index=True,
        width="stretch",
    )
    for s in graph.systems:
        with st.expander(f"{s.system_id} · {s.name} – full details"):
            st.json(s.model_dump())

with tab_ev:
    names = {s.system_id: s.name for s in graph.systems}
    for tech, packet in technique_packets(graph).items():
        st.subheader(f"{tech}  ·  {len(packet['evidence'])} items")
        for ev in graph.evidence_for(technique=tech):
            if ev.technique != tech:
                continue
            who = ", ".join(names.get(i, i) for i in ev.system_ids)
            with st.expander(f"[{ev.evidence_id}] {ev.source_type} · {ev.source_ref} · {who}"):
                cols = st.columns([2, 1]) if ev.image_path else [st.container()]
                with cols[0]:
                    st.write(ev.summary)
                    if ev.measurements:
                        st.table([{"quantity": x.quantity, "value": x.value, "unit": x.unit, "read from plot": x.approximate} for x in ev.measurements])
                    for o in ev.observations:
                        st.write(f"- {o}")
                    if ev.conditions:
                        st.caption("Conditions: " + "; ".join(ev.conditions))
                    if ev.quote:
                        st.caption(f"“{ev.quote}”")
                if ev.image_path:
                    cols[1].image(str(result_dir / ev.image_path))

with tab_graph:
    st.caption("Paper → material systems → techniques. The number on each arrow is how many evidence items link them.")
    st.graphviz_chart(to_dot(graph), width="stretch")

with tab_struct:
    st.info(
        "**Mock-up.** The real structure generator is being built separately. For now this tab always shows "
        "the same demo crystal, sized by the settings, so the workflow can be tried end to end."
    )
    c1, c2, c3 = st.columns([2, 1, 2])
    system = c1.selectbox("Material system", graph.systems, format_func=lambda s: f"{s.system_id} · {s.name}")
    method = c2.radio("Simulation", ["DFT", "MD"], horizontal=True, help="DFT: small periodic cell (≲200 atoms). MD: large cell (thousands of atoms).")
    realism = c3.slider(
        "Computationally tractable  ←→  Closer to experiment",
        0.0, 1.0, 0.3, 0.05,
        help="Left: smallest, cheapest cell. Right: bigger, more realistic cell (in the mock-up: larger and with more thermal noise).",
    )
    if st.button("Generate structure", type="primary"):
        st.session_state["structure"] = (str(result_dir), generate_structure(system, method, realism))

    owner, gen = st.session_state.get("structure", (None, None))
    if gen and owner == str(result_dir):
        st.subheader(f"{gen.system_name} · {gen.method} · realism {gen.realism:.2f}")
        info, estimate = gen.info(), gen.compute_estimate()
        view, facts = st.columns([3, 2])
        with view:
            v1, v2 = st.columns(2)
            style = v1.radio("Style", ["Ball and stick", "Space filling"], horizontal=True, index=0 if len(gen.atoms) < 3000 else 1)
            show_cell = v2.checkbox("Show cell", value=True)
            structure_viewer(gen.atoms, style=style, show_cell=show_cell)
            st.caption("Drag to rotate · scroll to zoom · right-drag to move")
        with facts:
            m1, m2, m3 = st.columns(3)
            m1.metric("Atoms", f"{info['atoms']:,}")
            m2.metric("Valence e⁻", f"{info['valence_electrons']:,}")
            m3.metric("Supercell", info["supercell"])
            a, b, c = info["cell_lengths_A"]
            rows = {
                "Formula": info["formula"],
                "Cell (Å)": f"{a} × {b} × {c}",
                "Volume (Å³)": f"{info['volume_A3']:,}",
                "Density (g/cm³)": info["density_g_cm3"],
            }
            st.table({"": list(rows), "value": [str(v) for v in rows.values()]})
            st.markdown("**Compute estimate** (order of magnitude)")
            st.table({"": list(estimate), "value": list(estimate.values())})
            st.caption("Species: " + ", ".join(f"{e} {n}" for e, n in info["species"].items()))
            for note in gen.notes:
                st.caption(f"• {note}")
        if st.button("Use this structure"):
            files = export(gen, result_dir / "structures")
            st.success(f"Saved for simulation: {', '.join(f.name for f in files)}")

    saved = sorted((result_dir / "structures").glob("*.json")) if (result_dir / "structures").exists() else []
    if saved:
        st.markdown("**Structures chosen for simulation**")
        for meta in saved:
            with st.expander(meta.stem):
                cols = st.columns(4)
                for col, f in zip(cols, sorted(meta.parent.glob(f"{meta.stem}.*"))):
                    col.download_button(f.suffix.lstrip("."), f.read_bytes(), file_name=f.name, key=f"dl-{f.name}")

with tab_dl:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for f in result_dir.rglob("*"):
            if f.is_file():
                z.write(f, f.relative_to(result_dir))
    st.download_button("Download everything (.zip)", buf.getvalue(), file_name=f"{result_dir.name}.zip")
    st.download_button("Download paper_graph.json", (result_dir / "paper_graph.json").read_bytes(), file_name="paper_graph.json")
    st.code(f"from paper2structure.agents.summarizer.knowledge_graph import load_graph\ng = load_graph('{result_dir / 'paper_graph.json'}')\nxrd = g.evidence_for(technique='XRD')", language="python")
