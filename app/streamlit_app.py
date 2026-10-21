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

from paper2structure.summarizer.graph import summarize_paper_sync
from paper2structure.summarizer.knowledge_graph import load_graph, to_dot, technique_packets
from paper2structure.summarizer.models import DEFAULT_MODEL, get_model

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
    model_name = st.text_input("Model", os.getenv("P2S_MODEL", DEFAULT_MODEL), help="Use 'dry-run' to test without an API key.")
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

tab_sys, tab_ev, tab_graph, tab_dl = st.tabs([f"Material systems ({len(graph.systems)})", f"Evidence ({len(graph.evidence)})", "Knowledge graph", "Download"])

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

with tab_dl:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for f in result_dir.rglob("*"):
            if f.is_file():
                z.write(f, f.relative_to(result_dir))
    st.download_button("Download everything (.zip)", buf.getvalue(), file_name=f"{result_dir.name}.zip")
    st.download_button("Download paper_graph.json", (result_dir / "paper_graph.json").read_bytes(), file_name="paper_graph.json")
    st.code(f"from paper2structure.summarizer.knowledge_graph import load_graph\ng = load_graph('{result_dir / 'paper_graph.json'}')\nxrd = g.evidence_for(technique='XRD')", language="python")
