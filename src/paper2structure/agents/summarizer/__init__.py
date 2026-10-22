"""The Summarizer: reads the PDF once (text, figures, material systems) and writes a
PaperGraph (paper_graph.json): metadata, material systems and per-technique evidence. graph.py has
its steps; cli.py runs it on its own. summarize_paper is imported on first use: the schemas and
load_graph do not need the PDF reader or the LLM agents."""
from paper2structure.agents.summarizer.knowledge_graph import load_graph
from paper2structure.agents.summarizer.schemas import Evidence, MaterialSystem, PaperGraph

__all__ = ["summarize_paper", "load_graph", "Evidence", "MaterialSystem", "PaperGraph"]


def __getattr__(name: str):
    if name == "summarize_paper":
        from paper2structure.agents.summarizer.graph import summarize_paper

        return summarize_paper
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
