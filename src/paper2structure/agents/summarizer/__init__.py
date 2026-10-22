"""Agent 1, the paper summarizer: reads the PDF once (text, figures, material systems) and writes a
PaperGraph (paper_graph.json): metadata, material systems and per-technique evidence. graph.py has
its steps; cli.py runs it on its own."""
from paper2structure.agents.summarizer.graph import summarize_paper, summarize_paper_sync
from paper2structure.agents.summarizer.knowledge_graph import load_graph
from paper2structure.agents.summarizer.schemas import Evidence, MaterialSystem, PaperGraph

__all__ = ["summarize_paper", "summarize_paper_sync", "load_graph", "Evidence", "MaterialSystem", "PaperGraph"]
