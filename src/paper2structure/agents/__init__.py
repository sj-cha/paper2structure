"""The five agents of the paper-to-structure workflow, one package each:

- ``summarizer`` (Agent 1): reads the paper once (text, figures, material systems) into a PaperGraph.
- ``characterization`` (Agent 2): one agent per technique; reads the figures and text tagged with it
  and returns structural constraints (a CharacterizationReport).
- ``planner`` (Agent 3): structural hypotheses as construction plans.
- ``constructor`` (Agent 4): a coding agent that builds each plan in a sandboxed workspace.
- ``reviser`` (Agent 5): judges the structures and writes the final report.

Agents 3-5 are built on ``structured_agent.StructuredAgent``. The workflow that runs them all is in
``paper2structure.graph``.
"""
