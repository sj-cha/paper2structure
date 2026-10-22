"""The five agents of the paper-to-structure workflow, one package each:

- ``summarizer`` (the Summarizer): reads the paper once (text, figures, material systems) into a PaperGraph.
- ``characterization`` (the characterization agents): one agent per technique; reads the figures and text tagged with it
  and returns structural constraints (a CharacterizationReport).
- ``planner`` (the Planner): structural hypotheses as construction plans.
- ``constructor`` (the Constructor): a coding agent that builds each plan in a sandboxed workspace.
- ``reviser`` (the Reviewer): judges the structures and writes the final report.

The characterization agents, the Planner, the Constructor and the Reviewer are built on
``structured_agent.StructuredAgent``. The workflow that runs them all is in
``paper2structure.graph``.
"""
