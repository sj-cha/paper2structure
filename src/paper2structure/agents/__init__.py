"""LLM agents of the paper-to-structure workflow.

- ``summarizer``: reads the paper once (text, figures, material systems); outside the orchestrator.
- ``characterization``: one agent per technique; reads figures and text, returns structural constraints.
- ``hypothesis``: proposes candidate structures from the constraints (not implemented yet).
- ``constructor``: builds atomistic models from a hypothesis (not implemented yet).
- ``revision``: checks built models against the constraints and decides what to redo (not implemented yet).

The orchestrator that runs them is in ``paper2structure.graph``.
"""
