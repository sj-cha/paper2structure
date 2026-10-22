"""Full CLI runs (main.py) with the real LLM (opt-in: pytest -m live, needs OPENAI_API_KEY)."""
import json
import os
import subprocess
import sys

import pytest

from helpers import REPO
from paper2structure.agents.characterization.schemas import ExperimentalEvidence
from paper2structure.agents.summarizer import load_graph

PDF = REPO / "pdf" / "1403.3526v1.pdf"  # a real paper (XRD and TEM); pdf/ is git-ignored


def _main(out, *args, timeout: float = 3600) -> dict:
    """Run main.py with `args` into `out`; returns result.json after checking the run finished."""
    if not os.environ.get("OPENAI_API_KEY"):
        pytest.skip("needs OPENAI_API_KEY")
    proc = subprocess.run([sys.executable, "main.py", *args, "--output_dir", str(out), "--random_seed", "0"],
                          cwd=REPO, capture_output=True, text=True, timeout=timeout)
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-3000:]
    result = json.loads((out / "result.json").read_text())
    assert result["status"] in ("accepted", "inconclusive", "revision_budget_exhausted"), result["errors"]
    assert result["structure_files"] and (out / "final_report.md").exists()
    return result


@pytest.mark.live
def test_cli_pt111_co_end_to_end(tmp_path):
    _main(tmp_path / "run", "--config", "config/pt111_co.yml")


@pytest.mark.live
@pytest.mark.skipif(not PDF.exists(), reason=f"{PDF} not found (pdf/ is git-ignored)")
def test_cli_pdf_end_to_end(tmp_path):
    """PDF -> summary -> characterization -> planning -> construction -> review in one graph run."""
    out = tmp_path / "run"
    _main(out, "--pdf", str(PDF), timeout=7200)
    paper = load_graph(out / "paper" / "paper_graph.json")
    assert paper.systems and paper.evidence
    assert ExperimentalEvidence.load(out / "characterization").reports  # at least one technique analyzed
    state = json.loads(sorted((out / "state").glob("*_FinalizeResults.json"))[-1].read_text())
    assert state["node_history"][:4] == ["SummarizePaper", "AnalyzeEvidence", "PlanStructures", "ConstructStructures"]
