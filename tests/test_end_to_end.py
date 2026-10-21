"""Full CLI run with the real LLM (opt-in: pytest -m live, needs OPENAI_API_KEY)."""
import json
import os
import subprocess
import sys

import pytest

from helpers import REPO


@pytest.mark.live
def test_cli_pt111_co_end_to_end(tmp_path):
    if not os.environ.get("OPENAI_API_KEY"):
        pytest.skip("needs OPENAI_API_KEY")
    out = tmp_path / "run"
    proc = subprocess.run([sys.executable, "run_construction.py", "--config", "config/pt111_co.yml",
                           "--output_dir", str(out), "--random_seed", "0"],
                          cwd=REPO, capture_output=True, text=True, timeout=3600)
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-3000:]
    result = json.loads((out / "result.json").read_text())
    assert result["status"] in ("accepted", "inconclusive", "revision_budget_exhausted")
    assert result["structure_files"] and (out / "final_report.md").exists()
