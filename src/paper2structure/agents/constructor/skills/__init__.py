"""Agent Skills for the Construction agent (pydantic-ai-harness `Skills`). Each workspace gets a copy
under .agents/skills/; the toolkit reference is generated from the docstrings of paper2structure.tools."""
from __future__ import annotations

import inspect
import textwrap
from pathlib import Path
from typing import Optional

from paper2structure.config import ValidationConfig

SKILLS_DIR = Path(__file__).resolve().parent
WORKSPACE_LIBRARY = ".agents/skills"


def api_reference() -> str:
    """Signatures and docstrings of the public paper2structure.tools API."""
    import paper2structure.tools as tools

    out = []
    for name in tools.__all__:
        obj = getattr(tools, name)
        if name == "Adsorbate":
            header = f"Adsorbate.from_smiles{inspect.signature(obj.from_smiles)} -> Adsorbate"
            doc = (inspect.getdoc(obj) or "") + "\n" + (inspect.getdoc(obj.from_smiles) or "")
        elif inspect.isclass(obj):
            header = f"class {name}{inspect.signature(obj)}"
            doc = inspect.getdoc(obj) or ""
        else:
            header = f"{name}{inspect.signature(obj)}"
            doc = inspect.getdoc(obj) or ""
        out.append("```\n" + header + "\n" + textwrap.indent(doc.strip(), "    ") + "\n```")
    return "\n\n".join(out)


def install(workdir: Path, validation: Optional[ValidationConfig] = None) -> Path:
    """Copy the skills into `workdir`/.agents/skills, filling in the generated toolkit reference and
    the run's validation thresholds (`{{<ValidationConfig field>}}`; default: the defaults)."""
    dest = Path(workdir) / WORKSPACE_LIBRARY
    reference = api_reference()
    values = (validation or ValidationConfig()).model_dump()
    for skill in sorted(p for p in SKILLS_DIR.iterdir() if (p / "SKILL.md").is_file()):
        (dest / skill.name).mkdir(parents=True, exist_ok=True)
        text = (skill / "SKILL.md").read_text().replace("{{API_REFERENCE}}", reference)
        for key, value in values.items():
            text = text.replace("{{" + key + "}}", str(value))
        (dest / skill.name / "SKILL.md").write_text(text)
    return dest
