from __future__ import annotations

import io
import os
from pathlib import Path
from typing import Optional

from ase import Atoms


def render_png(atoms: Atoms, path: Optional[str | os.PathLike] = None, radii: float = 0.85) -> bytes:
    """Top and side views of a structure (ase.visualize.plot.plot_atoms), as PNG bytes; also
    written to `path` when given."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from ase.visualize.plot import plot_atoms

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    for ax, rotation, title in zip(axes, ("0x,0y,0z", "-90x,0y,0z"), ("top view (along -z)", "side view")):
        plot_atoms(atoms, ax, rotation=rotation, radii=radii)
        ax.set_title(title, fontsize=10)
        ax.set_axis_off()
    fig.suptitle(f"{atoms.get_chemical_formula()} ({len(atoms)} atoms)", fontsize=11)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    data = buf.getvalue()
    if path is not None:
        Path(path).write_bytes(data)
    return data
