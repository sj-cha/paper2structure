"""Interactive 3D crystal viewer (3Dmol.js) for the Structure tab: rotate, zoom, unit cell, element legend."""

from __future__ import annotations

import io
import json

import streamlit as st
from ase import Atoms
from ase.data import atomic_numbers
from ase.data.colors import jmol_colors
from ase.io import write

_3DMOL = "https://cdn.jsdelivr.net/npm/3dmol@2.4.2/build/3Dmol-min.js"


def structure_viewer(atoms: Atoms, style: str = "Ball and stick", show_cell: bool = True, height: int = 520) -> None:
    periodic = bool(atoms.pbc.all())
    buf = io.BytesIO()
    write(buf, atoms, format="cif" if periodic else "xyz")
    if style == "Space filling":
        look = {"sphere": {"scale": 0.95, "colorscheme": "Jmol"}}
    elif len(atoms) > 3000:  # bonds for this many atoms make the page slow
        look = {"sphere": {"scale": 0.3, "colorscheme": "Jmol"}}
    else:
        look = {"sphere": {"scale": 0.28, "colorscheme": "Jmol"}, "stick": {"radius": 0.12, "colorscheme": "Jmol"}}
    legend = "".join(
        f'<span class="chip"><i style="background:{_hex(e)}"></i>{e}</span>' for e in sorted(set(atoms.get_chemical_symbols()))
    )
    html = f"""
<style>
  body {{ margin: 0; font-family: sans-serif; }}
  #v {{ width: 100%; height: {height - 44}px; position: relative; border: 1px solid #e3e3e8; border-radius: 8px; }}
  .bar {{ display: flex; gap: 8px; align-items: center; flex-wrap: wrap; margin-top: 8px; font-size: 13px; }}
  .chip {{ display: inline-flex; align-items: center; gap: 4px; }}
  .chip i {{ width: 12px; height: 12px; border-radius: 50%; border: 1px solid #0003; display: inline-block; }}
  button {{ font-size: 12px; padding: 2px 8px; border: 1px solid #ccc; border-radius: 6px; background: #fff; cursor: pointer; }}
</style>
<div id="v"></div>
<div class="bar"><button id="spin">Spin</button><button id="reset">Reset view</button>{legend}</div>
<script src="{_3DMOL}"></script>
<script>
  const viewer = $3Dmol.createViewer("v", {{ backgroundColor: "white" }});
  const model = viewer.addModel({json.dumps(buf.getvalue().decode())}, "{'cif' if periodic else 'xyz'}", {{ doAssembly: false, duplicateAssemblyAtoms: false }});
  viewer.setStyle({{}}, {json.dumps(look)});
  if ({json.dumps(show_cell and periodic)}) viewer.addUnitCell(model, {{ box: {{ color: "#555" }} }});
  viewer.zoomTo(); viewer.render();
  let spinning = false;
  document.getElementById("spin").onclick = () => {{ spinning = !spinning; viewer.spin(spinning ? "y" : false); }};
  document.getElementById("reset").onclick = () => {{ viewer.zoomTo(); viewer.render(); }};
</script>
"""
    st.iframe(html, height=height)  # our own HTML: only element symbols and coordinates go into it


def _hex(symbol: str) -> str:
    r, g, b = (round(255 * x) for x in jmol_colors[atomic_numbers[symbol]])
    return f"#{r:02x}{g:02x}{b:02x}"
