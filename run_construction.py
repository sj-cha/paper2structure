from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s %(message)s")
os.environ.setdefault("PYDANTIC_AI_NO_BANNER", "1")

from paper2structure.config import load_config  # noqa: E402
from paper2structure.graph.workflow import build_deps, run_workflow  # noqa: E402
from paper2structure.agents.constructor.workspace import sandbox_available  # noqa: E402


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config/config.yml")
    p.add_argument("--output_dir", default="runs/run_001")
    p.add_argument("--random_seed", type=int, required=True,
                   help="seeds conformer embedding, adsorbate placement and solvent packing")
    p.add_argument("--max_revisions", type=int, default=1,
                   help="loops the reviewer may take back through evidence -> planning -> construction")
    p.add_argument("--max_repairs", type=int, default=2,
                   help="construction retries for structures failing deterministic validation")
    p.add_argument("--gpu", type=int, default=None,
                   help="GPU index for CUDA-accelerated neighbor lists, in the workflow and in the agent's "
                        "sandboxed scripts (sets CUDA_VISIBLE_DEVICES and construction.gpu)")
    p.add_argument("--skip_preflight", action="store_true", help="skip the per-agent connectivity checks")
    return p


def _resolve(path: str, config_path: str) -> str:
    """Input paths are relative to the working directory, falling back to the config's folder."""
    p = Path(path)
    if p.is_absolute() or p.exists():
        return str(p.resolve())
    alt = Path(config_path).resolve().parent / p
    return str(alt) if alt.exists() else str(p)


def main() -> int:
    args = _build_parser().parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: needs OPENAI_API_KEY in env (source .env).", file=sys.stderr)
        return 2
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    cfg = load_config(args.config)
    if args.gpu is not None:
        cfg.construction.gpu = True  # the agent's sandboxed scripts may use it too
    cfg.inputs.paper_summary = _resolve(cfg.inputs.paper_summary, args.config)
    cfg.inputs.experimental_evidence = _resolve(cfg.inputs.experimental_evidence, args.config)

    os.makedirs(args.output_dir, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(Path(args.output_dir).resolve() / ".mpl"))

    # Snapshot the config + resolved CLI run-level values
    snapshot = {"config": cfg.model_dump(mode="json"), "cli": vars(args)}
    with open(os.path.join(args.output_dir, "config.json"), "w") as fh:
        json.dump(snapshot, fh, indent=2)

    deps = build_deps(cfg, args.output_dir, seed=args.random_seed, max_revisions=args.max_revisions,
                      max_repairs=args.max_repairs)
    cc = cfg.construction
    print(f"[paper2structure] out={args.output_dir} model={cfg.campaign.model} seed={args.random_seed} "
          f"budget(revisions={args.max_revisions}, repairs={args.max_repairs}, "
          f"construction_requests={cc.max_requests}) sandbox={cc.sandbox}", flush=True)

    if cc.sandbox == "bubblewrap":
        ok, err = sandbox_available()
        if not ok:
            print(f"ERROR: bubblewrap sandbox unavailable: {err} (or set construction.sandbox: none)", file=sys.stderr)
            return 4
    else:
        print("[paper2structure] WARNING: construction.sandbox is 'none': the agent's commands run unsandboxed",
              flush=True)

    if not args.skip_preflight:
        for name, agent in (("planner", deps.planner), ("constructor", deps.constructor),
                            ("reviser", deps.reviser)):
            ok, err = agent.preflight()
            if not ok:
                print(f"ERROR: {name} preflight failed: {err}", file=sys.stderr)
                return 3
            print(f"[paper2structure] {name} preflight OK.", flush=True)

    result = asyncio.run(run_workflow(deps))

    print("\n================ PAPER2STRUCTURE SUMMARY ================", flush=True)
    print(f"status: {result.status}   revisions: {result.revisions}", flush=True)
    for sid, files in result.structure_files.items():
        mark = "*" if sid in result.accepted_structure_ids else " "
        print(f" {mark} {sid}: {', '.join(files.values())}", flush=True)
    for e in result.errors:
        print(f"error: {e}", flush=True)
    print(f"\nreport: {os.path.join(result.run_dir, result.report)} (* = accepted)", flush=True)
    return 0 if result.status != "failed" else 1


if __name__ == "__main__":
    sys.exit(main())
