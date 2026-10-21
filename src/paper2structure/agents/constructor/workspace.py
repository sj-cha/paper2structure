"""Construction workspaces. Each construction attempt gets a directory in which the Construction
agent (a pydantic-ai-harness Coder) writes and runs its scripts. Commands run in a bubblewrap
sandbox: the host is read-only, the network is off, only the workspace is writable, the paths in
`construction.hide_paths` are hidden and no API key is in the environment."""
from __future__ import annotations

import glob
import os
import shutil
import subprocess
import sys
from pathlib import Path

from pydantic_ai.workspaces import (CommandResult, LocalWorkspaceBackend, Workspace, WorkspaceBackend,
                                   WorkspaceTimeoutError)
from pydantic_ai_harness import BubblewrapWorkspace

from paper2structure.config import ConstructionConfig
from paper2structure.tools._geometry import ensure_warp_cache

PASSTHROUGH_ENV = ("CUDA_VISIBLE_DEVICES", "WARP_CACHE_PATH", "P2S_NEIGHBOR_BACKEND", "P2S_DEVICE")
PROJECT_ROOT = Path(__file__).resolve().parents[4]


def command_env(workdir: Path, gpu: bool = True) -> dict[str, str]:
    """Environment of every command in a workspace: this interpreter first on PATH, HOME in the
    workspace's .home/ (the sandbox keeps its login files there) and the GPU settings; nothing else
    from this process (no API keys)."""
    ensure_warp_cache()
    home = workdir / ".home"
    home.mkdir(parents=True, exist_ok=True)
    env = {"PATH": f"{Path(sys.executable).parent}:/usr/local/bin:/usr/bin:/bin", "HOME": str(home),
           "LANG": "C.UTF-8", "MPLCONFIGDIR": str(workdir / ".mpl"), "PYTHONDONTWRITEBYTECODE": "1",
           "PYDANTIC_AI_NO_BANNER": "1", "OMP_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "4",
           "MKL_NUM_THREADS": "4"}
    for key in PASSTHROUGH_ENV:
        if key in os.environ:
            env[key] = os.environ[key]
    if not gpu:  # no devices in the sandbox: skip CUDA start-up attempts
        env["P2S_NEIGHBOR_BACKEND"] = "ase"
        env["CUDA_VISIBLE_DEVICES"] = ""
    return env


def _hidden(raw: str) -> list[Path]:
    path = Path(os.path.expanduser(raw))
    if path.is_absolute():
        return [path]
    return list(dict.fromkeys(p.resolve() for p in (Path.cwd() / path, PROJECT_ROOT / path)))


def bwrap_args(cc: ConstructionConfig, workdir: Path) -> list[str]:
    """Extra bubblewrap arguments: hide secrets and, with `gpu`, expose the NVIDIA devices and Warp's
    kernel cache. CUDA fails under the harness's network-off seccomp socket filter, so GPU sandboxes
    keep the network off with their own (empty) network namespace instead."""
    workdir = workdir.resolve()
    args: list[str] = []
    for raw in cc.hide_paths:
        for path in _hidden(raw):
            if workdir == path or path in workdir.parents:  # never hide the workspace itself
                continue
            if path.is_dir():
                args += ["--tmpfs", str(path)]
            elif path.exists():
                args += ["--ro-bind", "/dev/null", str(path)]
    if cc.gpu:
        if not cc.network:
            args.append("--unshare-net")
        for dev in sorted(glob.glob("/dev/nvidia*")):
            args += ["--dev-bind", dev, dev]
        cache = os.environ.get("WARP_CACHE_PATH")
        if cache and os.path.isdir(cache):
            args += ["--bind", cache, cache]
    return args


def sandbox_available() -> tuple[bool, str]:
    """Whether bubblewrap can create a sandbox on this host."""
    exe = shutil.which("bwrap", path=f"{Path(sys.executable).parent}:{os.environ.get('PATH', '')}")
    if exe is None:
        return False, "bwrap not found; install it with `conda install -c conda-forge bubblewrap`"
    probe = subprocess.run([exe, "--ro-bind", "/", "/", "--dev", "/dev", "--unshare-user", "--unshare-net", "true"],
                           capture_output=True, text=True, timeout=30)
    return probe.returncode == 0, probe.stderr.strip()


def make_workspace(workdir: Path, cc: ConstructionConfig) -> WorkspaceBackend:
    """The workspace backend for `workdir`, sandboxed unless `construction.sandbox` is "none"."""
    workdir = Path(workdir).resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    backend = LocalWorkspaceBackend(workdir, env=command_env(workdir, gpu=cc.gpu or cc.sandbox == "none"))
    if cc.sandbox == "none":
        return backend
    return BubblewrapWorkspace(Workspace(backend), network=cc.network or cc.gpu, bwrap_args=bwrap_args(cc, workdir))


async def replay(script: Path, out_dir: Path, cc: ConstructionConfig) -> CommandResult:
    """Run `script` on its own in a fresh, empty directory, as a user would, in the same sandbox."""
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    shutil.copy2(script, out_dir / script.name)
    cmd = ["python", script.name]
    if cc.memory_limit_gb:
        cmd = ["prlimit", f"--as={int(cc.memory_limit_gb * 1024 ** 3)}", *cmd]
    try:
        return await make_workspace(out_dir, cc).run(cmd, timeout=cc.code_timeout)
    except WorkspaceTimeoutError as e:
        return CommandResult(exit_code=124, stdout=str(e.stdout or ""),
                             stderr=f"{e.stderr or ''}\ntimed out after {cc.code_timeout:.0f} s")
