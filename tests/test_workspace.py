"""Construction workspaces: the bubblewrap sandbox, the from-scratch replay and the skills library."""
import asyncio

import pytest

from helpers import GOOD_SCRIPT, REPO, make_config
from paper2structure.agents.constructor import skills
from paper2structure.agents.constructor.workspace import bwrap_args, make_workspace, replay, sandbox_available

SANDBOX, WHY = sandbox_available()
needs_sandbox = pytest.mark.skipif(not SANDBOX, reason=f"bubblewrap unavailable: {WHY}")

PROBE = '''\
import os, socket
import paper2structure.tools
try:
    socket.create_connection(("1.1.1.1", 53), timeout=2)
    print("network: open")
except OSError:
    print("network: blocked")
try:
    print("secret:", open({secret!r}).read() or "<empty>")
except OSError:
    print("secret: unreadable")
try:
    open({outside!r}, "w").write("x")
    print("outside: writable")
except OSError:
    print("outside: read-only")
print("api key:", "OPENAI_API_KEY" in os.environ)
'''


@needs_sandbox
def test_sandbox_blocks_network_secrets_and_host_writes(tmp_path, monkeypatch):
    secret = tmp_path / "host" / ".env"
    secret.parent.mkdir()
    secret.write_text("OPENAI_API_KEY=sk-do-not-leak")
    outside = REPO / ".sandbox_probe"  # on the host filesystem (the sandbox's /tmp is private)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-do-not-leak")
    cc = make_config(construction={"hide_paths": [str(secret)]}).construction
    ws_dir = tmp_path / "ws"
    (ws_dir).mkdir()
    (ws_dir / "probe.py").write_text(PROBE.format(secret=str(secret), outside=str(outside)))
    try:
        res = asyncio.run(make_workspace(ws_dir, cc).run(["python", "probe.py"], timeout=120))
    finally:
        written = outside.exists()
        outside.unlink(missing_ok=True)
    assert res.exit_code == 0, res.stderr
    assert not written
    assert "network: blocked" in res.stdout and "outside: read-only" in res.stdout
    assert "sk-do-not-leak" not in res.stdout and "api key: False" in res.stdout


def test_mounts_use_real_paths_behind_symlinks(tmp_path, monkeypatch):
    # a home directory linked to another file system: bwrap cannot create a mount point below the link
    real = tmp_path / "real"
    (real / ".ssh").mkdir(parents=True)
    (real / ".netrc").write_text("machine example.org password secret")
    (real / "warp").mkdir()
    home = tmp_path / "home"
    home.symlink_to(real)
    monkeypatch.setenv("WARP_CACHE_PATH", str(home / "warp"))
    cc = make_config(construction={"hide_paths": [str(home / ".ssh"), str(home / ".netrc")], "gpu": True}).construction
    args = bwrap_args(cc, tmp_path / "ws")
    real = real.resolve()
    assert {("--tmpfs", str(real / ".ssh")), ("/dev/null", str(real / ".netrc")),
            ("--bind", str(real / "warp"))} <= set(zip(args, args[1:]))
    assert not any(a.startswith(str(home)) for a in args)


@needs_sandbox
def test_replay_runs_a_script_alone_from_scratch(tmp_path):
    cc = make_config().construction
    script = tmp_path / "ws" / "build_P1.py"
    script.parent.mkdir()
    script.write_text(GOOD_SCRIPT)
    (tmp_path / "ws" / "leftover.txt").write_text("not copied")
    res = asyncio.run(replay(script, tmp_path / "replay" / "build_P1", cc))
    assert res.exit_code == 0, res.stderr
    out = tmp_path / "replay" / "build_P1"
    assert (out / "structures" / "pt111_co_top.extxyz").exists() and (out / "structures" / "pt111_co_top.vasp").exists()
    assert not (out / "leftover.txt").exists()
    broken = tmp_path / "ws" / "build_broken.py"
    broken.write_text("open('leftover.txt').read()\n")  # depends on a file that is not part of the script
    assert asyncio.run(replay(broken, tmp_path / "replay" / "build_broken", cc)).exit_code != 0


def test_skills_install_with_generated_reference(tmp_path):
    lib = skills.install(tmp_path)
    names = sorted(p.name for p in lib.iterdir())
    assert names == ["custom-structures", "nanocrystals", "solvation", "structure-toolkit", "surfaces-and-adsorbates"]
    toolkit = (lib / "structure-toolkit" / "SKILL.md").read_text()
    assert toolkit.startswith("---\nname: structure-toolkit\ndescription: ")
    assert "{{API_REFERENCE}}" not in toolkit
    for fn in ("build_bulk(", "place_adsorbates(", "add_molecule(", "carve_nanocrystal(", "emit("):
        assert fn in toolkit
