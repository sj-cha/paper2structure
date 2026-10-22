"""The web tool: a small Starlette app (Starlette and uvicorn are already installed) serving the page
in app/static/ and a JSON API over app/projects.py. Run from the repo root:

    .venv/bin/python app/server.py                     # http://127.0.0.1:8600
    .venv/bin/python app/server.py --port 8700 --host 0.0.0.0
    P2S_WORKERS=5 .venv/bin/python app/server.py       # papers processed at the same time (default 3)

Long steps run as jobs on a worker pool (projects.JobManager); the page polls /api/jobs/<id>.
"""

from __future__ import annotations

import argparse
import inspect
import json
import logging
import os
import sys
from functools import wraps
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))  # app/projects.py and app/structure_info.py

os.environ.setdefault("PYDANTIC_AI_NO_BANNER", "1")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)

import uvicorn  # noqa: E402
from starlette.applications import Starlette  # noqa: E402
from starlette.concurrency import run_in_threadpool  # noqa: E402
from starlette.requests import Request  # noqa: E402
from starlette.responses import FileResponse, JSONResponse, Response  # noqa: E402
from starlette.routing import Mount, Route  # noqa: E402
from starlette.staticfiles import StaticFiles  # noqa: E402

import projects as P  # noqa: E402
from paper2structure.agents.constructor.workspace import sandbox_available  # noqa: E402

os.environ.setdefault("MPLCONFIGDIR", str(P.ROOT / ".mpl"))
P.ROOT.mkdir(parents=True, exist_ok=True)
JOBS = P.JobManager()


def api(fn):
    """JSON endpoint: KeyError -> 404, ValueError/RuntimeError -> 400 with the message; sync handlers
    (file reads, ASE) run in the threadpool so they never stall the job polling."""

    async def call(request: Request):
        try:
            result = await fn(request) if inspect.iscoroutinefunction(fn) else await run_in_threadpool(fn, request)
            return result if isinstance(result, Response) else JSONResponse(result)
        except KeyError as e:
            return JSONResponse({"error": e.args[0] if e.args else str(e)}, status_code=404)  # str(KeyError) adds quotes
        except json.JSONDecodeError as e:  # a file a worker is still writing: the page retries
            return JSONResponse({"error": f"a file is being written, try again: {e}"}, status_code=503)
        except (ValueError, RuntimeError) as e:
            return JSONResponse({"error": str(e)}, status_code=400)
        except Exception as e:  # noqa: BLE001 - the page shows it
            logging.getLogger("webtool").exception("%s failed", fn.__name__)
            return JSONResponse({"error": f"{type(e).__name__}: {e}"}, status_code=500)

    return wraps(fn)(call)


def _project(request: Request) -> P.Project:
    return P.Project.load(request.path_params["pid"])


async def index(request: Request):
    return FileResponse(HERE / "static" / "index.html")


@api
def info(request: Request):
    ok, why = sandbox_available()
    return {"models": P.model_choices(), "efforts": list(P.EFFORTS), "model_types": P.MODEL_TYPES,
            "keys": P.key_status(), "sandbox": {"available": ok, "reason": why}, "workers": P.WORKERS,
            "root": str(P.ROOT)}


@api
async def set_key(request: Request):
    body = await request.json()
    provider, key = body.get("provider", "openai"), body.get("key", "").strip()
    if provider not in P.PROVIDER_KEYS:
        raise ValueError(f"unknown provider {provider!r}")
    if not key:
        raise ValueError("empty key")
    os.environ[P.PROVIDER_KEYS[provider][0]] = key  # this server process only; .env is the durable place
    return {"keys": P.key_status()}


@api
def list_projects(request: Request):
    out = []
    for p in P.Project.list_all():
        m = p.meta
        runs = list((p.dir / "runs").glob("*/settings.json")) if (p.dir / "runs").exists() else []
        job = JOBS.active(p.id)
        out.append({"id": p.id, "title": m["title"], "pdf_name": m["pdf_name"], "created": m["created"],
                    "model": m["model"], "effort": m["effort"], "summary": bool(m.get("summary")),
                    "evidence": (p.dir / "characterization").exists() and any((p.dir / "characterization").glob("*.json")),
                    "runs": len(runs), "active_job": job.id if job else None})
    return out


@api
async def create_project(request: Request):
    """One project and one summarize job per uploaded PDF (the field may repeat)."""
    form = await request.form()
    files = [f for f in form.getlist("pdf") if getattr(f, "filename", "")]
    if not files:
        raise ValueError("choose one or more PDF files")
    model = form.get("model") or P.default_model()
    effort = form.get("effort") or "low"
    if effort not in P.EFFORTS:
        raise ValueError(f"reasoning effort must be one of {P.EFFORTS}")
    P.check_model(model)  # fail now, with the provider's key variable named, rather than inside every job
    options = {"analyze_figures": form.get("analyze_figures", "true") == "true",
               "max_figures": max(1, min(60, int(form.get("max_figures") or 30))),
               "then_characterize": form.get("then_characterize", "true") == "true"}
    blobs = [(f.filename, await f.read()) for f in files]
    bad = [name for name, data in blobs if not data.startswith(b"%PDF")]
    if bad:  # validate everything before creating anything
        raise ValueError(f"not a PDF: {', '.join(bad)}")
    created = []
    for filename, data in blobs:
        project = P.Project.create(data, filename, model, effort, **options)
        job = JOBS.submit(project, "summarize", lambda say, p=project: P.run_summarize(p, say))
        created.append({"project_id": project.id, "job_id": job.id, "title": project.meta["title"]})
    return {"projects": created}


@api
def get_project(request: Request):
    project = _project(request)
    job = JOBS.active(project.id)
    return P.project_state(project, job.id if job else None)


@api
def delete_project(request: Request):
    project = _project(request)
    if JOBS.active(project.id):
        raise RuntimeError("a job is running on this paper; wait for it to finish")
    project.delete()
    return {"deleted": project.id}


@api
async def update_settings(request: Request):
    project = _project(request)
    if JOBS.active(project.id):
        raise RuntimeError("a job is running on this paper; change the model after it finishes")
    body = await request.json()
    if "model" in body:
        P.check_model(str(body["model"]))  # the provider's key must exist
        project.meta["model"] = str(body["model"])
    if "effort" in body:
        if body["effort"] not in P.EFFORTS:
            raise ValueError(f"reasoning effort must be one of {P.EFFORTS}")
        project.meta["effort"] = body["effort"]
    project.save()
    return {"model": project.meta["model"], "effort": project.meta["effort"]}


@api
async def summarize(request: Request):
    project = _project(request)
    body = await request.json() if await request.body() else {}
    if "analyze_figures" in body:
        project.meta["analyze_figures"] = bool(body["analyze_figures"])
    if "then_characterize" in body:
        project.meta["then_characterize"] = bool(body["then_characterize"])
    project.save()
    job = JOBS.submit(project, "summarize", lambda say: P.run_summarize(project, say))
    return {"job_id": job.id}


@api
async def characterize(request: Request):
    project = _project(request)
    settings = P.Settings.parse(await request.json() if await request.body() else {})
    P._check_summary(project)
    job = JOBS.submit(project, "characterize", lambda say: P.run_characterize(project, settings, say))
    return {"job_id": job.id}


@api
async def generate(request: Request):
    project = _project(request)
    settings = P.Settings.parse(await request.json())
    P._check_systems(project, settings)  # fail now, not in the job, so the page can show it at once
    job = JOBS.submit(project, "generate", lambda say: P.run_generate(project, settings, say))
    return {"job_id": job.id}


@api
async def make_variant(request: Request):
    """Re-run a structure's saved build script with new parameter values (a fast job, no agent)."""
    project, run, sid = _project(request), request.path_params["run"], request.path_params["sid"]
    body = await request.json() if await request.body() else {}
    values = body.get("params") or {}
    if not isinstance(values, dict):
        raise ValueError("params must be an object of parameter name -> value")
    P.check_variant(project, run, sid, values)  # a bad value fails now, with the parameter named
    job = JOBS.submit(project, "variant", lambda say: P.run_variant(project, run, sid, values, say))
    return {"job_id": job.id}


@api
def get_variant_model(request: Request):
    return P.structure_model(_project(request), request.path_params["run"], request.path_params["name"],
                             variant=request.path_params["vid"])


@api
def list_jobs(request: Request):
    return [j.to_dict(full=False) for j in JOBS.recent()]


@api
def get_job(request: Request):
    job = JOBS.jobs.get(request.path_params["jid"])
    if job is None:
        raise KeyError("no such job")
    return job.to_dict()


@api
def cancel_job(request: Request):
    return JOBS.cancel(request.path_params["jid"]).to_dict(full=False)


@api
def get_run(request: Request):
    return P.run_detail(_project(request), request.path_params["run"])


@api
def get_structure_model(request: Request):
    return P.structure_model(_project(request), request.path_params["run"], request.path_params["sid"])


@api
def get_run_zip(request: Request):
    project, name = _project(request), request.path_params["run"]
    data = P.run_zip(project, name)
    return Response(data, media_type="application/zip",
                    headers={"Content-Disposition": f'attachment; filename="{project.id}_{name}.zip"'})


@api
def get_file(request: Request):
    return FileResponse(_project(request).file(request.path_params["path"]))


app = Starlette(routes=[
    Route("/", index),
    Route("/api/info", info),
    Route("/api/key", set_key, methods=["POST"]),
    Route("/api/projects", list_projects),
    Route("/api/projects", create_project, methods=["POST"]),
    Route("/api/projects/{pid}", get_project),
    Route("/api/projects/{pid}", delete_project, methods=["DELETE"]),
    Route("/api/projects/{pid}/settings", update_settings, methods=["POST"]),
    Route("/api/projects/{pid}/summarize", summarize, methods=["POST"]),
    Route("/api/projects/{pid}/characterize", characterize, methods=["POST"]),
    Route("/api/projects/{pid}/generate", generate, methods=["POST"]),
    Route("/api/projects/{pid}/runs/{run}", get_run),
    Route("/api/projects/{pid}/runs/{run}/zip", get_run_zip),
    Route("/api/projects/{pid}/runs/{run}/structures/{sid}/model", get_structure_model),
    Route("/api/projects/{pid}/runs/{run}/structures/{sid}/variants", make_variant, methods=["POST"]),
    Route("/api/projects/{pid}/runs/{run}/variants/{vid}/structures/{name}/model", get_variant_model),
    Route("/api/projects/{pid}/files/{path:path}", get_file),
    Route("/api/jobs", list_jobs),
    Route("/api/jobs/{jid}", get_job),
    Route("/api/jobs/{jid}/cancel", cancel_job, methods=["POST"]),
    Mount("/static", StaticFiles(directory=HERE / "static"), name="static"),
])


def main() -> None:
    parser = argparse.ArgumentParser(description="Paper -> Structure web tool")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8600)
    args = parser.parse_args()
    ok, why = sandbox_available()
    print(f"[paper2structure] projects in {P.ROOT}; {P.WORKERS} workers; sandbox: {'bubblewrap' if ok else f'none ({why})'}",
          flush=True)
    print(f"[paper2structure] open http://{args.host}:{args.port}", flush=True)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
