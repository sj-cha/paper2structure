"""The web tool: a small Starlette app (Starlette and uvicorn come with Streamlit) serving the page
in app/static/ and a JSON API over app/projects.py. Run from the repo root:

    .venv/bin/python app/server.py                     # http://127.0.0.1:8600
    .venv/bin/python app/server.py --port 8700 --host 0.0.0.0

Long steps run as jobs on a worker thread (projects.JobManager); the page polls /api/jobs/<id>.
"""

from __future__ import annotations

import argparse
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
    """JSON endpoint: KeyError -> 404, ValueError/RuntimeError -> 400 with the message, both sync and async handlers."""
    import inspect

    async def call(request: Request):
        try:
            result = fn(request)
            if inspect.isawaitable(result):
                result = await result
            return result if isinstance(result, Response) else JSONResponse(result)
        except KeyError as e:
            return JSONResponse({"error": e.args[0] if e.args else str(e)}, status_code=404)  # str(KeyError) adds quotes
        except (ValueError, RuntimeError) as e:
            return JSONResponse({"error": str(e)}, status_code=400)
        except Exception as e:  # noqa: BLE001 - e.g. a file the worker is still writing; the page shows it and retries
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
            "has_key": bool(os.environ.get("OPENAI_API_KEY")), "sandbox": {"available": ok, "reason": why},
            "root": str(P.ROOT)}


@api
async def set_key(request: Request):
    key = (await request.json()).get("key", "").strip()
    if not key:
        raise ValueError("empty key")
    os.environ["OPENAI_API_KEY"] = key  # this server process only; .env is the durable place
    return {"has_key": True}


@api
def list_projects(request: Request):
    out = []
    for p in P.Project.list_all():
        m = p.meta
        runs = list((p.dir / "runs").glob("*/settings.json")) if (p.dir / "runs").exists() else []
        job = JOBS.active(p.id)
        out.append({"id": p.id, "title": m["title"], "pdf_name": m["pdf_name"], "created": m["created"],
                    "model": m["model"], "effort": m["effort"], "summary": bool(m.get("summary")),
                    "plan": bool(m.get("current_plan")), "runs": len(runs), "active_job": job.id if job else None})
    return out


@api
async def create_project(request: Request):
    form = await request.form()
    pdf = form.get("pdf")
    if pdf is None or not getattr(pdf, "filename", ""):
        raise ValueError("choose a PDF file")
    data = await pdf.read()
    if not data.startswith(b"%PDF"):
        raise ValueError(f"{pdf.filename} is not a PDF")
    model = form.get("model") or P.default_model()
    effort = form.get("effort") or "low"
    if effort not in P.EFFORTS:
        raise ValueError(f"reasoning effort must be one of {P.EFFORTS}")
    project = P.Project.create(data, pdf.filename, model, effort, form.get("analyze_figures", "true") == "true",
                               max(1, min(60, int(form.get("max_figures") or 30))))
    job = JOBS.submit(project.id, "summarize", lambda say: P.run_summarize(project, say))
    return {"project_id": project.id, "job_id": job.id}


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
        project.save()
    job = JOBS.submit(project.id, "summarize", lambda say: P.run_summarize(project, say))
    return {"job_id": job.id}


@api
async def analyze(request: Request):
    project = _project(request)
    settings = P.Settings.parse(await request.json())
    P._check_systems(project, settings)  # fail now, not in the job, so the page can show it at once
    job = JOBS.submit(project.id, "analyze", lambda say: P.run_analyze(project, settings, say))
    return {"job_id": job.id}


@api
async def generate(request: Request):
    project = _project(request)
    settings = P.Settings.parse(await request.json())
    P._check_systems(project, settings)
    job = JOBS.submit(project.id, "generate", lambda say: P.run_generate(project, settings, say))
    return {"job_id": job.id}


@api
def get_job(request: Request):
    job = JOBS.jobs.get(request.path_params["jid"])
    if job is None:
        raise KeyError("no such job")
    return job.to_dict()


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
    Route("/api/projects/{pid}/analyze", analyze, methods=["POST"]),
    Route("/api/projects/{pid}/generate", generate, methods=["POST"]),
    Route("/api/projects/{pid}/runs/{run}", get_run),
    Route("/api/projects/{pid}/runs/{run}/zip", get_run_zip),
    Route("/api/projects/{pid}/runs/{run}/structures/{sid}/model", get_structure_model),
    Route("/api/projects/{pid}/files/{path:path}", get_file),
    Route("/api/jobs/{jid}", get_job),
    Mount("/static", StaticFiles(directory=HERE / "static"), name="static"),
])


def main() -> None:
    parser = argparse.ArgumentParser(description="Paper -> Structure web tool")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8600)
    args = parser.parse_args()
    ok, why = sandbox_available()
    print(f"[paper2structure] projects in {P.ROOT}; sandbox: {'bubblewrap' if ok else f'none ({why})'}", flush=True)
    print(f"[paper2structure] open http://{args.host}:{args.port}", flush=True)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
