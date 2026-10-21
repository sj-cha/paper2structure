import os
import warnings

os.environ.setdefault("PYDANTIC_AI_NO_BANNER", "1")
os.environ.setdefault("P2S_NEIGHBOR_BACKEND", "ase")  # unit tests skip the torch/warp start-up cost
warnings.filterwarnings("ignore")

import pytest  # noqa: E402
from pydantic_ai import models  # noqa: E402

from paper2structure.tools._geometry import ensure_warp_cache  # noqa: E402

ensure_warp_cache()  # before anything imports warp/nvalchemiops
models.ALLOW_MODEL_REQUESTS = False  # only tests marked `live` may call a real LLM


@pytest.fixture(autouse=True)
def _writable_caches(tmp_path_factory, monkeypatch):
    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path_factory.getbasetemp() / "mpl"))
