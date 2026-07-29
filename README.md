# dopa

## Authenticated scoring API

The FastAPI service accepts one MP4 or MOV ad, runs the pinned TRIBE v2
video-only model and average-CTR regressor, and returns a short-lived cortical
surface model plus the five most responsive Destrieux regions.

It requires a CUDA host, the trained regressor checkpoint, `ffprobe`,
and a Python 3.12 environment:

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt -r requirements.api.txt
cp .env.example .env
# Edit DOPA_MODEL_PATH and any host-specific paths, then load .env.
set -a
source .env
set +a
.venv/bin/uvicorn dopa_api.app:app --host 127.0.0.1 --port 8000
```

`DOPA_ALLOWED_ORIGINS` defaults to the two localhost forms on port 3000.
Production origins must be listed explicitly. Supabase access tokens are
verified against the project’s public ES256 JWKS; no service-role key belongs
in this service.

The CPU-only API suite uses injected scorer and renderer output:

```bash
uv pip install --python .venv/bin/python -r requirements.test.txt
.venv/bin/python -m pytest -q
```

TRIBE v2 is used with attribution under CC BY-NC 4.0. This integration is for
the selected non-commercial demo use.
