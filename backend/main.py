"""Entrypoint for Vercel / serverless deployments."""

import sys
from pathlib import Path

# Add backend/src and repo root to Python path
_backend_dir = Path(__file__).resolve().parent
_src_dir = _backend_dir / "src"
_repo_dir = _backend_dir.parent
for p in (_src_dir, _repo_dir):
    p_str = str(p)
    if p_str not in sys.path:
        sys.path.insert(0, p_str)

from retinaguard.api.main import app  # noqa: E402, F401
