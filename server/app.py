# server/app.py — Root-level wrapper for OpenEnv compatibility
# The actual server implementation lives in src/server/app.py.
# This file re-exports the FastAPI app so that openenv validate can find it.

import sys
import os

# Ensure src/ is in the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from server.app import app  # noqa: E402, F401


def main():
    """Entry point for openenv serve / project.scripts."""
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    host = os.getenv("HOST", "0.0.0.0")
    uvicorn.run(app, host=host, port=port, workers=1)


if __name__ == "__main__":
    main()
