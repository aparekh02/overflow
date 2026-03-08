"""
FastAPI application for the Overflow Environment.

Exposes the OverflowEnvironment over HTTP and WebSocket endpoints.

Usage:
    uvicorn server.app:app --reload --host 0.0.0.0 --port 8000
"""

import inspect

from openenv.core.env_server.http_server import create_app

from ..models import OverflowAction, OverflowObservation
from .overflow_environment import OverflowEnvironment


def _create_overflow_app():
    """Build app across create_app variants that may expect a factory or an instance."""
    try:
        first_param = next(iter(inspect.signature(create_app).parameters.values()))
        annotation_text = str(first_param.annotation)
    except (StopIteration, TypeError, ValueError):
        annotation_text = "typing.Callable"

    expects_instance = (
        "Environment" in annotation_text and "Callable" not in annotation_text
    )
    env_arg = OverflowEnvironment() if expects_instance else OverflowEnvironment
    return create_app(
        env_arg, OverflowAction, OverflowObservation, env_name="overflow_env"
    )


app = _create_overflow_app()


def main():
    """Entry point for direct execution via uv run or python -m."""
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
