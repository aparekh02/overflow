"""
FastAPI application for the APEX-Agents Environment.

Exposes the ApexAgentsEnvironment over HTTP and WebSocket endpoints.

Usage:
    uvicorn server.app:app --reload --host 0.0.0.0 --port 8000
"""

import inspect

from openenv.core.env_server.http_server import create_app

from ..models import ApexAction, ApexObservation
from .apex_agents_environment import ApexAgentsEnvironment


def _create_apex_app():
    """Build app across create_app variants that may expect a factory or an instance."""
    try:
        first_param = next(iter(inspect.signature(create_app).parameters.values()))
        annotation_text = str(first_param.annotation)
    except (StopIteration, TypeError, ValueError):
        annotation_text = "typing.Callable"

    expects_instance = (
        "Environment" in annotation_text and "Callable" not in annotation_text
    )
    env_arg = ApexAgentsEnvironment() if expects_instance else ApexAgentsEnvironment
    return create_app(
        env_arg,
        ApexAction,
        ApexObservation,
        env_name="apex_agents_env",
        max_concurrent_envs=8,
    )


app = _create_apex_app()


def main():
    """Entry point for direct execution via uv run or python -m."""
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
