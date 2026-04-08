"""FastAPI app for the Medical Coding Assistant environment."""

try:
    from openenv.core.env_server.http_server import create_app
except Exception as exc:  # pragma: no cover
    raise ImportError(
        "openenv is required for the web interface. Install dependencies with 'uv sync'."
    ) from exc

try:
    from ..models import MedicalCodingAction, MedicalCodingObservation
    from .medical_coding_environment import MedicalCodingEnvironment
except (ImportError, ModuleNotFoundError):
    from models import MedicalCodingAction, MedicalCodingObservation
    from server.medical_coding_environment import MedicalCodingEnvironment


app = create_app(
    MedicalCodingEnvironment,
    MedicalCodingAction,
    MedicalCodingObservation,
    env_name="medical-coding-assistant",
    max_concurrent_envs=4,
)


def main() -> None:
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
