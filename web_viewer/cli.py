from __future__ import annotations

import argparse

from .config import ViewerConfig
from .server import create_app


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the JEPA experiment web viewer.")
    parser.add_argument("--output-dir", default="out.jepa_world_model_trainer", help="Directory containing experiment runs.")
    parser.add_argument("--host", default="127.0.0.1", help="Host/IP for the Flask server.")
    parser.add_argument("--port", type=int, default=5001, help="Port for the Flask server.")
    parser.add_argument("--debug", action="store_true", help="Enable Flask debug mode.")
    args = parser.parse_args()
    cfg = ViewerConfig.from_env(output_dir=args.output_dir, host=args.host, port=args.port, debug=args.debug)
    app = create_app(cfg)
    app.run(host=cfg.host, port=cfg.port, debug=cfg.debug)


if __name__ == "__main__":
    main()
