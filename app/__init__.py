# app/__init__.py
import logging
from logging.handlers import RotatingFileHandler
import os

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

from .config import Config


def create_app() -> Flask:
    app = Flask(__name__, static_folder="static", static_url_path="/")
    app.config["MAX_CONTENT_LENGTH"] = Config.MAX_CONTENT_LENGTH

    # CORS
    CORS(app)

    # Logging setup
    _configure_logging(app)

    # Health check
    @app.route("/health", methods=["GET"])
    def health():
        return jsonify({
            "status": "ok",
            "training_root": str(Config.UPLOAD_TRAINING_ROOT),
            "dataset_root": str(Config.UPLOAD_DATASET_ROOT),
        })

    # Serve index.html for UI
    @app.route("/", methods=["GET"])
    def index():
        return send_from_directory(app.static_folder, "index.html")

    # Register blueprints
    from .routes.api import bp as api_bp
    app.register_blueprint(api_bp)

    # Request / response logging
    @app.before_request
    def log_request():
        app.logger.info(
            "REQ %s %s | args=%s | form=%s | json=%s | files=%s",
            request.method,
            request.path,
            dict(request.args),
            dict(request.form),
            (request.get_json(silent=True) if request.is_json else None),
            list(getattr(request.files, "keys", lambda: [])()),
        )

    @app.after_request
    def log_response(response):
        # Avoid touching response.data when in direct passthrough (prevents the Flask warning)
        if not response.direct_passthrough:
            try:
                body_preview = response.get_data(as_text=True)
                if len(body_preview) > 500:
                    body_preview = body_preview[:500] + "... [truncated]"
            except Exception:
                body_preview = "<unable to read body>"
        else:
            body_preview = "<passthrough>"

        app.logger.info(
            "RESP %s %s | status=%s | body=%s",
            request.method,
            request.path,
            response.status_code,
            body_preview,
        )
        return response

    return app


def _configure_logging(app: Flask) -> None:
    # Base logger
    app.logger.setLevel(logging.INFO)

    # File handler (rotating logs)
    file_handler = RotatingFileHandler(
        Config.LOG_FILE, maxBytes=5 * 1024 * 1024, backupCount=5, encoding="utf-8"
    )
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "[%(asctime)s] %(levelname)s in %(module)s: %(message)s"
    )
    file_handler.setFormatter(formatter)
    app.logger.addHandler(file_handler)

    # Also log to stderr (optional)
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)
    app.logger.addHandler(stream_handler)
