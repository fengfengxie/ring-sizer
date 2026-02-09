#!/usr/bin/env python3
"""Simple web demo for ring-size-cv.

Upload an image, run measurement, and return JSON + debug overlay.
"""

from __future__ import annotations

import json
import sys
import uuid
from pathlib import Path
from typing import Dict, Any

import cv2
from flask import Flask, jsonify, render_template, request, send_from_directory
from werkzeug.utils import secure_filename

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from measure_finger import measure_finger

APP_ROOT = Path(__file__).resolve().parent
UPLOAD_DIR = APP_ROOT / "uploads"
RESULTS_DIR = APP_ROOT / "results"
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}

app = Flask(__name__)


def _allowed_file(filename: str) -> bool:
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS


def _save_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/results/<path:filename>")
def serve_result(filename: str):
    return send_from_directory(RESULTS_DIR, filename)


@app.route("/uploads/<path:filename>")
def serve_upload(filename: str):
    return send_from_directory(UPLOAD_DIR, filename)


@app.route("/api/measure", methods=["POST"])
def api_measure():
    if "image" not in request.files:
        return jsonify({"success": False, "error": "Missing image file"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"success": False, "error": "Empty filename"}), 400

    if not _allowed_file(file.filename):
        return jsonify({"success": False, "error": "Unsupported file type"}), 400

    finger_index = request.form.get("finger_index", "index")
    edge_method = request.form.get("edge_method", "auto")

    run_id = uuid.uuid4().hex[:12]
    safe_name = secure_filename(file.filename)
    upload_name = f"{run_id}__{safe_name}"
    upload_path = UPLOAD_DIR / upload_name
    upload_path.parent.mkdir(parents=True, exist_ok=True)
    file.save(upload_path)

    image = cv2.imread(str(upload_path))
    if image is None:
        return jsonify({"success": False, "error": "Failed to load image"}), 400

    debug_name = f"{run_id}__debug.png"
    debug_path = RESULTS_DIR / debug_name

    result = measure_finger(
        image=image,
        finger_index=finger_index,
        edge_method=edge_method,
        debug_path=str(debug_path),
    )

    result_json_name = f"{run_id}__result.json"
    result_json_path = RESULTS_DIR / result_json_name
    _save_json(result_json_path, result)

    payload = {
        "success": result.get("fail_reason") is None,
        "result": result,
        "debug_image_url": f"/results/{debug_name}",
        "input_image_url": f"/uploads/{upload_name}",
        "result_json_url": f"/results/{result_json_name}",
    }

    return jsonify(payload)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
