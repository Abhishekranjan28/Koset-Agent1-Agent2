# app/routes/api.py
import os
from pathlib import Path
from typing import Dict, Any

from flask import Blueprint, current_app, request, jsonify
from werkzeug.utils import secure_filename

from ..config import Config
from ..utils.storage import save_files
from ..core.original_logic import (
    analyze_script_source_with_hf,
    agent2_data_profiler,
    estimate_training_cost_v2,
    refine_classical_model_estimates,
)

import pandas as pd
from PIL import Image

bp = Blueprint("api", __name__)


# ---------- helpers ----------

def _profile_single_file(p: Path) -> Dict[str, Any]:
    """
    Wraps folder profiler but narrows to this specific file when possible.
    """
    base_prof = agent2_data_profiler(str(p.parent))
    suffix = p.suffix.lower()

    # Tabular
    if suffix in {".csv", ".tsv", ".parquet", ".xlsx"}:
        if suffix in {".csv", ".tsv"}:
            df = pd.read_csv(p)
        elif suffix == ".xlsx":
            df = pd.read_excel(p)
        else:
            df = pd.read_parquet(p)

        return {
            "tabular": {
                "data_type": "tabular",
                "num_samples": int(len(df)),
                "num_features": int(df.shape[1]),
                "avg_sample_size_bytes": float(
                    df.memory_usage(deep=True).sum() / max(1, len(df))
                ),
                "total_size_bytes_estimate": int(df.memory_usage(deep=True).sum()),
            }
        }

    # Image
    if suffix in {".png", ".jpg", ".jpeg", ".bmp"}:
        with Image.open(p) as img:
            w, h = img.size
            c = len(img.getbands())
        size_bytes = os.path.getsize(p)
        return {
            "image": {
                "data_type": "image",
                "num_samples": 1,
                "input_shape": [int(c), int(h), int(w)],
                "avg_sample_size_bytes": float(size_bytes),
                "total_size_bytes_estimate": int(size_bytes),
            }
        }

    # Text
    if suffix in {".txt", ".md"}:
        seq_lengths, text_sizes = [], []
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                seq_lengths.append(len(line.strip().split()))
                text_sizes.append(len(line.encode("utf-8")))
        return {
            "text": {
                "data_type": "text",
                "num_samples": int(len(seq_lengths)),
                "avg_seq_len_tokens": float(
                    sum(seq_lengths) / max(1, len(seq_lengths))
                ) if seq_lengths else 0.0,
                "max_seq_len_tokens": int(max(seq_lengths)) if seq_lengths else 0,
                "avg_sample_size_bytes": float(
                    sum(text_sizes) / max(1, len(text_sizes))
                ) if text_sizes else 0.0,
                "total_size_bytes_estimate": int(sum(text_sizes)),
            }
        }

    # Fallback to whole-folder profile
    return base_prof


def _find_training_source_in_dir(training_dir: Path) -> str | None:
    """
    Fallback for manual/advanced usage if a directory is passed to /analyze.

    Priority:
    1) Explicit ENTRYPOINT_CANDIDATES (by stem, ANY extension; so train.py or train.ipynb)
    2) Any .ipynb
    3) Any .py
    """
    candidate_stems = {Path(n).stem.lower() for n in Config.ENTRYPOINT_CANDIDATES}

    # 1) Named entrypoints (any extension if stem matches)
    for dirpath, _, filenames in os.walk(training_dir):
        for fn in filenames:
            stem = Path(fn).stem.lower()
            if stem in candidate_stems:
                return str(Path(dirpath) / fn)

    # 2) Any notebook
    ipynb_files = list(training_dir.rglob("*.ipynb"))
    if ipynb_files:
        return str(ipynb_files[0])

    # 3) Any .py
    py_files = list(training_dir.rglob("*.py"))
    if py_files:
        return str(py_files[0])

    return None


# =====================================================
# 1) POST /upload
# =====================================================
@bp.route("/upload", methods=["POST"])
def upload():
    """
    multipart/form-data:
      training_files: <files> (optional)
      dataset_files : <files> (optional)
      project_name  : (optional)
      dataset_name  : (optional)

    IMPORTANT:
      - We only compute "primary" files from THIS upload, not from existing files in the folder.
      - For remote / URL-based sources (GitHub, HF, raw .py, etc.) you can skip this
        and call POST /analyze directly with a "source" string.
    """
    try:
        training_files = request.files.getlist("training_files")
        dataset_files = request.files.getlist("dataset_files")

        if not training_files and not dataset_files:
            return jsonify({"error": "No files provided (training_files/dataset_files)"}), 400

        project_name = request.form.get("project_name", "default_project")
        dataset_name = request.form.get("dataset_name", "default_dataset")

        training_info = None
        dataset_info = None

        # ---- training upload ----
        if training_files:
            training_target = Config.UPLOAD_TRAINING_ROOT / secure_filename(project_name)
            saved_tr, skipped_tr = save_files(
                training_files,
                training_target,
                Config.TRAINING_ALLOWED_EXTS,
                unzip=True,
            )

            # choose primary training file ONLY from saved_tr (not scanning whole folder)
            candidate_stems = {Path(n).stem.lower() for n in Config.ENTRYPOINT_CANDIDATES}

            entrypoints = []
            for path_str in saved_tr:
                p = Path(path_str)
                if p.stem.lower() in candidate_stems and p.suffix.lower() in {".py", ".ipynb"}:
                    entrypoints.append(path_str)

            if entrypoints:
                training_primary_file = entrypoints[0]
            else:
                training_primary_file = saved_tr[0] if saved_tr else None

            training_info = {
                "project_name": project_name,
                "training_path": str(training_target),
                "saved_files": saved_tr,
                "skipped_files": skipped_tr,
                "entrypoint_candidates": entrypoints,
                "training_primary_file": training_primary_file,
            }

        # ---- dataset upload ----
        if dataset_files:
            dataset_target = Config.UPLOAD_DATASET_ROOT / secure_filename(dataset_name)
            saved_ds, skipped_ds = save_files(
                dataset_files,
                dataset_target,
                Config.DATASET_ALLOWED_EXTS,
                unzip=True,
            )

            dataset_primary_file = saved_ds[0] if saved_ds else None

            dataset_info = {
                "dataset_name": dataset_name,
                "dataset_path": str(dataset_target),
                "saved_files": saved_ds,
                "skipped_files": skipped_ds,
                "dataset_primary_file": dataset_primary_file,
            }

        resp = {
            "message": "Upload complete",
            "training": training_info,
            "dataset": dataset_info,
        }
        current_app.logger.info("UPLOAD result: %s", resp)
        return jsonify(resp), 201

    except Exception as e:
        current_app.logger.exception("UPLOAD ERROR")
        return jsonify({"error": str(e)}), 500


# =====================================================
# 2) POST /analyze
# =====================================================
@bp.route("/analyze", methods=["POST"])
def analyze():
    """
    JSON body (UI flow):

    Preferred UI usage (uploaded files):
    {
      "training_path": "<FILE path returned as training_primary_file>",
      "dataset_path": "<FILE path returned as dataset_primary_file>",  # optional
      "use_llm_fallback": true/false
    }

    Advanced usage (Intelligent Script Loader via `source`):
      "source" can be:
        - Local path to .py or .ipynb
        - Raw JSON config string (hyperparameters)
        - GitHub/GitLab repository URL (will clone & search train.py/main.py/finetune.py)
        - Hugging Face model URL: https://huggingface.co/<org>/<model>
        - Raw .py URL over HTTP(S)
        - Cloud storage placeholder (s3://, gs://, etc.) -> handled as a stub

    Rules:
      - If "source" is provided, we use it directly and do NOT require training_path.
      - If "source" is absent, we resolve from "training_path" (file or directory).
    """
    payload = request.get_json(force=True, silent=True) or {}
    current_app.logger.info("ANALYZE request payload: %s", payload)

    try:
        use_llm = bool(payload.get("use_llm_fallback", True))
        overrides = payload.get("overrides") or {}

        source = payload.get("source")
        training_path = payload.get("training_path")
        dataset_path = payload.get("dataset_path")

        # ----- resolve training source (.py or .ipynb or URL/string) -----
        if not source:
            if not training_path:
                return jsonify({"error": "Provide either 'source' or 'training_path'"}), 400

            tpath = Path(training_path)
            if not tpath.exists():
                return jsonify({"error": f"'training_path' does not exist: {training_path}"}), 400

            # Normal UI flow: training_path is a FILE that was just uploaded
            if tpath.is_file():
                source = str(tpath)
            else:
                # Advanced/legacy: directory mode (not used by normal UI)
                resolved = _find_training_source_in_dir(tpath)
                if not resolved:
                    return jsonify({
                        "error": "No .py or .ipynb entrypoint found under 'training_path' directory"
                    }), 400
                source = resolved

        # ----- resolve data profile -----
        data_profile = None
        scoped_to = None
        if dataset_path:
            dpath = Path(dataset_path)
            if not dpath.exists():
                return jsonify({"error": f"'dataset_path' does not exist: {dataset_path}"}), 400

            if dpath.is_dir():
                # fallback if someone passes a dir
                data_profile = agent2_data_profiler(str(dpath))
            elif dpath.is_file():
                data_profile = _profile_single_file(dpath)
                scoped_to = str(dpath)

        # ----- agent1: script analysis (handles .py AND .ipynb AND all intelligent source types) -----
        agent1_raw = analyze_script_source_with_hf(source, use_llm_fallback=use_llm)
        agent1_result = agent1_raw["result"]

        # ----- refine for classical ML -----
        agent1_refined = refine_classical_model_estimates(agent1_result, data_profile)

        # apply overrides
        for key in ("epochs", "batch_size", "params_total", "trainable_params"):
            if key in overrides and overrides[key] is not None:
                agent1_refined[key] = overrides[key]

        # ----- cost estimation (ONLY v2: FLOPs/TFLOPs model) -----
        profile_for_cost = data_profile or {}
        v2 = estimate_training_cost_v2(agent1_refined, profile_for_cost)

        resp = {
            "source_used": source,
            "training_path": training_path,
            "dataset_path": dataset_path,
            "dataset_scoped_to": scoped_to,
            "agent1_result_refined": agent1_refined,
            "data_profile": data_profile,
            "estimates_v2": v2,  # only v2 returned
        }
        current_app.logger.info(
            "ANALYZE result: source=%s params=%s epochs=%s batch_size=%s",
            source,
            agent1_refined.get("params_total"),
            agent1_refined.get("epochs"),
            agent1_refined.get("batch_size"),
        )
        return jsonify(resp), 200

    except Exception:
        # Log full traceback server-side
        current_app.logger.exception("ANALYZE ERROR")
        # UI-friendly message
        return jsonify({"error": "Not able to recommend"}), 500
