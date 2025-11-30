# app/utils/storage.py
from __future__ import annotations
from pathlib import Path
from typing import Iterable, Tuple, Set, List
from werkzeug.utils import secure_filename
import zipfile


def allowed(filename: str, allowed_exts: Set[str]) -> bool:
    return Path(filename).suffix.lower() in allowed_exts


def _safe_extractall(zf: zipfile.ZipFile, dest_dir: Path) -> None:
    for member in zf.infolist():
        member_path = dest_dir / member.filename
        # prevent zip-slip
        if not str(member_path.resolve()).startswith(str(dest_dir.resolve())):
            continue
        zf.extract(member, dest_dir)


def save_files(
    files: Iterable,
    dest_dir: Path,
    allowed_exts: Set[str],
    unzip: bool = True
) -> Tuple[List[str], List[str]]:
    dest_dir.mkdir(parents=True, exist_ok=True)
    saved, skipped = [], []

    for f in files:
        fname = secure_filename(getattr(f, "filename", "") or "")
        if not fname:
            continue
        if not allowed(fname, allowed_exts):
            skipped.append(fname)
            continue

        out_path = dest_dir / fname
        f.save(str(out_path))
        saved.append(str(out_path))

        if unzip and out_path.suffix.lower() == ".zip":
            try:
                with zipfile.ZipFile(out_path, "r") as zf:
                    _safe_extractall(zf, dest_dir)
            except zipfile.BadZipFile:
                # silently skip bad zips
                pass

    return saved, skipped
