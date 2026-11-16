"""Upload validation: extension, size and 'magic bytes' must all agree before we decode anything."""
from __future__ import annotations

from pathlib import PurePath

ALLOWED_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".pdf"}
MAX_UPLOAD_MB = 15

_MAGIC = {
    ".png": [b"\x89PNG\r\n\x1a\n"],
    ".jpg": [b"\xff\xd8\xff"],
    ".jpeg": [b"\xff\xd8\xff"],
    ".tif": [b"II*\x00", b"MM\x00*"],
    ".tiff": [b"II*\x00", b"MM\x00*"],
    ".pdf": [b"%PDF"],
}


def validate_upload(filename: str, data: bytes, max_mb: int = MAX_UPLOAD_MB):
    """Return (ok, message). Never raises."""
    ext = PurePath(filename or "").suffix.lower()
    if ext not in ALLOWED_EXTS:
        return False, f"Unsupported file type '{ext or 'none'}'. Please upload PNG, JPG, JPEG, TIFF or PDF."
    if not data:
        return False, "The file is empty (0 bytes)."
    if len(data) > max_mb * 1024 * 1024:
        return False, (f"File is {len(data) / 1024 / 1024:.1f} MB; the limit is {max_mb} MB. "
                       "Please upload a smaller or lower-resolution file.")
    if not any(data.startswith(sig) for sig in _MAGIC[ext]):
        return False, (f"The content of this file does not look like a real {ext.upper().lstrip('.')} "
                       "(wrong file signature). It was rejected for safety.")
    return True, "OK"
