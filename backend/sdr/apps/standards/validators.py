import logging
import os
import re
from typing import List

from fastapi import UploadFile

logger = logging.getLogger(__name__)

try:
    import magic
    MAGIC_AVAILABLE = True
except ImportError:
    MAGIC_AVAILABLE = False
    logger.warning(
        "python-magic is not installed. File validation will fall back to"
        "extension-only checks. Install python-magic for production use."
    )

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

MAX_UPLOAD_FILES = 5
MAX_FILE_SIZE_MB = 20
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024
MAX_NAME_LENGTH = 255
MAX_FILENAME_LENGTH = 255

_ALLOWED_EXTENSIONS = {".pdf", ".doc", ".docx"}

_ALLOWED_MIMES: dict[str, list[bytes]] = {
    "application/pdf": [b"%PDF"],
    "application/msword": [b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"],
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": [b"PK\x03\x04"],
}

_VALID_NAME_RE = re.compile(r"^[\w\s\-\.\(\)/:#&,]+$", re.UNICODE)

_SUSPICIOUS_FILENAME_PATTERNS = frozenset([
    "script", "executable", ".bat", ".cmd", ".exe",
    ".scr", ".pif", ".com", "javascript", "vbscript",
    "powershell", ".shell",
])


# ---------------------------------------------------------------------------
# File validation
# ---------------------------------------------------------------------------

def validate_standard_file(file: UploadFile) -> bool:
    if file.size == 0:
        raise ValueError("File cannot be empty.")

    if file.size and file.size > MAX_FILE_SIZE_BYTES:
        raise ValueError(f"File size must be less than {MAX_FILE_SIZE_MB}MB.")
    _validate_filename(file.filename)

    file_mime: str | None = None

    if MAGIC_AVAILABLE:
        file_mime = _validate_mime_with_magic(file)
    else:
        _validate_extension_fallback(file.filename)

    if file_mime == "application/pdf" or (
        file_mime is None and file.filename.lower().endswith(".pdf")
    ):
        _validate_pdf_structure(file)

    return True


def _validate_filename(filename: str) -> None:
    if len(filename) > MAX_FILENAME_LENGTH:
        raise ValueError(f"File name is too long. Maximum {MAX_FILENAME_LENGTH} characters allowed.")

    illegal_chars = {"..", "/", "\\", "<", ">", ":", '"', "|", "?", "*"}
    if any(char in filename for char in illegal_chars):
        raise ValueError("Invalid file name. Please use a safe file name.")

    name_lower = filename.lower()
    if any(pattern in name_lower for pattern in _SUSPICIOUS_FILENAME_PATTERNS):
        raise ValueError("File name contains suspicious patterns. Please use a descriptive name.")


def _validate_mime_with_magic(file: UploadFile) -> str:
    try:
        content = file.file.read(1024)
        file_mime = magic.from_buffer(content, mime=True)
        file.file.seek(0)
    except Exception as exc:
        logger.warning(
            "_validate_mime_with_magic: python-magic failed for '%s': %s. "
            "Falling back to extension check.",
            file.filename,
            exc,
        )
        _validate_extension_fallback(file.filename)
        return None

    if file_mime not in _ALLOWED_MIMES:
        raise ValueError("File must be a PDF, DOC, or DOCX document.")

    file_header = file.file.read(8)
    file.file.seek(0)

    if not any(file_header.startswith(magic_bytes) for magic_bytes in _ALLOWED_MIMES[file_mime]):
        raise ValueError("File content does not match its declared type.")

    return file_mime


def _validate_extension_fallback(filename: str) -> None:
    ext = os.path.splitext(filename.lower())[1]
    if ext not in _ALLOWED_EXTENSIONS:
        raise ValueError("File must be a PDF, DOC, or DOCX document.")


def _validate_pdf_structure(file: UploadFile) -> None:
    try:
        header = file.file.read(1024)
        file.file.seek(0)
        if not header.startswith(b"%PDF"):
            raise ValueError("Invalid PDF structure.")
    except ValueError:
        raise
    except Exception as exc:
        logger.warning("_validate_pdf_structure: could not read PDF header: %s", exc)
        raise ValueError("Unable to validate PDF content.")

def validate_standard_name(name: str) -> str:
    if not name or not name.strip():
        raise ValueError("Standard name cannot be empty.")

    name = name.strip()

    if len(name) > MAX_NAME_LENGTH:
        raise ValueError(f"Standard name is too long. Maximum {MAX_NAME_LENGTH} characters allowed.")

    if not _VALID_NAME_RE.match(name):
        raise ValueError(
            "Standard name contains invalid characters. "
            "Only letters, numbers, spaces, hyphens, dots, underscores, "
            "parentheses, slashes, and colons are allowed."
        )

    return name