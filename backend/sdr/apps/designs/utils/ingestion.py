"""
Document ingestion for design uploads.

- PDFs are stored as-is.
This module is used for PDFs and for design document replace.
"""

from typing import Tuple

from fastapi import UploadFile

from ..models import Design


def process_incoming_document(file: UploadFile) -> Tuple[UploadFile, str, str]:
    """
    Process an uploaded design document: pass through PDF as-is.

    Args:
        file: The uploaded file (PDF).

    Returns:
        Tuple of (file_to_save, source_format, original_filename).
        file_to_save: The same file to store in Design.document.
        source_format: Design.SOURCE_FORMAT_PDF.
        original_filename: Original upload name (for display), or empty for PDF.
    """
    name_lower = (file.filename or "").lower()
    original_filename = file.filename or ""

    if name_lower.endswith(".pdf"):
        return (file, Design.SOURCE_FORMAT_PDF, "")

    return (file, Design.SOURCE_FORMAT_PDF, "")
