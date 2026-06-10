import os
from typing import List
from fastapi import UploadFile

try:
    import magic
    MAGIC_AVAILABLE = True
except ImportError:
    MAGIC_AVAILABLE = False

# Allowed design document formats: PDF
ALLOWED_DESIGN_EXTENSIONS = (".pdf",)
ALLOWED_DESIGN_MIME_TYPES = (
    "application/pdf",
)
MAX_DESIGN_FILE_SIZE = 10 * 1024 * 1024  # 10MB


def validate_design_document_file(file: UploadFile):
    """
    Validate design document: only .pdf allowed.
    """
    if file.size and file.size > MAX_DESIGN_FILE_SIZE:
        raise ValueError("File size must be less than 10MB.")

    if file.size == 0:
        raise ValueError("File cannot be empty.")

    name_lower = file.filename.lower()
    if not any(name_lower.endswith(ext) for ext in ALLOWED_DESIGN_EXTENSIONS):
        raise ValueError("File must be a PDF document (.pdf).")

    if MAGIC_AVAILABLE:
        try:
            content = file.file.read(1024)
            file.file.seek(0)
            file_mime = magic.from_buffer(content, mime=True)
            if file_mime not in ALLOWED_DESIGN_MIME_TYPES:
                raise ValueError("File must be a valid PDF document.")
        except Exception as e:
            if isinstance(e, ValueError):
                raise
            if not any(name_lower.endswith(ext) for ext in ALLOWED_DESIGN_EXTENSIONS):
                raise ValueError("File must be a PDF document (.pdf).")

    if any(char in file.filename for char in ["..", "/", "\\", "<", ">", ":", '"', "|", "?", "*"]):
        raise ValueError("Invalid file name. Please use a safe file name.")

    if len(file.filename) > 255:
        raise ValueError("File name is too long. Maximum 255 characters allowed.")


def validate_pdf_file(file: UploadFile):
    """
    Comprehensive PDF file validation including MIME type and file size.
    Kept for backward compatibility where only PDF is accepted.
    """
    if file.size and file.size > MAX_DESIGN_FILE_SIZE:
        raise ValueError("File size must be less than 10MB.")

    if file.size == 0:
        raise ValueError("File cannot be empty.")

    if MAGIC_AVAILABLE:
        try:
            content = file.file.read(1024)
            file.file.seek(0)
            file_mime = magic.from_buffer(content, mime=True)
            if file_mime != "application/pdf":
                raise ValueError("File must be a valid PDF document.")
        except Exception:
            if not file.filename.lower().endswith(".pdf"):
                raise ValueError("File must be a PDF document.")
    else:
        if not file.filename.lower().endswith(".pdf"):
            raise ValueError("File must be a PDF document.")

    if any(char in file.filename for char in ["..", "/", "\\", "<", ">", ":", '"', "|", "?", "*"]):
        raise ValueError("Invalid file name. Please use a safe file name.")

    if len(file.filename) > 255:
        raise ValueError("File name is too long. Maximum 255 characters allowed.")


def validate_file_upload(files: List[UploadFile]):
    """
    Validate multiple design document uploads (PDF).
    """
    if not files:
        raise ValueError("No files provided.")

    if len(files) > 10:
        raise ValueError("Maximum 10 files allowed per upload.")

    for file in files:
        validate_design_document_file(file)
