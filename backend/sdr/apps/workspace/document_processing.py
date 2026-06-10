import os
import re
import logging
import tempfile
import inspect
from contextlib import contextmanager
from email import policy
from email.parser import BytesParser
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional
from bs4 import BeautifulSoup
import fitz
import base64

import pdfminer.high_level
import pytesseract
from pdf2image import convert_from_path
from sdr.core.config import settings
from sdr.apps.workspace.services.storage import storage_service

logger = logging.getLogger(__name__)

try:
    import pymupdf4llm

    PYMUPDF4LLM_AVAILABLE = True
except ImportError:
    pymupdf4llm = None
    PYMUPDF4LLM_AVAILABLE = False
    logger.warning(
        "pymupdf4llm is not installed. PDF ingestion will fall back to "
        "PyMuPDF plain-text extraction."
    )


@contextmanager
def get_local_file_path(file_path: str):
    """
    Context manager that provides a local file path.
    Downloads the file from MinIO to a temporary location, yields the path,
    and then cleans up the temporary file.

    Usage:
        with get_local_file_path(standard.document) as fp:
            text = get_text_from_pdf(fp)

    Args:
        file_path: Relative or absolute string path.

    Yields:
        str: Local file path
    """
    if not file_path:
        yield ""
        return
        
    path_str = str(file_path)
    
    # If it's already an absolute path, just yield it
    if path_str.startswith('/') or os.path.isabs(path_str):
        logger.info(f"Using existing absolute file path: {path_str}")
        yield path_str
        return
        
    # Otherwise, fetch from MinIO
    # Create a temporary file
    fd, temp_path = tempfile.mkstemp(suffix=".pdf")
    os.close(fd) # Close the file descriptor so Minio/others can write to it
    
    try:
        logger.info(f"Downloading {path_str} from MinIO to temporary file {temp_path}")
        storage_service.download_to_file(path_str, temp_path)
        yield temp_path
    finally:
        # Clean up the temporary file after the context manager exits
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
                logger.debug(f"Cleaned up temporary file {temp_path}")
            except Exception as e:
                logger.warning(f"Failed to delete temporary file {temp_path}: {e}")


SECTION_REGEX = re.compile(r"^(\d+(?:\.\d+)*|\w(?:\.\d+)*)\s+(.+)$")


def extract_sections_from_lines(lines: List[str]) -> List[str]:
    sections: List[str] = []
    for line in lines:
        match = SECTION_REGEX.match(line)
        if match:
            sections.append(f"{match.group(1)} {match.group(2).strip()}")
        elif line.isupper() and len(line.split()) <= 6:
            sections.append(line.title())
    return sections


def clean_tokens(text: str) -> List[str]:
    tokens = re.findall(r"[a-zA-Z0-9]{4,}", text.lower())
    return tokens[:30]


def extract_snippet(page_text: str, tokens: List[str]) -> str:
    if not page_text:
        return ""
    lines = [line.strip() for line in page_text.splitlines() if line.strip()]
    for line in lines:
        lower_line = line.lower()
        if any(token in lower_line for token in tokens):
            return line[:240]
    return lines[0][:240] if lines else ""


def pick_section(page: Dict[str, Any], tokens: List[str]) -> Optional[str]:
    for section in page.get("sections", []):
        lower_section = section.lower()
        if any(token in lower_section for token in tokens):
            return section
    return page.get("sections", [None])[0] if page.get("sections") else None


def default_location_result() -> Dict[str, Any]:
    return {
        "label": "Document analysis required",
        "navigation": {
            "page": None,
            "section": None,
            "quote": None,
            "confidence_score": 0.0,
            "extraction_method": "fallback",
            "navigation_url": "#page=1",
        },
    }


def locate_text_in_document(search_text: str, design_pages: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Locate the best matching page/section for the provided text."""
    if not design_pages or not search_text:
        return default_location_result()

    tokens = clean_tokens(search_text)
    if not tokens:
        return default_location_result()

    best_match = None
    for page in design_pages:
        lower_text = page["lower_text"]
        token_hits = sum(lower_text.count(token) for token in tokens)
        similarity = SequenceMatcher(None, lower_text[:2000], search_text.lower()[:2000]).ratio()
        score = token_hits + similarity * 5

        if not best_match or score > best_match["score"]:
            snippet = extract_snippet(page["text"], tokens)
            section = pick_section(page, tokens)
            best_match = {
                "score": score,
                "page": page["page_number"],
                "section": section,
                "quote": snippet,
            }

    if not best_match or best_match["score"] == 0:
        return default_location_result()

    label_parts = [f"Page {best_match['page']}"]
    if best_match["section"]:
        label_parts.append(f"Section {best_match['section']}")
    if best_match["quote"]:
        label_parts.append(f"Quote: '{best_match['quote']}'")

    confidence = round(min(1.0, best_match["score"] / (len(tokens) + 1)), 2)

    return {
        "label": " | ".join(label_parts),
        "navigation": {
            "page": best_match["page"],
            "section": best_match["section"],
            "quote": best_match["quote"],
            "confidence_score": confidence,
            "extraction_method": "page_keyword_search",
            "navigation_url": f"#page={best_match['page']}",
        },
    }


def prepare_design_document(raw_text: str) -> Dict[str, Any]:
    """Split a design document into page-aware chunks and add annotations."""
    if not raw_text:
        return {"annotated_text": "", "pages": []}

    page_chunks = [chunk.strip() for chunk in re.split(r"\f+", raw_text) if chunk.strip()]
    pages = []

    for idx, chunk in enumerate(page_chunks, start=1):
        lines = [line.strip() for line in chunk.splitlines() if line.strip()]
        sections = extract_sections_from_lines(lines)
        pages.append(
            {
                "page_number": idx,
                "text": chunk,
                "lower_text": chunk.lower(),
                "lines": lines,
                "sections": sections,
            }
        )

    annotated_text = "\n\n".join(
        f"[PAGE {page['page_number']}]\n{page['text']}" for page in pages
    )

    return {"annotated_text": annotated_text or raw_text, "pages": pages}


def _call_pymupdf4llm_to_markdown(
    file_path: str,
    *,
    write_images: bool = False,
    image_output_dir: Optional[str] = None,
    start_page: Optional[int] = None,
    end_page: Optional[int] = None,
):
    if not PYMUPDF4LLM_AVAILABLE or pymupdf4llm is None:
        raise RuntimeError("pymupdf4llm is unavailable")

    to_markdown = getattr(pymupdf4llm, "to_markdown", None)
    if to_markdown is None:
        raise RuntimeError("pymupdf4llm.to_markdown is unavailable")

    kwargs: Dict[str, Any] = {}
    try:
        parameters = inspect.signature(to_markdown).parameters
    except Exception:
        parameters = {}

    if "page_chunks" in parameters:
        kwargs["page_chunks"] = True
    if "write_images" in parameters:
        kwargs["write_images"] = write_images
    if image_output_dir:
        for candidate in (
            "image_path",
            "image_dir",
            "image_folder",
            "image_output_dir",
        ):
            if candidate in parameters:
                kwargs[candidate] = image_output_dir
                break

    if start_page is not None or end_page is not None:
        try:
            doc = fitz.open(file_path)
            total_pages = len(doc)
            doc.close()
            sp = max(1, start_page) if start_page is not None else 1
            ep = min(total_pages, end_page) if end_page is not None else total_pages
            kwargs["pages"] = list(range(sp - 1, ep))
        except Exception as e:
            logger.warning("Failed to determine total pages for pymupdf4llm filtering: %s", e)

    try:
        return to_markdown(file_path, **kwargs)
    except TypeError:
        if image_output_dir:
            kwargs.setdefault("image_path", image_output_dir)
        kwargs.setdefault("page_chunks", True)
        kwargs.setdefault("write_images", write_images)
        return to_markdown(file_path, **kwargs)


def _normalise_markdown_pages(markdown_result: Any) -> List[Dict[str, Any]]:
    if isinstance(markdown_result, str):
        text = markdown_result.strip()
        return [{"page_number": 1, "text": text}] if text else []

    if not isinstance(markdown_result, list):
        return []

    pages: List[Dict[str, Any]] = []
    for index, item in enumerate(markdown_result, start=1):
        if isinstance(item, dict):
            page_number = item.get("page") or item.get("page_number") or index
            text = (
                item.get("text")
                or item.get("md")
                or item.get("markdown")
                or item.get("content")
                or ""
            )
        else:
            page_number = index
            text = str(item or "")

        text = text.strip()
        if not text:
            continue
        pages.append({"page_number": int(page_number), "text": text})

    return pages


def _extract_pdf_text_pages_with_fitz(
    file_path: str,
    start_page: Optional[int] = None,
    end_page: Optional[int] = None,
) -> List[Dict[str, Any]]:
    pages: List[Dict[str, Any]] = []
    pdf_document = fitz.open(file_path)
    try:
        for page_index in range(len(pdf_document)):
            current_page = page_index + 1
            if start_page is not None and current_page < start_page:
                continue
            if end_page is not None and current_page > end_page:
                continue
            
            page = pdf_document.load_page(page_index)
            text = (page.get_text("text") or "").strip()
            if not text:
                continue
            pages.append({"page_number": page_index + 1, "text": text})
    finally:
        pdf_document.close()
    return pages


def convert_pdf_to_markdown(
    file_path: str,
    *,
    write_images: bool = False,
    image_output_dir: Optional[str] = None,
    start_page: Optional[int] = None,
    end_page: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Converts an entire PDF to markdown in one pass when pymupdf4llm is
    available, with a safe PyMuPDF text fallback when it is not.
    """
    if not os.path.exists(file_path):
        logger.error("convert_pdf_to_markdown: file not found: %s", file_path)
        return {
            "text": "",
            "pages": [],
            "conversion_method": "missing_file",
            "images_output_dir": image_output_dir,
        }

    if PYMUPDF4LLM_AVAILABLE:
        try:
            logger.info(
                "convert_pdf_to_markdown: using pymupdf4llm for %s "
                "(write_images=%s, image_output_dir=%s).",
                file_path,
                write_images,
                image_output_dir,
            )
            markdown_result = _call_pymupdf4llm_to_markdown(
                file_path,
                write_images=write_images,
                image_output_dir=image_output_dir,
                start_page=start_page,
                end_page=end_page,
            )
            pages = _normalise_markdown_pages(markdown_result)
            text = "\n\n".join(page["text"] for page in pages)
            return {
                "text": text,
                "pages": pages,
                "conversion_method": "pymupdf4llm_markdown",
                "images_output_dir": image_output_dir,
            }
        except Exception as exc:
            logger.warning(
                "convert_pdf_to_markdown: pymupdf4llm conversion failed for %s: %s. "
                "Falling back to PyMuPDF plain-text extraction.",
                file_path,
                exc,
            )

    try:
        pages = _extract_pdf_text_pages_with_fitz(file_path, start_page=start_page, end_page=end_page)
        text = "\n\n".join(page["text"] for page in pages)
        return {
            "text": text,
            "pages": pages,
            "conversion_method": "fitz_text_fallback",
            "images_output_dir": image_output_dir,
        }
    except Exception as exc:
        logger.warning(
            "convert_pdf_to_markdown: PyMuPDF fallback failed for %s: %s",
            file_path,
            exc,
        )
        return {
            "text": "",
            "pages": [],
            "conversion_method": "failed",
            "images_output_dir": image_output_dir,
        }

def extract_content_from_pdf(
    file_path: str,
    start_page: Optional[int] = None,
    end_page: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Markdown-first PDF extraction path for standards ingestion.
    Returns the existing response shape while sourcing text from the
    centralized markdown conversion helper.
    """
    conversion = convert_pdf_to_markdown(file_path, start_page=start_page, end_page=end_page)
    logger.info(
        "extract_content_from_pdf: conversion_method=%s file=%s pages=%d text_len=%d",
        conversion.get("conversion_method"),
        file_path,
        len(conversion.get("pages") or []),
        len(conversion.get("text") or ""),
    )
    return {
        "text": conversion.get("text", ""),
        "images": [],
        "pages": conversion.get("pages", []),
        "conversion_method": conversion.get("conversion_method"),
    }

def get_document_content(
    file_path: str,
    document_name: str,
    start_page: Optional[int] = None,
    end_page: Optional[int] = None,
) -> Dict[str, Any]:
    """Extract content from a design document (PDF) by extension."""
    return extract_content_from_pdf(file_path, start_page=start_page, end_page=end_page)
