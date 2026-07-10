import io
import logging
from typing import Any, Dict, List

from PIL import Image

logger = logging.getLogger(__name__)

try:
    import pytesseract

    HAS_TESSERACT = True
except ImportError:
    HAS_TESSERACT = False


def _merge_nearby_blocks(blocks: Dict[int, Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Merge OCR blocks that are in the same horizontal band and close horizontally."""
    items = sorted(blocks.values(), key=lambda b: (b["top"], b["left"]))
    merged: List[Dict[str, Any]] = []
    for item in items:
        for merged_item in merged:
            y_close = abs(item["top"] - merged_item["top"]) <= 30
            x_adjacent = item["left"] <= merged_item["right"] + 30
            if y_close and x_adjacent:
                merged_item["left"] = min(merged_item["left"], item["left"])
                merged_item["top"] = min(merged_item["top"], item["top"])
                merged_item["right"] = max(merged_item["right"], item["right"])
                merged_item["bottom"] = max(merged_item["bottom"], item["bottom"])
                merged_item["text"] += " " + item["text"]
                break
        else:
            merged.append(dict(item))
    return merged


def _collect_ocr_blocks(image: "Image.Image") -> List[Dict[str, Any]]:
    data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)

    blocks: Dict[int, Dict[str, Any]] = {}
    n_boxes = len(data["level"])
    for i in range(n_boxes):
        conf = int(data["conf"][i])
        text = data["text"][i].strip()
        if conf < 30 or not text:
            continue
        block_num = data["block_num"][i]
        x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
        if block_num not in blocks:
            blocks[block_num] = {"left": x, "top": y, "right": x + w, "bottom": y + h, "text": text}
        else:
            blocks[block_num]["left"] = min(blocks[block_num]["left"], x)
            blocks[block_num]["top"] = min(blocks[block_num]["top"], y)
            blocks[block_num]["right"] = max(blocks[block_num]["right"], x + w)
            blocks[block_num]["bottom"] = max(blocks[block_num]["bottom"], y + h)
            blocks[block_num]["text"] += " " + text

    return _merge_nearby_blocks(blocks)


def extract_diagram_text(image_bytes: bytes) -> str:
    """OCR the diagram and return visible text labels, space-joined."""
    if not HAS_TESSERACT:
        return ""
    try:
        image = Image.open(io.BytesIO(image_bytes))
        if image.mode != "RGB":
            image = image.convert("RGB")
        merged_blocks = _collect_ocr_blocks(image)
        return " ".join(block["text"] for block in merged_blocks).strip()
    except pytesseract.TesseractNotFoundError:
        return ""
    except Exception:
        logger.debug("extract_diagram_text: OCR failed; returning empty string.", exc_info=True)
        return ""
