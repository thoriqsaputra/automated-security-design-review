import io
import logging
from typing import Any, Dict, List

from PIL import Image, ImageChops, ImageDraw, ImageFont

logger = logging.getLogger(__name__)

try:
    import pytesseract
    HAS_TESSERACT = True
except ImportError:
    HAS_TESSERACT = False


def _merge_nearby_blocks(blocks: Dict[int, Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Merge OCR blocks that are in the same horizontal band and close horizontally.

    Tesseract often assigns separate block_num to words in the same visual box
    (e.g. "DMZ" and "Boundary" become two blocks). This pass merges blocks whose
    top edges are within 30 px of each other AND whose horizontal extent is within
    30 px, so multi-word labels get a single marker.
    """
    items = sorted(blocks.values(), key=lambda b: (b["top"], b["left"]))
    merged: List[Dict[str, Any]] = []
    for item in items:
        for m in merged:
            y_close = abs(item["top"] - m["top"]) <= 30
            x_adjacent = item["left"] <= m["right"] + 30
            if y_close and x_adjacent:
                m["left"] = min(m["left"], item["left"])
                m["top"] = min(m["top"], item["top"])
                m["right"] = max(m["right"], item["right"])
                m["bottom"] = max(m["bottom"], item["bottom"])
                m["text"] += " " + item["text"]
                break
        else:
            merged.append(dict(item))
    return merged


def apply_visual_markers(image_bytes: bytes) -> bytes:
    """
    Applies numbered markers (Set-of-Mark) to the image for Vision LLM grounding.

    Each distinct text region detected by OCR gets a single red [N] label placed
    ABOVE the region so it does not obscure the component name. Multi-word labels
    in the same visual area are merged into one marker. Shape detection adds blue
    [N] markers only for non-text elements not already covered by OCR markers.
    """
    if not HAS_TESSERACT:
        logger.warning("pytesseract not installed. Returning unmarked diagram.")
        return image_bytes

    try:
        image = Image.open(io.BytesIO(image_bytes))
        if image.mode != "RGB":
            image = image.convert("RGB")

        try:
            data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
        except pytesseract.TesseractNotFoundError:
            logger.warning("Tesseract binary not found. Returning unmarked diagram.")
            return image_bytes

        draw = ImageDraw.Draw(image)
        font = None
        for font_name in [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "DejaVuSans-Bold.ttf",
            "Arial.ttf",
            "arial.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        ]:
            try:
                font = ImageFont.truetype(font_name, 20)
                break
            except (IOError, OSError):
                continue
        if font is None:
            try:
                font = ImageFont.load_default(size=16)
            except TypeError:
                font = ImageFont.load_default()

        # Collect OCR blocks grouped by block_num
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

        # Fix 2: merge spatially adjacent blocks (same box, multi-word labels)
        merged_blocks = _merge_nearby_blocks(blocks)

        marker_id = 1
        # Track text bboxes for shape detection containment check
        marked_rects: List[tuple] = []

        for bbox in merged_blocks:
            left = bbox["left"]
            top = bbox["top"]
            right = bbox["right"]
            bottom = bbox["bottom"]

            label = f"[{marker_id}]"
            lb = draw.textbbox((0, 0), label, font=font)
            text_w = lb[2] - lb[0]
            text_h = lb[3] - lb[1]
            margin = 4

            # Fix 1: place marker ABOVE the text block, left-aligned with it.
            # This keeps the component label fully readable.
            rect_left = left
            rect_top = top - text_h - margin * 2 - 3
            rect_right = rect_left + text_w + margin * 2
            rect_bottom = rect_top + text_h + margin * 2

            # Fallback: if no room above (near top edge), place below the block
            if rect_top < 0:
                rect_top = bottom + 3
                rect_bottom = rect_top + text_h + margin * 2

            draw.rectangle([rect_left, rect_top, rect_right, rect_bottom], fill="red")
            draw.text((rect_left + margin, rect_top + margin), label, fill="white", font=font)
            marked_rects.append((left, top, right, bottom))
            marker_id += 1

        # Second pass: mark non-text visual elements not already covered by OCR markers
        try:
            marker_id = _mark_visual_shapes(image, draw, font, marker_id, marked_rects)
        except Exception:
            logger.debug("Shape detection pass failed — skipping.", exc_info=True)

        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format="PNG")
        return img_byte_arr.getvalue()

    except Exception as e:
        logger.exception(f"Failed to apply visual markers: {e}")
        return image_bytes


def _contains_any_marked_center(
    bx0: int, by0: int, bx1: int, by1: int, existing_rects: List[tuple]
) -> bool:
    """True if the center point of any already-marked text bbox falls inside this shape bbox.

    This is more reliable than IoU for detecting when a shape *contains* a text label,
    because the text bbox is typically much smaller than the enclosing box shape.
    """
    for ex0, ey0, ex1, ey1 in existing_rects:
        cx = (ex0 + ex1) / 2
        cy = (ey0 + ey1) / 2
        if bx0 <= cx <= bx1 and by0 <= cy <= by1:
            return True
    return False


def _mark_visual_shapes(
    image: "Image.Image",
    draw: "ImageDraw.ImageDraw",
    font: Any,
    start_marker_id: int,
    existing_rects: List[tuple],
    *,
    min_area: int = 1500,
    max_area_fraction: float = 0.20,
) -> int:
    """
    Detect non-text visual elements (boxes, icons) using PIL color quantization and
    draw numbered blue markers on them. Returns the next available marker_id.

    Skips any shape whose bounding box contains the center of an already-marked text
    region (Fix 3). Filters very elongated shapes like arrows (Fix 4).
    """
    img_w, img_h = image.size
    total_pixels = img_w * img_h
    max_area = int(total_pixels * max_area_fraction)

    quantized = image.quantize(colors=16, method=2).convert("RGB")
    pixels = list(quantized.getdata())
    unique_colors: set = set()
    for px in pixels:
        unique_colors.add(px)

    marker_id = start_marker_id
    seen_bboxes: List[tuple] = list(existing_rects)

    r_ch, g_ch, b_ch = quantized.split()

    for color in unique_colors:
        r_mask = r_ch.point(lambda v, c=color[0]: 255 if v == c else 0)
        g_mask = g_ch.point(lambda v, c=color[1]: 255 if v == c else 0)
        b_mask = b_ch.point(lambda v, c=color[2]: 255 if v == c else 0)
        mask = ImageChops.multiply(ImageChops.multiply(r_mask, g_mask), b_mask)

        bbox = mask.getbbox()
        if bbox is None:
            continue

        bx0, by0, bx1, by1 = bbox
        width = bx1 - bx0
        height = by1 - by0
        area = width * height

        # Fix 4: filter by area and aspect ratio to skip arrowheads and thin lines
        if area < min_area or area > max_area:
            continue
        aspect = max(width, height) / max(min(width, height), 1)
        if aspect > 5:
            continue

        # Skip near-white and near-black regions (background, text strokes)
        avg_r, avg_g, avg_b = color
        brightness = (avg_r + avg_g + avg_b) / 3
        if brightness > 230 or brightness < 15:
            continue

        # Fix 3: skip shapes that contain an already-marked text block center
        if _contains_any_marked_center(bx0, by0, bx1, by1, existing_rects):
            continue

        # Also skip if this shape heavily overlaps a previously seen shape bbox
        already_seen = False
        for sb0, sb1, sb2, sb3 in seen_bboxes:
            inter_w = max(0, min(bx1, sb2) - max(bx0, sb0))
            inter_h = max(0, min(by1, sb3) - max(by0, sb1))
            if inter_w * inter_h > 0.5 * area:
                already_seen = True
                break
        if already_seen:
            continue

        seen_bboxes.append((bx0, by0, bx1, by1))

        label = f"[{marker_id}]"
        lb = draw.textbbox((0, 0), label, font=font)
        text_w = lb[2] - lb[0]
        text_h = lb[3] - lb[1]
        margin = 3

        rx0 = max(0, bx0)
        ry0 = max(0, by0 - text_h - margin * 2)
        rx1 = rx0 + text_w + margin * 2
        ry1 = ry0 + text_h + margin * 2

        draw.rectangle([rx0, ry0, rx1, ry1], fill=(0, 80, 200))
        draw.text((rx0 + margin, ry0 + margin), label, fill="white", font=font)
        marker_id += 1

    return marker_id
