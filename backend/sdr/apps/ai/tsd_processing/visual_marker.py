import io
import logging
from typing import Dict, Any

from PIL import Image, ImageChops, ImageDraw, ImageFont

logger = logging.getLogger(__name__)

try:
    import pytesseract
    HAS_TESSERACT = True
except ImportError:
    HAS_TESSERACT = False

def apply_visual_markers(image_bytes: bytes) -> bytes:
    """
    Applies numbered markers (Set-of-Mark) to the image for Vision LLM grounding.
    Uses OCR (pytesseract) to find text regions and draws a labeled box next to them.
    If pytesseract is missing or fails, returns the original image_bytes.
    """
    if not HAS_TESSERACT:
        logger.warning("pytesseract not installed. Returning unmarked diagram.")
        return image_bytes

    try:
        image = Image.open(io.BytesIO(image_bytes))
        # Ensure image is in RGB mode for drawing colored boxes
        if image.mode != "RGB":
            image = image.convert("RGB")
            
        try:
            data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
        except pytesseract.TesseractNotFoundError:
            logger.warning("Tesseract binary not found. Returning unmarked diagram.")
            return image_bytes
            
        draw = ImageDraw.Draw(image)
        font = None
        for font_name in ["DejaVuSans-Bold.ttf", "Arial.ttf", "arial.ttf", "LiberationSans-Regular.ttf"]:
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

        # Group by block_num to avoid labeling every single word
        blocks: Dict[int, Dict[str, Any]] = {}
        
        n_boxes = len(data['level'])
        for i in range(n_boxes):
            # Only consider words with reasonable confidence
            conf = int(data['conf'][i])
            text = data['text'][i].strip()
            if conf < 30 or not text:
                continue
                
            block_num = data['block_num'][i]
            x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
            
            if block_num not in blocks:
                blocks[block_num] = {"left": x, "top": y, "right": x + w, "bottom": y + h, "text": text}
            else:
                blocks[block_num]["left"] = min(blocks[block_num]["left"], x)
                blocks[block_num]["top"] = min(blocks[block_num]["top"], y)
                blocks[block_num]["right"] = max(blocks[block_num]["right"], x + w)
                blocks[block_num]["bottom"] = max(blocks[block_num]["bottom"], y + h)
                blocks[block_num]["text"] += " " + text

        # Draw markers for each block
        marker_id = 1
        # Track marked bboxes so shape detection can skip overlapping regions
        marked_rects: list = []

        for block_num, bbox in blocks.items():
            left = bbox["left"]
            top = bbox["top"]
            right = bbox["right"]
            bottom = bbox["bottom"]

            label = f"[{marker_id}]"
            text_bbox = draw.textbbox((0, 0), label, font=font)
            text_w = text_bbox[2] - text_bbox[0]
            text_h = text_bbox[3] - text_bbox[1]

            margin = 4
            rect_left = max(0, left - text_w - margin * 2 - 2)
            rect_top = max(0, top - margin)
            rect_right = rect_left + text_w + margin * 2
            rect_bottom = rect_top + text_h + margin * 2

            if rect_left == 0:
                rect_left = left
                rect_right = rect_left + text_w + margin * 2
                rect_top = max(0, top - text_h - margin * 2 - 2)
                rect_bottom = rect_top + text_h + margin * 2

            draw.rectangle([rect_left, rect_top, rect_right, rect_bottom], fill="red")
            draw.text((rect_left + margin, rect_top + margin), label, fill="white", font=font)
            marked_rects.append((left, top, right, bottom))
            marker_id += 1

        # Second pass: mark non-text visual elements (shapes, boxes, icons) using
        # region-based segmentation without OpenCV.
        try:
            marker_id = _mark_visual_shapes(image, draw, font, marker_id, marked_rects)
        except Exception:
            logger.debug("Shape detection pass failed — skipping.", exc_info=True)

        # Save back to bytes
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')
        return img_byte_arr.getvalue()
        
    except Exception as e:
        logger.exception(f"Failed to apply visual markers: {e}")
        return image_bytes


def _mark_visual_shapes(
    image: "Image.Image",
    draw: "ImageDraw.ImageDraw",
    font: Any,
    start_marker_id: int,
    existing_rects: list,
    *,
    min_area: int = 800,
    max_area_fraction: float = 0.20,
    iou_overlap_threshold: float = 0.30,
) -> int:
    """
    Detect non-text visual elements (boxes, icons, shapes) using PIL only and
    draw numbered markers on them. Returns the next available marker_id.

    Strategy: quantize colors to a small palette, find each color region's bounding
    box, filter by size, skip regions that heavily overlap already-marked text blocks.
    """
    img_w, img_h = image.size
    total_pixels = img_w * img_h
    max_area = int(total_pixels * max_area_fraction)

    # Reduce colors to detect distinct filled regions without external libraries
    quantized = image.quantize(colors=16, method=2).convert("RGB")
    # Work pixel-by-pixel via getdata — collect unique colors
    pixels = list(quantized.getdata())
    unique_colors: set = set()
    for px in pixels:
        unique_colors.add(px)

    marker_id = start_marker_id
    seen_bboxes: list = list(existing_rects)

    # Split channels once; reused per color iteration
    r_ch, g_ch, b_ch = quantized.split()

    for color in unique_colors:
        # Build per-channel binary masks and combine: pixel is in region iff all 3 match
        r_mask = r_ch.point(lambda v, c=color[0]: 255 if v == c else 0)
        g_mask = g_ch.point(lambda v, c=color[1]: 255 if v == c else 0)
        b_mask = b_ch.point(lambda v, c=color[2]: 255 if v == c else 0)
        mask = ImageChops.multiply(ImageChops.multiply(r_mask, g_mask), b_mask)

        bbox = mask.getbbox()
        if bbox is None:
            continue

        bx0, by0, bx1, by1 = bbox
        area = (bx1 - bx0) * (by1 - by0)

        if area < min_area or area > max_area:
            continue

        # Skip near-white and near-black background regions
        avg_r, avg_g, avg_b = color
        brightness = (avg_r + avg_g + avg_b) / 3
        if brightness > 230 or brightness < 15:
            continue

        # Skip if this bbox heavily overlaps any already-marked region
        if _has_significant_overlap(bx0, by0, bx1, by1, seen_bboxes, iou_overlap_threshold):
            continue

        seen_bboxes.append((bx0, by0, bx1, by1))

        label = f"[{marker_id}]"
        text_bbox_dims = draw.textbbox((0, 0), label, font=font)
        text_w = text_bbox_dims[2] - text_bbox_dims[0]
        text_h = text_bbox_dims[3] - text_bbox_dims[1]
        margin = 3

        # Place marker at top-left corner of the shape bounding box
        rx0 = max(0, bx0)
        ry0 = max(0, by0 - text_h - margin * 2)
        rx1 = rx0 + text_w + margin * 2
        ry1 = ry0 + text_h + margin * 2

        # Draw a blue marker (distinct from red text markers)
        draw.rectangle([rx0, ry0, rx1, ry1], fill=(0, 80, 200))
        draw.text((rx0 + margin, ry0 + margin), label, fill="white", font=font)
        marker_id += 1

    return marker_id


def _has_significant_overlap(
    x0: int, y0: int, x1: int, y1: int,
    existing: list,
    threshold: float,
) -> bool:
    """Returns True if the box overlaps any existing box by more than threshold (IoU-style)."""
    area = max(0, x1 - x0) * max(0, y1 - y0)
    if area == 0:
        return True
    for ex0, ey0, ex1, ey1 in existing:
        inter_x0 = max(x0, ex0)
        inter_y0 = max(y0, ey0)
        inter_x1 = min(x1, ex1)
        inter_y1 = min(y1, ey1)
        inter_area = max(0, inter_x1 - inter_x0) * max(0, inter_y1 - inter_y0)
        if inter_area == 0:
            continue
        # Overlap fraction relative to the new box
        if inter_area / area >= threshold:
            return True
    return False
