import fitz  # PyMuPDF
from typing import Tuple, List

def parse_pdf(file_bytes: bytes) -> Tuple[str, List[bytes]]:
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    text_content = ""
    images = []
    
    for page in doc:
        text_content += page.get_text("text") + "\n"
        for img in page.get_images(full=True):
            xref = img[0]
            base_image = doc.extract_image(xref)
            images.append(base_image["image"])
            
    return text_content, images
