import fitz
import os

def extract_pdf_to_images(pdf_path, output_dir, prefix):
    os.makedirs(output_dir, exist_ok=True)
    doc = fitz.open(pdf_path)
    for i, page in enumerate(doc):
        pix = page.get_pixmap(dpi=150)
        output_path = os.path.join(output_dir, f"{prefix}_page_{i}.png")
        pix.save(output_path)
        print(f"Saved {output_path}")

extract_pdf_to_images("ok images.pdf", "images", "ok")
extract_pdf_to_images("NG part.pdf", "images", "ng")
