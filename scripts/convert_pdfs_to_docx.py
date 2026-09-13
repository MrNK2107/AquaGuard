"""
AquaGuard PDF to Word (DOCX) Converter
======================================
Converts all PDF files in memo/pdf to editable Word documents in memo/word.
"""

from pathlib import Path
from pdf2docx import Converter

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
MEMO_DIR = WORKSPACE_ROOT / "memo"
PDF_DIR = MEMO_DIR / "pdf"
WORD_DIR = MEMO_DIR / "word"

WORD_DIR.mkdir(parents=True, exist_ok=True)

def convert_all():
    pdf_files = list(PDF_DIR.glob("*.pdf"))
    if not pdf_files:
        print("No PDF files found in memo/pdf")
        return
    
    print(f"Found {len(pdf_files)} PDF files to convert to Word (.docx)...")
    for pdf_path in pdf_files:
        docx_name = pdf_path.stem + ".docx"
        docx_path = WORD_DIR / docx_name
        print(f"Converting: {pdf_path.name} -> {docx_name}")
        
        cv = Converter(str(pdf_path))
        cv.convert(str(docx_path), start=0, end=None)
        cv.close()
        print(f"  [OK] Saved to: {docx_path}")

    print("\n[SUCCESS] All PDFs converted to Word (.docx) in memo/word/")

if __name__ == "__main__":
    convert_all()
