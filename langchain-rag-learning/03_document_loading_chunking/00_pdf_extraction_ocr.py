from pathlib import Path
import fitz
import pymupdf4llm 
from langchain_community.document_loaders import PyPDFLoader


OUTPUT_DIR = Path("03_document_loading_chunking/outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

#save the markdown 
MARKDOWN_OUTPUT = OUTPUT_DIR / "Cardiac_Arrest.md"


# 1. CHECK WHETHER OCR IS NEEDED

print("\n" + "=" * 70)
print("STEP 1: CHECKING PDF FOR NATIVE TEXT / OCR NEED")
print("=" * 70)

doc = fitz.open("03_document_loading_chunking/documents/Cardiac Arrest.pdf")
ocr_pages = []
for page_number, page in enumerate(doc):

    text = page.get_text().strip()

    char_count = len(text)

    if char_count < 50:

        print(
            f"Page {page_number + 1}: "
            f"OCR likely needed "
            f"({char_count} characters)"
        )

        ocr_pages.append(
            page_number + 1
        )

    else:

        print(
            f"Page {page_number + 1}: "
            f"Native text available "
            f"({char_count} characters)"
        )
doc.close()

print("\n" + "-" * 70)

if ocr_pages:

    print(
        "OCR candidate pages:",
        ocr_pages
    )

else:

    print(
        "No OCR required. "
        "All pages contain usable native text."
    )


# 3. PyMuPDF4LLM MARKDOWN EXTRACTION

print("\n" + "=" * 70)
print("STEP 3: EXTRACTING PDF TO MARKDOWN")
print("=" * 70)

# extract pdf as Markdown 
markdown = pymupdf4llm.to_markdown(
    "03_document_loading_chunking/documents/Cardiac Arrest.pdf"
)

MARKDOWN_OUTPUT.write_text(
    markdown,
    encoding="utf-8"
)
print(
    f"Markdown extraction saved to:\n"
    f"{MARKDOWN_OUTPUT}"
)

print(
    f"Total characters extracted "
    f"with PyMuPDF4LLM: {len(markdown)}"
)

# 4. QUALITY PREVIEW
# ============================================================

print("\n" + "=" * 70)
print("STEP 3: EXTRACTION PREVIEW")
print("=" * 70)


print(
    markdown[:3000]
)
