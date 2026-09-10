# Module 3 — Document Loading & Chunking

## Objective

Learn how external documents are prepared before they can be embedded and used in a RAG system.

The module covers:

- PDF text extraction
- OCR detection
- Markdown extraction
- LangChain Documents
- Chunking
- Chunk overlap
- Metadata
- Complete ingestion preparation

---

## Pipeline

The final ingestion pipeline developed in this module is:

```text
PDF
 ↓
Check Native Text
 ↓
PyMuPDF4LLM
 ↓
Markdown
 ↓
LangChain Document
 ↓
Metadata
 ↓
Chunking
 ↓
Chunks
```

If usable native text is missing:

```text
PDF Page
 ↓
OCR Required
 ↓
OCR Processing
 ↓
Extracted Text
```

OCR is treated as a fallback rather than being applied unnecessarily.

---

## Why Extraction Quality Matters

A PDF is a visual document format.

Extracting text from it can introduce problems involving:

```text
Multi-column layouts
Tables
Headings
Figures
Equations
Scanned pages
Reading order
```

Poor extraction eventually affects the complete RAG pipeline:

```text
Poor Extraction
      ↓
Poor Chunks
      ↓
Poor Embeddings
      ↓
Poor Retrieval
      ↓
Poor Answers
```

---

## PyPDFLoader vs PyMuPDF4LLM

Both approaches were compared using the Cardiac Arrest research paper.

`PyPDFLoader` produced usable plain text, but complex structures such as tables were flattened.

PyMuPDF4LLM produced Markdown that preserved more document structure.

Therefore the final primary extraction pipeline uses:

```text
PDF
 ↓
PyMuPDF4LLM
 ↓
Markdown
```

`PyPDFLoader` remains useful for learning and comparison.

---

## OCR Detection

Before applying OCR, pages are inspected for usable native text.

Conceptually:

```text
Page
 ↓
Extract Native Text
 ↓
Enough Text?
 ├── Yes → Use native extraction
 └── No  → OCR candidate
```

The Cardiac Arrest research paper contained usable native text, so OCR was not required.

---

## LangChain Document

Extracted Markdown is converted into a LangChain `Document`.

Example:

```python
Document(
    page_content=markdown_text,
    metadata={
        "source": "Cardiac Arrest.pdf",
        "extraction_method": "pymupdf4llm",
        "format": "markdown",
        "domain": "healthcare",
        "document_type": "research_paper",
        "topic": "cardiac_arrest_prediction"
    }
)
```

A `Document` contains:

```text
page_content
+
metadata
```

---

## Chunking

Large documents should not normally be embedded as one giant piece of text.

Instead:

```text
Document
 ↓
Text Splitter
 ↓
Chunk 1
Chunk 2
Chunk 3
...
```

This project uses:

```python
RecursiveCharacterTextSplitter
```

---

## RecursiveCharacterTextSplitter

Example:

```python
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len
)
```

With `length_function=len`, the chunk size represents characters rather than tokens.

The splitter attempts to split text using natural boundaries before falling back to smaller separators.

---

## Chunk Size

Chunk size controls approximately how much content is stored in each chunk.

```text
Small Chunks
 ↓
Higher retrieval precision
but
Less surrounding context
```

```text
Large Chunks
 ↓
More context
but
Potentially more irrelevant information
```

There is no universally correct chunk size.

It should be tuned according to the documents and questions.

---

## Chunk Overlap

Overlap preserves information near chunk boundaries.

Example:

```text
Chunk 1
AAAA BBBB CCCC

Chunk 2
     CCCC DDDD EEEE
```

The repeated section provides continuity.

Too little overlap may lose relationships across boundaries.

Too much overlap creates redundant information and embeddings.

---

## Metadata

Metadata provides information about where a chunk came from.

Example:

```python
{
    "source": "Cardiac Arrest.pdf",
    "domain": "healthcare",
    "document_type": "research_paper",
    "topic": "cardiac_arrest_prediction",
    "chunk_id": 42
}
```

Metadata later enables:

```text
Filtering
Source tracking
Debugging
Citations
Retrieval analysis
```

---

## Chunk IDs

Each chunk receives an identifier:

```python
for index, chunk in enumerate(chunks):
    chunk.metadata["chunk_id"] = index
```

This makes retrieved chunks easier to inspect and trace.

---

## Module Files

```text
03_document_loading_chunking/
│
├── documents/
│   └── Cardiac Arrest.pdf
│
├── outputs/
│   └── Cardiac_Arrest.md
│
├── 00_pdf_extraction_ocr.py
├── 01_document_loader.py
├── 02_basic_chunking.py
├── 03_chunk_overlap.py
├── 04_metadata.py
├── 05_complete_ingestion.py
└── README.md
```

---

## Key Takeaway

Document preparation is not simply:

```text
PDF → Text
```

A stronger ingestion pipeline considers:

```text
Extraction Quality
      ↓
Document Structure
      ↓
Metadata
      ↓
Chunking Strategy
      ↓
High-Quality Chunks
```

These chunks become the input to the embedding pipeline in Module 4.