from pypdf import PdfReader
from .embeddings import generate_embeddings
from .vector_store import load_index, save_index
import numpy as np
import faiss


def extract_text(file_path):
    try:
        reader = PdfReader(file_path)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        return text
    except:
        return ""


def chunk_text(text):
    # Paragraph-based chunking
    chunks = text.split("\n\n")

    cleaned_chunks = []
    for chunk in chunks:
        chunk = chunk.strip()
        if len(chunk) > 50:
            cleaned_chunks.append(chunk)

    return cleaned_chunks


def process_pdf(file_path):
    text = extract_text(file_path)

    if not text.strip():
        return

    chunks = chunk_text(text)

    embeddings = generate_embeddings(chunks)
    embeddings = np.array(embeddings).astype("float32")

    index, metadata = load_index()

    if index is None:
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)

    index.add(embeddings)
    metadata.extend(chunks)

    save_index(index, metadata)
def extract_sections(text):

    sections = {
        "skills": "",
        "education": "",
        "projects": "",
        "experience": ""
    }

    text_lower = text.lower()

    # Simple keyword based splitting
    if "skills" in text_lower:
        parts = text_lower.split("skills")
        if len(parts) > 1:
            sections["skills"] = parts[1][:500]

    if "education" in text_lower:
        parts = text_lower.split("education")
        if len(parts) > 1:
            sections["education"] = parts[1][:500]

    if "projects" in text_lower:
        parts = text_lower.split("projects")
        if len(parts) > 1:
            sections["projects"] = parts[1][:500]

    if "experience" in text_lower:
        parts = text_lower.split("experience")
        if len(parts) > 1:
            sections["experience"] = parts[1][:500]

    return sections
