import numpy as np
from .vector_store import load_index
from .embeddings import generate_embeddings

def get_full_resume_text():
    index, metadata = load_index()
    if metadata:
        return "\n".join(metadata)
    return ""

def extract_section(full_text, section_name):
    text = full_text

    upper_text = text.upper()

    if section_name.upper() not in upper_text:
        return "Section not found."

    # Find start index of section
    start_index = upper_text.index(section_name.upper()) + len(section_name)

    # Cut from section start onward
    section_text = text[start_index:]

    # Stop at next all-caps heading (simple heuristic)
    lines = section_text.split("\n")
    cleaned_lines = []

    for line in lines:
        line_strip = line.strip()

        # Stop if another big heading appears
        if line_strip.isupper() and len(line_strip) > 5:
            break

        cleaned_lines.append(line_strip)

    return "\n".join(cleaned_lines).strip()

def rag_chat(question):
    q = question.lower()

    full_text = get_full_resume_text()

    if not full_text:
        return {"answer": "No documents uploaded yet."}

    if "education" in q:
        return {"answer": extract_section(full_text, "EDUCATION")}

    if "projects" in q:
        return {"answer": extract_section(full_text, "PROJECTS")}

    if "skill" in q:
        return {"answer": extract_section(full_text, "TECHNICAL SKILLS")}

    if "summary" in q:
        return {"answer": extract_section(full_text, "PROFESSIONAL SUMMARY")}

    # Fallback to semantic search for other questions
    index, metadata = load_index()
    query_embedding = generate_embeddings([question])
    query_embedding = np.array(query_embedding).astype("float32")

    D, I = index.search(query_embedding, k=3)

    results = [metadata[i] for i in I[0]]
    return {"answer": "\n".join(results)}
