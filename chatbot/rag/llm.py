from transformers import pipeline

generator = None

def load_model():
    global generator
    if generator is None:
        print("Loading FLAN-T5 model...")
        generator = pipeline("text2text-generation", model="google/flan-t5-base")
        print("Model loaded successfully!")
    return generator


def generate_answer(context, question):
    model = load_model()

    if not context.strip():
        return "No relevant information found."

    prompt = f"""
Answer the question using only the provided context.
If the answer is not in the context, say: Not mentioned in the document.

Context:
{context}

Question:
{question}

Answer:
"""

    result = model(
        prompt,
        max_new_tokens=100,
        do_sample=False
    )

    return result[0]["generated_text"].strip()
