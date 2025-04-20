import modal
import pdfplumber  # Better PDF extraction
from sentence_transformers import SentenceTransformer
import numpy as np
import io
import re
from typing import List, Dict

# Define image with improved dependencies
image = modal.Image.debian_slim().pip_install(
    "numpy==1.26.4",
    "pdfplumber",  # Replaced pypdf
    "sentence-transformers",
    "huggingface_hub[hf_xet]",  # For better model downloads
    "langchain"  # Keeping for advanced splitting if needed
)

app = modal.App("rag-pdf-processor", image=image)
vol = modal.Volume.from_name("pdf-storage", create_if_missing=True)

# Better embedding model
embedding_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

def clean_text(text: str) -> str:
    """Clean extracted PDF text"""
    text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
    text = re.sub(r'[^\w\s.-]', ' ', text)  # Remove special chars
    return text.strip()

def split_text(text: str) -> List[str]:
    """Improved text splitting with paragraph awareness"""
    # First split by paragraphs
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    
    chunks = []
    for para in paragraphs:
        # If paragraph is too long, split further
        if len(para) > 500:
            words = para.split()
            for i in range(0, len(words), 300):
                chunks.append(' '.join(words[i:i+300]))
        else:
            chunks.append(para)
    return chunks

@app.function(
    volumes={"/data": vol},
    timeout=600,
    image=image
)
def process_pdf(pdf_bytes: bytes) -> Dict:
    """Enhanced PDF processing with better text extraction"""
    text = ""
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += clean_text(page_text) + "\n\n"
    
    chunks = split_text(text)
    
    # Generate embeddings with better model
    embeddings = embedding_model.encode(chunks)
    
    # Store with metadata
    with open("/data/chunks.txt", "w") as f:
        f.write("\n".join(chunks))
    np.save("/data/embeddings.npy", embeddings)
    
    return {"status": "success", "num_chunks": len(chunks)}

@app.function(
    volumes={"/data": vol},
    image=image
)
def query(question: str) -> Dict:
    """Enhanced query with context awareness"""
    try:
        embeddings = np.load("/data/embeddings.npy")
        with open("/data/chunks.txt", "r") as f:
            chunks = f.read().split("\n")
        
        question_embed = embedding_model.encode(question)
        scores = np.dot(embeddings, question_embed)
        
        # Get top 3 answers for better context
        top_indices = np.argsort(scores)[-3:][::-1]
        
        return {
            "answers": [chunks[i] for i in top_indices],
            "scores": [float(scores[i]) for i in top_indices],
            "best_answer": chunks[top_indices[0]],
            "confidence": float(scores[top_indices[0]])
        }
    except FileNotFoundError:
        return {"error": "No processed PDF found. Process a PDF first."}

# CLI entrypoints
@app.local_entrypoint()
def upload(file_path: str):
    """Process a local PDF file"""
    with open(file_path, "rb") as f:
        pdf_bytes = f.read()
    result = process_pdf.remote(pdf_bytes)
    print(f"Processed {file_path}: {result}")

@app.local_entrypoint()
def ask(question: str):
    """Enhanced query interface"""
    result = query.remote(question)
    if "error" in result:
        print("Error:", result["error"])
    else:
        print("\nBest Answer:")
        print(result["best_answer"])
        print(f"\nConfidence: {result['confidence']:.2f}")
        
        print("\nAdditional Context:")
        for i, (answer, score) in enumerate(zip(result["answers"], result["scores"])):
            print(f"\n[{i+1}] (Score: {score:.2f}):")
            print(answer)


# Add this to your existing Modal app
@app.function()
def get_query():
    # This returns the query function itself
    return query