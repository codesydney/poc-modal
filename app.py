import modal
import pdfplumber
from sentence_transformers import SentenceTransformer
import numpy as np
import io
import re
from typing import List, Dict
import lancedb
import pyarrow as pa
import os
import shutil

image = (
    modal.Image.debian_slim()
    .apt_install("git")  # Needed for LanceDB's internal operations
    .pip_install(
        "numpy==1.26.4",
        "pdfplumber",
        "sentence-transformers",
        "lancedb>=0.5.0",
        "pyarrow>=12.0.0",
        "pandas",
        "huggingface_hub[hf_transfer]",
        "huggingface_hub[hf_xet]"
    )
)

app = modal.App("rag-pdf-processor-lancedb", image=image)
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
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    
    chunks = []
    for para in paragraphs:
        if len(para) > 500:
            words = para.split()
            for i in range(0, len(words), 300):
                chunks.append(' '.join(words[i:i+300]))
        else:
            chunks.append(para)
    return chunks

def create_lancedb_table(chunks: List[str], embeddings: np.ndarray) -> str:
    """Create table from PDF chunks and embeddings"""
    # Process in temporary directory first
    tmp_db_path = "/tmp/lancedb"
    db = lancedb.connect(tmp_db_path)
    
    data = pa.Table.from_arrays(
        [
            pa.array([str(i) for i in range(len(chunks))]),
            pa.array(chunks),
            pa.array(embeddings.tolist()),
        ],
        names=["id", "text", "vector"]
    )
    
    # Create in temporary location
    db.create_table("pdf_chunks", data=data, mode="overwrite")
    
    # Now persist to volume
    vol_db_path = "/data/lancedb"
    if os.path.exists(tmp_db_path):
        if os.path.exists(vol_db_path):
            shutil.rmtree(vol_db_path)
        shutil.copytree(tmp_db_path, vol_db_path)
    
    return f"Table created and persisted to {vol_db_path}"

@app.function(volumes={"/data": vol}, timeout=600)
def process_pdf(pdf_bytes: bytes) -> Dict:
    """Process PDF and store embeddings in LanceDB"""
    text = ""
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += clean_text(page_text) + "\n\n"
    
    chunks = split_text(text)
    embeddings = embedding_model.encode(chunks)
   
    # Store in LanceDB
    create_lancedb_table(chunks, embeddings)
    
    # Explicitly commit volume changes
    vol.commit()

    # Directly use '/data' for the volume
    db = lancedb.connect("/data/lancedb")  # Access volume via /data path
    print("Tables after creation:", db.table_names())  # Should show ["pdf_chunks"]
    
    return {"status": "success", "num_chunks": len(chunks)}

@app.function(volumes={"/data": vol})
def query(question: str) -> dict:
    vol.reload()  # Ensure we have latest volume contents
    
    try:
        db = lancedb.connect("/data/lancedb")
        table_names = db.table_names()
        print("Available tables:", table_names)

        if not table_names or "pdf_chunks" not in table_names:
            return {"error": "No tables found. Please process a PDF first."}

        # Verify table exists and is accessible
        try:
            table = db.open_table("pdf_chunks")
            # Test read
            test_read = table.to_arrow().to_pandas()
            if len(test_read) == 0:
                return {"error": "Table exists but is empty"}
        except Exception as e:
            return {"error": f"Table access failed: {str(e)}"}

        # Proceed with query
        question_embed = embedding_model.encode(question).tolist()
        results = table.search(question_embed).limit(3).to_pandas()

        return {
            "answers": results["text"].tolist(),
            "scores": results["_distance"].tolist(),
            "best_answer": results["text"].iloc[0],
            "confidence": float(1 - results["_distance"].iloc[0])
        }

    except Exception as e:
        return {"error": f"Query failed: {str(e)}"}

# CLI entrypoints (unchanged)
@app.local_entrypoint()
def upload(file_path: str):
    """Process a local PDF file"""
    with open(file_path, "rb") as f:
        pdf_bytes = f.read()
    result = process_pdf.remote(pdf_bytes)
    print(f"Processed {file_path}: {result}")

@app.local_entrypoint()
def ask(question: str, top_k: int = 3):
    """Query the processed PDF"""
    result = query.remote(question)
    if "error" in result:
        print("Error:", result["error"])
    else:
        print("\nBest Answer:")
        print(result["best_answer"])
        print(f"\nConfidence: {1 - result['scores'][0]:.2f}")
        
        print("\nAdditional Context:")
        for i, (answer, score) in enumerate(zip(result["answers"], result["scores"])):
            print(f"\n[{i+1}] (Score: {1 - score:.2f}):")
            print(answer)