from flask import Flask, request, jsonify
import chromadb
from sentence_transformers import SentenceTransformer
import subprocess

app = Flask(__name__)

# Initialize ChromaDB
chroma_client = chromadb.Client()
collection = chroma_client.create_collection(name="knowledge_base")

# Load embedding model
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Function to query Ollama
def query_ollama(prompt):
    result = subprocess.run(
        ["ollama", "run", "llama3"],
        input=prompt.encode("utf-8"),
        capture_output=True
    )
    return result.stdout.decode("utf-8")

@app.route("/add", methods=["POST"])
def add_document():
    data = request.json
    text = data.get("text")
    embedding = embedder.encode([text])[0].tolist()
    collection.add(documents=[text], embeddings=[embedding], ids=[str(len(collection.get()['ids']))])
    return jsonify({"message": "Document added!"})

@app.route("/ask", methods=["POST"])
def ask():
    data = request.json
    question = data.get("question")
    q_embedding = embedder.encode([question])[0].tolist()
    
    # Retrieve most relevant docs
    results = collection.query(query_embeddings=[q_embedding], n_results=1)
    context = results["documents"][0][0] if results["documents"] else ""

    # Ask Ollama with context
    answer = query_ollama(f"Context: {context}\n\nQuestion: {question}")
    return jsonify({"answer": answer})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
