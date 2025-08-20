import os
import urllib.request
from flask import Flask, request, jsonify
import chromadb
from sentence_transformers import SentenceTransformer
from gpt4all import GPT4All

app = Flask(__name__)

# -------------------------
from huggingface_hub import hf_hub_download

# -------------------------
# Download GPT4All J Groovy model from Hugging Face
model_path = hf_hub_download(repo_id="axelniyo/ggml-gpt4all-j-v1.3-groovy"
, 
                             filename="ggml-gpt4all-j-v1.3-groovy.bin")
model = GPT4All(model_path)


# -------------------------
# Initialize ChromaDB
# -------------------------
chroma_client = chromadb.Client()
collection = chroma_client.create_collection(name="knowledge_base")

# -------------------------
# Load embedding model
# -------------------------
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# -------------------------
# Routes
# -------------------------
@app.route("/add", methods=["POST"])
def add_document():
    data = request.json
    text = data.get("text")
    if not text:
        return jsonify({"error": "No text provided"}), 400

    embedding = embedder.encode([text])[0].tolist()
    collection.add(
        documents=[text],
        embeddings=[embedding],
        ids=[str(len(collection.get()["ids"]))]
    )
    return jsonify({"message": "Document added!"})

@app.route("/ask", methods=["POST"])
def ask():
    data = request.json
    question = data.get("question")
    if not question:
        return jsonify({"error": "No question provided"}), 400

    q_embedding = embedder.encode([question])[0].tolist()

    # Retrieve most relevant document
    results = collection.query(query_embeddings=[q_embedding], n_results=1)
    context = results["documents"][0][0] if results["documents"] else ""

    # Generate answer using GPT4All Mini
    prompt = f"Context: {context}\n\nQuestion: {question}\nAnswer:"
    answer = model.generate(prompt)

    return jsonify({"answer": answer})

# -------------------------
# Run Flask app on Render
# -------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
