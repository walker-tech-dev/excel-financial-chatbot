import agno
from flask import request
from milvus_setup import setup_milvus_collection
from excel_to_milvus import process_excel, insert_texts_to_milvus

app = agno.Agno()

collection = setup_milvus_collection()

@app.route('/ingest_excel', methods=['POST'])
def ingest_excel():
    file = request.files['file']
    texts = process_excel(file)
    insert_texts_to_milvus(texts, collection)
    return {"status": "success", "message": "Excel ingested and embeddings stored."}

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json.get('query')
    # Vectorize user input, search Milvus, get context, and query Ollama (to be implemented)
    return {"response": "Chatbot reply (implement retrieval and LLM call here)"}

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)