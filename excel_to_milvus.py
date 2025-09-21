import pandas as pd
from pymilvus import Collection
import requests

# These should be configured or imported from your config
OLLAMA_URL = "http://localhost:11434/api/embeddings"
OLLAMA_MODEL = "llama2"
MILVUS_COLLECTION = "excel_vectors"

def get_embedding(text):
    response = requests.post(OLLAMA_URL, json={"model": OLLAMA_MODEL, "prompt": text})
    response.raise_for_status()
    return response.json()["embedding"]

def process_excel(excel_file):
    df = pd.read_excel(excel_file)
    texts = df.astype(str).agg(" ".join, axis=1).tolist()
    return texts

def insert_texts_to_milvus(texts, collection: Collection):
    data_to_insert = [[], [], []]  # id, text, embedding
    for text in texts:
        embedding = get_embedding(text)
        data_to_insert[1].append(text)
        data_to_insert[2].append(embedding)
    collection.insert(data_to_insert)