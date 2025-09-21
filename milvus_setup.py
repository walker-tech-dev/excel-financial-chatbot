from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection

MILVUS_HOST = "localhost"
MILVUS_PORT = "19530"
MILVUS_COLLECTION = "excel_vectors"

def setup_milvus_collection():
    connections.connect(host=MILVUS_HOST, port=MILVUS_PORT)
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=1024),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=4096),  # Adjust dim as per your model
    ]
    schema = CollectionSchema(fields, "Excel text embeddings")
    if MILVUS_COLLECTION not in Collection.list():
        collection = Collection(name=MILVUS_COLLECTION, schema=schema)
    else:
        collection = Collection(MILVUS_COLLECTION)
    return collection