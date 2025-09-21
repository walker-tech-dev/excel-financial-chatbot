from pymilvus import connections, utility

def test_milvus_connection():
    try:
        # Connect to Milvus
        connections.connect("default", host="localhost", port="19530")
        print("✅ Successfully connected to Milvus!")
        
        # Check Milvus version
        print(f"Milvus version: {utility.get_server_version()}")
        
        # List collections (should be empty initially)
        collections = utility.list_collections()
        print(f"Current collections: {collections}")
        
        return True
    except Exception as e:
        print(f"❌ Failed to connect to Milvus: {e}")
        return False

if __name__ == "__main__":
    test_milvus_connection()
