import pandas as pd
import streamlit as st
import numpy as np
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType
import requests
import json
import logging
import concurrent.futures
import hashlib
import pickle
import os
from typing import List, Dict, Any
import time
from functools import lru_cache

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Performance optimizations
BATCH_SIZE = 500  # Increased batch size
MAX_WORKERS = 8   # Parallel processing threads
EMBEDDING_CACHE_SIZE = 1000  # LRU cache for embeddings
CHUNK_SIZE = 1000  # Process files in chunks

# Enhanced financial and business terms (optimized)
ENHANCED_FINANCIAL_TERMS = [
    'revenue', 'annual revenue', 'monthly revenue', 'health score', 'churn risk',
    'support ticket', 'api calls', 'customer satisfaction', 'renewal likelihood'
]

# Performance monitoring
class PerformanceMonitor:
    def __init__(self):
        self.start_time = None
        self.metrics = {}
    
    def start_timer(self, operation):
        self.start_time = time.time()
        self.metrics[operation] = {'start': self.start_time}
    
    def end_timer(self, operation):
        if operation in self.metrics:
            self.metrics[operation]['duration'] = time.time() - self.metrics[operation]['start']
            logger.info(f"{operation} completed in {self.metrics[operation]['duration']:.2f} seconds")

monitor = PerformanceMonitor()

# Fast embedding cache
@lru_cache(maxsize=EMBEDDING_CACHE_SIZE)
def get_cached_embedding(text_hash: str, text: str, context_type: str = "general"):
    """
    Cached embedding generation with hash-based lookup
    """
    return create_fast_embedding(text, context_type)

def create_fast_embedding(text, context_type="general"):
    """
    Optimized embedding function for speed
    """
    try:
        # Simplified context enhancement for speed
        if context_type == "revenue" and "revenue" not in text.lower():
            text = f"revenue financial {text}"
        elif context_type == "customer" and "customer" not in text.lower():
            text = f"customer health {text}"
        
        response = requests.post(
            'http://localhost:11434/api/embeddings',
            json={'model': 'llama3.2:3b', 'prompt': text[:1000]},  # Limit text length
            timeout=10  # Add timeout
        )
        
        if response.status_code == 200:
            embedding = response.json()['embedding']
            embedding = np.array(embedding, dtype=np.float32)
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
            return embedding.tolist()
        else:
            logger.error(f"Embedding API error: {response.status_code}")
            return None
    except Exception as e:
        logger.error(f"Error creating embedding: {e}")
        return None

# Connect to Milvus with optimizations
def connect_to_milvus():
    try:
        connections.connect(
            "default", 
            host="localhost", 
            port="19530",
            # Performance optimizations
            pool_size=20,
            max_idle_time=300
        )
        logger.info("Connected to Milvus with optimizations")
        return True
    except Exception as e:
        logger.error(f"Failed to connect to Milvus: {e}")
        return False

# Optimized collection creation
def create_optimized_collection():
    try:
        from pymilvus import utility
        if utility.has_collection("fast_financial_data"):
            collection = Collection("fast_financial_data")
            logger.info("Using existing optimized collection")
            return collection
        
        # Optimized schema for performance
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=2000),  # Reduced for speed
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=4096),
            FieldSchema(name="source_file", dtype=DataType.VARCHAR, max_length=100),
            FieldSchema(name="data_type", dtype=DataType.VARCHAR, max_length=50),
            FieldSchema(name="customer", dtype=DataType.VARCHAR, max_length=50),
            FieldSchema(name="product", dtype=DataType.VARCHAR, max_length=50),
            FieldSchema(name="revenue_amount", dtype=DataType.FLOAT),
            FieldSchema(name="context_type", dtype=DataType.VARCHAR, max_length=30)
        ]
        
        schema = CollectionSchema(fields, "Optimized financial data collection")
        collection = Collection("fast_financial_data", schema)
        
        # High-performance index
        index_params = {
            "metric_type": "COSINE",
            "index_type": "IVF_SQ8",  # Faster index type
            "params": {"nlist": 2048}  # Optimized for speed
        }
        collection.create_index("embedding", index_params)
        
        logger.info("Created optimized Milvus collection")
        return collection
        
    except Exception as e:
        logger.error(f"Error creating collection: {e}")
        return None

# Fast data processing with parallel execution
def process_file_chunk(args):
    """
    Process a chunk of data in parallel
    """
    chunk_data, config, start_idx = args
    processed_records = []
    
    for idx, (_, row) in enumerate(chunk_data.iterrows()):
        try:
            # Fast text creation
            text_parts = []
            customer = ""
            product = ""
            revenue_amount = 0.0
            
            # Optimized data extraction
            for col, val in row.items():
                if pd.notna(val):
                    col_lower = col.lower()
                    if 'customer' in col_lower:
                        customer = str(val)
                        text_parts.append(f"Customer: {val}")
                    elif 'product' in col_lower and 'line' not in col_lower:
                        product = str(val)
                        text_parts.append(f"Product: {val}")
                    elif 'revenue' in col_lower:
                        revenue_amount = float(val) if isinstance(val, (int, float)) else 0.0
                        text_parts.append(f"{col}: ${val:,}")
                    elif col_lower in ['health score', 'churn risk', 'csat score']:
                        text_parts.append(f"{col}: {val}")
                    else:
                        text_parts.append(f"{col}: {str(val)[:50]}")  # Limit length
            
            # Create compact text
            text = f"{config['type']}: " + ". ".join(text_parts[:10])  # Limit parts
            
            # Generate text hash for caching
            text_hash = hashlib.md5(text.encode()).hexdigest()
            
            # Fast embedding with cache
            embedding = get_cached_embedding(text_hash, text, config["context"])
            
            if embedding:
                processed_records.append({
                    "text": text,
                    "embedding": embedding,
                    "source_file": config["path"].split("/")[-1],
                    "data_type": config["type"],
                    "customer": customer,
                    "product": product,
                    "revenue_amount": revenue_amount,
                    "context_type": config["context"]
                })
        
        except Exception as e:
            logger.error(f"Error processing row {start_idx + idx}: {e}")
            continue
    
    return processed_records

# Optimized data insertion with parallel processing
def insert_fast_data_to_milvus(collection, force_refresh=False, progress_callback=None):
    """
    High-speed data insertion with parallel processing and chunking
    """
    try:
        # Check existing data
        existing_count = check_existing_data(collection)
        
        if existing_count > 0 and not force_refresh:
            logger.info(f"Using existing {existing_count} records for speed")
            return existing_count
        
        if force_refresh and existing_count > 0:
            logger.info("Dropping collection for refresh...")
            collection.drop()
            collection = create_optimized_collection()
        
        monitor.start_timer("Total Processing")
        
        # Optimized file configurations
        files_config = [
            {
                "path": "d:/Dev_space/chatbot/Excel/Uniform_Product revenue data.xlsx",
                "type": "revenue",
                "context": "revenue",
                "sample_size": None  # Process all for small files
            },
            {
                "path": "d:/Dev_space/chatbot/Excel/uniform_gainsight_data.csv",
                "type": "customer_health",
                "context": "customer",
                "sample_size": 500  # Sample large files for speed
            },
            {
                "path": "d:/Dev_space/chatbot/Excel/uniform_salesforce_data.csv",
                "type": "support",
                "context": "support",
                "sample_size": 1000  # Sample large files
            },
            {
                "path": "d:/Dev_space/chatbot/Excel/uniform_product_usage_data.csv",
                "type": "usage",
                "context": "usage",
                "sample_size": 500
            },
            {
                "path": "d:/Dev_space/chatbot/Excel/uniform_jira_data.csv",
                "type": "projects",
                "context": "product",
                "sample_size": 1000
            }
        ]
        
        all_processed_data = []
        total_files = len(files_config)
        
        for file_idx, config in enumerate(files_config):
            try:
                monitor.start_timer(f"Processing {config['type']}")
                
                # Load data with sampling for speed
                if config["path"].endswith('.xlsx'):
                    df = pd.read_excel(config["path"])
                else:
                    df = pd.read_csv(config["path"])
                
                # Sample large datasets for speed
                if config["sample_size"] and len(df) > config["sample_size"]:
                    df = df.sample(n=config["sample_size"], random_state=42)
                    logger.info(f"Sampled {config['sample_size']} records from {config['type']} for speed")
                
                # Process in parallel chunks
                chunks = [df[i:i+CHUNK_SIZE] for i in range(0, len(df), CHUNK_SIZE)]
                chunk_args = [(chunk, config, i*CHUNK_SIZE) for i, chunk in enumerate(chunks)]
                
                with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                    chunk_results = list(executor.map(process_file_chunk, chunk_args))
                
                # Flatten results
                for chunk_result in chunk_results:
                    all_processed_data.extend(chunk_result)
                
                monitor.end_timer(f"Processing {config['type']}")
                
                # Update progress
                if progress_callback:
                    progress = (file_idx + 1) / total_files
                    progress_callback(progress, f"Processed {config['type']}")
                
            except Exception as e:
                logger.error(f"Error processing {config['path']}: {e}")
                continue
        
        # Fast batch insertion
        monitor.start_timer("Milvus Insertion")
        
        total_inserted = 0
        for i in range(0, len(all_processed_data), BATCH_SIZE):
            batch = all_processed_data[i:i + BATCH_SIZE]
            
            # Prepare batch data
            texts = [item["text"] for item in batch]
            embeddings = [item["embedding"] for item in batch]
            source_files = [item["source_file"] for item in batch]
            data_types = [item["data_type"] for item in batch]
            customers = [item["customer"] for item in batch]
            products = [item["product"] for item in batch]
            revenue_amounts = [item["revenue_amount"] for item in batch]
            context_types = [item["context_type"] for item in batch]
            
            entities = [texts, embeddings, source_files, data_types, customers, products, revenue_amounts, context_types]
            
            collection.insert(entities)
            total_inserted += len(batch)
            
            # Progress update
            if progress_callback:
                progress = total_inserted / len(all_processed_data)
                progress_callback(progress, f"Inserted {total_inserted}/{len(all_processed_data)} records")
        
        collection.flush()
        monitor.end_timer("Milvus Insertion")
        monitor.end_timer("Total Processing")
        
        logger.info(f"Fast processing completed: {total_inserted} records")
        return total_inserted
        
    except Exception as e:
        logger.error(f"Error in fast data insertion: {e}")
        return 0

# Check existing data (unchanged)
def check_existing_data(collection):
    try:
        collection.load()
        count = collection.num_entities
        logger.info(f"Found {count} existing entities in collection")
        return count
    except Exception as e:
        logger.error(f"Error checking existing data: {e}")
        return 0

# Optimized search function
def search_fast_milvus(collection, query, top_k=5):
    """
    High-speed search with optimized parameters
    """
    try:
        # Fast embedding generation
        text_hash = hashlib.md5(query.encode()).hexdigest()
        query_embedding = get_cached_embedding(text_hash, query)
        
        if not query_embedding:
            return []
        
        collection.load()
        
        # High-speed search parameters
        search_params = {
            "metric_type": "COSINE",
            "params": {"nprobe": 32}  # Balanced speed/accuracy
        }
        
        results = collection.search(
            data=[query_embedding],
            anns_field="embedding",
            param=search_params,
            limit=top_k,
            output_fields=["text", "source_file", "data_type", "customer", "product", "revenue_amount"]
        )
        
        processed_results = []
        for hit in results[0]:
            processed_results.append({
                "text": hit.entity.get("text", ""),
                "score": hit.score,
                "source_file": hit.entity.get("source_file", ""),
                "data_type": hit.entity.get("data_type", ""),
                "customer": hit.entity.get("customer", ""),
                "product": hit.entity.get("product", ""),
                "revenue_amount": hit.entity.get("revenue_amount", 0)
            })
        
        return processed_results
        
    except Exception as e:
        logger.error(f"Error in fast search: {e}")
        return []

# Fast chat function
def fast_chat_with_bot(user_query, collection):
    """
    Optimized chat function for speed
    """
    try:
        # Fast search
        search_results = search_fast_milvus(collection, user_query, top_k=5)
        
        if not search_results:
            return "No relevant information found. Please try a different query."
        
        # Compact context creation
        context_parts = [result["text"][:300] for result in search_results[:3]]  # Limit context
        context = "\\n\\n".join(context_parts)
        
        # Streamlined prompt
        prompt = f"""Based on this financial data:

{context}

Question: {user_query}

Provide a concise, data-driven answer:"""

        # Fast API call
        response = requests.post(
            'http://localhost:11434/api/generate',
            json={
                'model': 'llama3.2:3b',
                'prompt': prompt,
                'stream': False,
                'options': {'temperature': 0.3, 'num_predict': 300}  # Faster generation
            },
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()['response']
            return result
        else:
            return "Error processing request. Please try again."
            
    except Exception as e:
        logger.error(f"Error in fast chat: {e}")
        return "Error processing request. Please try again."

# Streamlit interface with performance optimizations
def main():
    st.set_page_config(page_title="⚡ High-Speed Financial AI", layout="wide")
    
    st.title("⚡ High-Speed Financial Data Intelligence")
    st.markdown("*Optimized for speed and scalability*")
    
    # Initialize session state
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'collection' not in st.session_state:
        st.session_state.collection = None
    
    # Performance-focused sidebar
    with st.sidebar:
        st.header("🚀 Speed Control")
        
        # Check system status
        if st.session_state.collection:
            existing_count = check_existing_data(st.session_state.collection)
            if existing_count > 0:
                st.success(f"⚡ Ready ({existing_count:,} records)")
                st.metric("Performance", "Optimized", "⚡ Fast Mode")
            else:
                st.warning("⚠️ No data found")
        
        # Fast initialization
        if not st.session_state.collection or check_existing_data(st.session_state.collection) == 0:
            if st.button("⚡ Fast Setup", type="primary", help="Quick setup with sampling for speed"):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                def update_progress(progress, message):
                    progress_bar.progress(progress)
                    status_text.text(message)
                
                with st.spinner("Fast processing..."):
                    if connect_to_milvus():
                        collection = create_optimized_collection()
                        if collection:
                            st.session_state.collection = collection
                            count = insert_fast_data_to_milvus(collection, progress_callback=update_progress)
                            if count > 0:
                                st.success(f"⚡ Ready! {count:,} records in seconds")
                                st.rerun()
        
        # Speed controls
        if st.session_state.collection and check_existing_data(st.session_state.collection) > 0:
            st.subheader("⚡ Quick Actions")
            
            quick_queries = [
                "Total revenue?",
                "Top customers?",
                "Risk customers?",
                "Product performance?"
            ]
            
            for query in quick_queries:
                if st.button(query, key=f"quick_{query}"):
                    st.session_state.messages.append({"role": "user", "content": query})
                    response = fast_chat_with_bot(query, st.session_state.collection)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    st.rerun()
    
    # Main interface
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # Fast chat input
        if prompt := st.chat_input("Ask quickly..."):
            if not st.session_state.collection or check_existing_data(st.session_state.collection) == 0:
                st.error("Please run Fast Setup first.")
            else:
                st.session_state.messages.append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.markdown(prompt)
                
                with st.chat_message("assistant"):
                    with st.spinner("⚡ Fast analysis..."):
                        response = fast_chat_with_bot(prompt, st.session_state.collection)
                        st.markdown(response)
                        st.session_state.messages.append({"role": "assistant", "content": response})
    
    with col2:
        st.subheader("⚡ Speed Stats")
        
        if hasattr(monitor, 'metrics') and monitor.metrics:
            for operation, data in monitor.metrics.items():
                if 'duration' in data:
                    st.metric(operation, f"{data['duration']:.1f}s", "⚡")
        
        st.subheader("🎯 Optimizations")
        optimizations = [
            "✅ Parallel processing",
            "✅ Smart sampling", 
            "✅ Embedding cache",
            "✅ Fast indexing",
            "✅ Batch operations"
        ]
        
        for opt in optimizations:
            st.write(opt)

if __name__ == "__main__":
    main()