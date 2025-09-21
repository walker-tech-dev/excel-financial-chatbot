import pandas as pd
import streamlit as st
import numpy as np
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType
import requests
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Enhanced financial and business terms based on dataset analysis
ENHANCED_FINANCIAL_TERMS = [
    # Revenue terms
    'revenue', 'annual revenue', 'monthly revenue', 'total revenue', 'revenue analysis',
    'income', 'earnings', 'sales', 'profit', 'loss', 'financial performance',
    
    # Customer health and relationship terms
    'health score', 'customer health', 'churn risk', 'renewal likelihood', 'retention',
    'customer satisfaction', 'csat score', 'customer success', 'loyalty',
    
    # Support and service terms
    'support ticket', 'ticket id', 'priority', 'status', 'resolution', 'ttr',
    'time to resolve', 'escalation', 'customer feedback', 'service level',
    
    # Product and usage terms
    'product usage', 'api calls', 'activations', 'devices', 'users', 'entitlements',
    'served devices', 'fulfillments', 'downloads', 'adoption', 'utilization',
    
    # Business intelligence terms
    'customer 360', 'risk analysis', 'portfolio performance', 'top customers',
    'business intelligence', 'kpi', 'metrics', 'dashboard', 'analytics',
    
    # Product lines
    'fna', 'fnb', 'fnc', 'product line', 'product mix',
    
    # Jira and project terms
    'jira', 'issue', 'bug', 'story', 'project', 'assignee', 'reporter',
    'epic', 'sprint', 'backlog', 'development', 'testing',
    
    # Time-based analysis
    'monthly', 'quarterly', 'annual', 'trend', 'growth', 'decline',
    'seasonality', 'forecast', 'projection'
]

# Connect to Milvus
def connect_to_milvus():
    try:
        connections.connect("default", host="localhost", port="19530")
        logger.info("Connected to Milvus successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to connect to Milvus: {e}")
        return False

# Enhanced embedding function with business context
def create_enhanced_embedding(text, context_type="general"):
    """
    Create embeddings with enhanced business and financial context
    """
    # Add context-specific keywords to boost relevance
    context_keywords = {
        "revenue": ["revenue", "financial", "income", "earnings", "profit", "sales"],
        "customer": ["customer", "client", "account", "health", "satisfaction", "churn"],
        "support": ["support", "ticket", "issue", "service", "resolution", "escalation"],
        "usage": ["usage", "utilization", "adoption", "activity", "engagement"],
        "product": ["product", "feature", "functionality", "capability", "offering"],
        "general": []
    }
    
    # Enhance text with context
    enhanced_text = text
    if context_type in context_keywords:
        keywords = context_keywords[context_type]
        enhanced_text = f"{' '.join(keywords)} {text}"
    
    # Add financial terms weighting
    for term in ENHANCED_FINANCIAL_TERMS:
        if term.lower() in text.lower():
            enhanced_text = f"{term} {enhanced_text}"
    
    # Create embedding using Ollama
    try:
        response = requests.post(
            'http://localhost:11434/api/embeddings',
            json={
                'model': 'llama3.2:3b',
                'prompt': enhanced_text
            }
        )
        if response.status_code == 200:
            embedding = response.json()['embedding']
            # Normalize embedding
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

# Enhanced context detection
def detect_query_context(query):
    """
    Detect the main context of a user query for better embedding
    """
    query_lower = query.lower()
    
    # Revenue context
    if any(term in query_lower for term in ['revenue', 'income', 'profit', 'sales', 'financial', 'money', '$']):
        return "revenue"
    
    # Customer context
    elif any(term in query_lower for term in ['customer', 'health', 'churn', 'renewal', 'satisfaction', 'csat']):
        return "customer"
    
    # Support context
    elif any(term in query_lower for term in ['support', 'ticket', 'issue', 'problem', 'resolution', 'escalation']):
        return "support"
    
    # Usage context
    elif any(term in query_lower for term in ['usage', 'utilization', 'api', 'devices', 'users', 'activations']):
        return "usage"
    
    # Product context
    elif any(term in query_lower for term in ['product', 'fna', 'fnb', 'fnc', 'feature']):
        return "product"
    
    return "general"

# Create Milvus collection with enhanced schema
def create_milvus_collection():
    try:
        # Check if collection exists
        from pymilvus import utility
        if utility.has_collection("financial_data"):
            collection = Collection("financial_data")
            logger.info("Using existing collection")
            return collection
        
        # Define enhanced schema
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=5000),  # Increased for complex data
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=4096),
            FieldSchema(name="source_file", dtype=DataType.VARCHAR, max_length=200),
            FieldSchema(name="data_type", dtype=DataType.VARCHAR, max_length=100),
            FieldSchema(name="customer", dtype=DataType.VARCHAR, max_length=100),
            FieldSchema(name="product", dtype=DataType.VARCHAR, max_length=100),
            FieldSchema(name="revenue_amount", dtype=DataType.FLOAT),
            FieldSchema(name="context_type", dtype=DataType.VARCHAR, max_length=50)
        ]
        
        schema = CollectionSchema(fields, "Enhanced financial data collection")
        collection = Collection("financial_data", schema)
        
        # Create enhanced index
        index_params = {
            "metric_type": "COSINE",
            "index_type": "IVF_FLAT",
            "params": {"nlist": 128}
        }
        collection.create_index("embedding", index_params)
        
        logger.info("Created new enhanced Milvus collection")
        return collection
        
    except Exception as e:
        logger.error(f"Error creating collection: {e}")
        return None

# Check if data already exists in collection
def check_existing_data(collection):
    """
    Check if collection already has data and return count
    """
    try:
        collection.load()
        count = collection.num_entities
        logger.info(f"Found {count} existing entities in collection")
        return count
    except Exception as e:
        logger.error(f"Error checking existing data: {e}")
        return 0

# Enhanced data insertion with metadata
def insert_enhanced_data_to_milvus(collection, force_refresh=False):
    """
    Insert data with enhanced metadata and context awareness
    Only processes if collection is empty or force_refresh is True
    """
    try:
        # Check if data already exists
        existing_count = check_existing_data(collection)
        
        if existing_count > 0 and not force_refresh:
            logger.info(f"Collection already contains {existing_count} records. Skipping data insertion.")
            return existing_count
        
        if force_refresh and existing_count > 0:
            logger.info("Force refresh requested. Dropping existing collection...")
            collection.drop()
            # Recreate collection
            collection = create_milvus_collection()
        
        logger.info("Starting fresh data insertion with enhanced embeddings...")
        
        # File configurations with enhanced metadata
        files_config = [
            {
                "path": "d:/Dev_space/chatbot/Excel/Uniform_Product revenue data.xlsx",
                "type": "revenue",
                "context": "revenue",
                "priority": 10
            },
            {
                "path": "d:/Dev_space/chatbot/Excel/uniform_gainsight_data.csv",
                "type": "customer_health",
                "context": "customer",
                "priority": 9
            },
            {
                "path": "d:/Dev_space/chatbot/Excel/uniform_salesforce_data.csv",
                "type": "support",
                "context": "support",
                "priority": 8
            },
            {
                "path": "d:/Dev_space/chatbot/Excel/uniform_product_usage_data.csv",
                "type": "usage",
                "context": "usage",
                "priority": 7
            },
            {
                "path": "d:/Dev_space/chatbot/Excel/uniform_jira_data.csv",
                "type": "projects",
                "context": "product",
                "priority": 6
            }
        ]
        
        all_data = []
        
        for config in files_config:
            try:
                # Load data
                if config["path"].endswith('.xlsx'):
                    df = pd.read_excel(config["path"])
                else:
                    df = pd.read_csv(config["path"])
                
                logger.info(f"Processing {config['type']} data: {df.shape}")
                
                # Process each row with enhanced context
                for idx, row in df.iterrows():
                    # Create enhanced text representation
                    text_parts = []
                    customer = ""
                    product = ""
                    revenue_amount = 0.0
                    
                    # Extract key information
                    for col, val in row.items():
                        if pd.notna(val):
                            if 'customer' in col.lower():
                                customer = str(val)
                                text_parts.append(f"Customer: {val}")
                            elif 'product' in col.lower() and 'line' not in col.lower():
                                product = str(val)
                                text_parts.append(f"Product: {val}")
                            elif 'revenue' in col.lower():
                                revenue_amount = float(val) if isinstance(val, (int, float)) else 0.0
                                text_parts.append(f"{col}: ${val:,}")
                            elif col.lower() in ['health score', 'churn risk', 'renewal likelihood', 'csat score']:
                                text_parts.append(f"{col}: {val}")
                            else:
                                text_parts.append(f"{col}: {val}")
                    
                    # Create comprehensive text
                    enhanced_text = f"Data Type: {config['type']}. " + ". ".join(text_parts)
                    
                    # Add business context
                    if config["type"] == "revenue":
                        enhanced_text += f". Business Context: Revenue analysis for {customer} using {product}."
                    elif config["type"] == "customer_health":
                        enhanced_text += f". Business Context: Customer health and success metrics for {customer}."
                    elif config["type"] == "support":
                        enhanced_text += f". Business Context: Support and service quality metrics for {customer}."
                    elif config["type"] == "usage":
                        enhanced_text += f". Business Context: Product usage and adoption patterns for {customer}."
                    elif config["type"] == "projects":
                        enhanced_text += f". Business Context: Project management and development activities for {customer}."
                    
                    # Create embedding with context
                    embedding = create_enhanced_embedding(enhanced_text, config["context"])
                    
                    if embedding:
                        all_data.append({
                            "text": enhanced_text,
                            "embedding": embedding,
                            "source_file": config["path"].split("/")[-1],
                            "data_type": config["type"],
                            "customer": customer,
                            "product": product,
                            "revenue_amount": revenue_amount,
                            "context_type": config["context"]
                        })
                
                logger.info(f"Processed {len(df)} rows from {config['type']} data")
                
            except Exception as e:
                logger.error(f"Error processing {config['path']}: {e}")
                continue
        
        # Insert in batches
        batch_size = 100
        total_inserted = 0
        
        for i in range(0, len(all_data), batch_size):
            batch = all_data[i:i + batch_size]
            
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
            
            if i % (batch_size * 10) == 0:
                logger.info(f"Inserted {total_inserted} entities so far...")
        
        collection.flush()
        logger.info(f"Successfully inserted {total_inserted} entities into Milvus")
        
        return total_inserted
        
    except Exception as e:
        logger.error(f"Error inserting data to Milvus: {e}")
        return 0

# Get collection statistics
def get_collection_stats(collection):
    """
    Get detailed statistics about the collection
    """
    try:
        collection.load()
        total_count = collection.num_entities
        
        # Get data type breakdown
        search_params = {"metric_type": "COSINE", "params": {"nprobe": 16}}
        
        # Sample some records to get statistics
        sample_results = collection.search(
            data=[[0.1] * 4096],  # Dummy vector for sampling
            anns_field="embedding",
            param=search_params,
            limit=100,
            output_fields=["data_type", "customer", "product", "revenue_amount"]
        )
        
        data_types = {}
        customers = set()
        products = set()
        total_revenue = 0
        
        for hit in sample_results[0]:
            entity = hit.entity
            data_type = entity.get("data_type", "unknown")
            data_types[data_type] = data_types.get(data_type, 0) + 1
            
            if entity.get("customer"):
                customers.add(entity.get("customer"))
            if entity.get("product"):
                products.add(entity.get("product"))
            if entity.get("revenue_amount", 0) > 0:
                total_revenue += entity.get("revenue_amount", 0)
        
        return {
            "total_records": total_count,
            "data_types": data_types,
            "unique_customers": len(customers),
            "unique_products": len(products),
            "sample_revenue": total_revenue
        }
        
    except Exception as e:
        logger.error(f"Error getting collection stats: {e}")
        return {"total_records": 0}

# Enhanced search function
def search_enhanced_milvus(collection, query, top_k=5):
    """
    Enhanced search with context awareness and business intelligence
    """
    try:
        # Detect query context
        context = detect_query_context(query)
        
        # Create enhanced query embedding
        query_embedding = create_enhanced_embedding(query, context)
        
        if not query_embedding:
            return []
        
        # Load collection
        collection.load()
        
        # Enhanced search parameters
        search_params = {
            "metric_type": "COSINE",
            "params": {"nprobe": 16}
        }
        
        # Perform search
        results = collection.search(
            data=[query_embedding],
            anns_field="embedding",
            param=search_params,
            limit=top_k,
            output_fields=["text", "source_file", "data_type", "customer", "product", "revenue_amount", "context_type"]
        )
        
        # Process and rank results
        processed_results = []
        for hit in results[0]:
            score = hit.score
            entity = hit.entity
            
            # Apply business logic boosting
            boost_factor = 1.0
            
            # Boost revenue-related results for financial queries
            if context == "revenue" and entity.get("data_type") == "revenue":
                boost_factor *= 1.5
            
            # Boost customer health for customer queries
            elif context == "customer" and entity.get("data_type") == "customer_health":
                boost_factor *= 1.3
            
            # Boost high-revenue customers
            if entity.get("revenue_amount", 0) > 50000:
                boost_factor *= 1.2
            
            final_score = score * boost_factor
            
            processed_results.append({
                "text": entity.get("text", ""),
                "score": final_score,
                "source_file": entity.get("source_file", ""),
                "data_type": entity.get("data_type", ""),
                "customer": entity.get("customer", ""),
                "product": entity.get("product", ""),
                "revenue_amount": entity.get("revenue_amount", 0),
                "context_type": entity.get("context_type", "")
            })
        
        # Sort by enhanced score
        processed_results.sort(key=lambda x: x["score"], reverse=True)
        
        logger.info(f"Found {len(processed_results)} enhanced results for query: {query}")
        
        return processed_results
        
    except Exception as e:
        logger.error(f"Error searching Milvus: {e}")
        return []

# Enhanced chat function with business intelligence
def enhanced_chat_with_bot(user_query, collection):
    """
    Enhanced chat function with improved context and business intelligence
    """
    try:
        # Search for relevant context
        search_results = search_enhanced_milvus(collection, user_query, top_k=8)
        
        if not search_results:
            return "I couldn't find relevant information in the database. Please try rephrasing your question."
        
        # Build enhanced context
        context_parts = []
        revenue_data = []
        customer_data = []
        
        for result in search_results:
            context_parts.append(result["text"])
            
            # Collect specific data types for analysis
            if result["data_type"] == "revenue" and result["revenue_amount"] > 0:
                revenue_data.append({
                    "customer": result["customer"],
                    "product": result["product"],
                    "amount": result["revenue_amount"]
                })
            
            if result["customer"]:
                customer_data.append(result["customer"])
        
        # Create comprehensive context
        enhanced_context = "\\n\\n".join(context_parts)
        
        # Add business intelligence summary
        if revenue_data:
            total_revenue = sum(item["amount"] for item in revenue_data)
            unique_customers = len(set(item["customer"] for item in revenue_data))
            enhanced_context += f"\\n\\nBusiness Intelligence Summary: Total revenue in context: ${total_revenue:,} across {unique_customers} customers."
        
        # Enhanced prompt with business context
        enhanced_prompt = f"""You are a business intelligence assistant analyzing financial and operational data.

Context Information:
{enhanced_context}

User Question: {user_query}

Instructions:
1. Provide specific, data-driven answers based on the context
2. Include relevant numbers, percentages, and financial figures when available
3. Identify trends, patterns, and business insights
4. Mention specific customers, products, and metrics when relevant
5. If analyzing revenue, provide totals and breakdowns
6. For customer health queries, mention risk factors and opportunities
7. Be concise but comprehensive in your analysis

Response:"""

        # Send to Ollama
        response = requests.post(
            'http://localhost:11434/api/generate',
            json={
                'model': 'llama3.2:3b',
                'prompt': enhanced_prompt,
                'stream': False
            }
        )
        
        if response.status_code == 200:
            result = response.json()['response']
            
            # Add data source attribution
            unique_sources = list(set(r["source_file"] for r in search_results))
            result += f"\\n\\n*Data sources: {', '.join(unique_sources)}*"
            
            return result
        else:
            return "I encountered an error while processing your request. Please try again."
            
    except Exception as e:
        logger.error(f"Error in enhanced chat: {e}")
        return "I encountered an error while processing your request. Please try again."

# Streamlit interface
def main():
    st.set_page_config(page_title="Enhanced Financial Data Chatbot", layout="wide")
    
    st.title("🔍 Enhanced Financial Data Intelligence Chatbot")
    st.markdown("*Powered by RAG with business intelligence and enhanced embeddings*")
    
    # Initialize session state
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'collection' not in st.session_state:
        st.session_state.collection = None
    
    # Sidebar for system status
    with st.sidebar:
        st.header("🔧 System Control")
        
        # Check if system is already initialized
        if st.session_state.collection:
            existing_count = check_existing_data(st.session_state.collection)
            if existing_count > 0:
                st.success(f"🟢 System Ready ({existing_count:,} records)")
                
                # Show collection statistics
                with st.expander("📊 Collection Statistics", expanded=False):
                    stats = get_collection_stats(st.session_state.collection)
                    st.write(f"**Total Records:** {stats['total_records']:,}")
                    if 'data_types' in stats:
                        st.write("**Data Types:**")
                        for dtype, count in stats['data_types'].items():
                            st.write(f"  • {dtype}: {count}")
                    if stats.get('unique_customers', 0) > 0:
                        st.write(f"**Customers:** {stats['unique_customers']}")
                    if stats.get('unique_products', 0) > 0:
                        st.write(f"**Products:** {stats['unique_products']}")
                
                # Option to refresh data
                st.subheader("🔄 Data Management")
                
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("🔄 Refresh Data", type="secondary", help="Re-process all datasets with fresh embeddings"):
                        with st.spinner("Refreshing all data..."):
                            if connect_to_milvus():
                                collection = create_milvus_collection()
                                if collection:
                                    st.session_state.collection = collection
                                    count = insert_enhanced_data_to_milvus(collection, force_refresh=True)
                                    if count > 0:
                                        st.success(f"✅ Refreshed {count:,} records")
                                        st.rerun()
                                    else:
                                        st.error("❌ Failed to refresh data")
                                else:
                                    st.error("❌ Failed to create collection")
                            else:
                                st.error("❌ Failed to connect to Milvus")
                
                with col2:
                    if st.button("📊 Update Stats", help="Refresh collection statistics"):
                        st.rerun()
            else:
                st.warning("⚠️ Collection exists but no data found")
        
        # Initialize system button (only show if not initialized)
        if not st.session_state.collection or check_existing_data(st.session_state.collection) == 0:
            if st.button("🚀 Initialize System", type="primary", help="Set up the RAG system with enhanced embeddings"):
                with st.spinner("Initializing enhanced system..."):
                    if connect_to_milvus():
                        st.success("✅ Connected to Milvus")
                        
                        with st.spinner("Setting up collection..."):
                            collection = create_milvus_collection()
                            if collection:
                                st.session_state.collection = collection
                                st.success("✅ Collection ready")
                                
                                # Check if data already exists
                                existing_count = check_existing_data(collection)
                                if existing_count > 0:
                                    st.info(f"ℹ️ Found {existing_count:,} existing records")
                                    st.success("✅ System ready with existing data!")
                                else:
                                    with st.spinner("Processing datasets (this may take 15-20 minutes)..."):
                                        count = insert_enhanced_data_to_milvus(collection, force_refresh=False)
                                        if count > 0:
                                            st.success(f"✅ Processed {count:,} records")
                                        else:
                                            st.error("❌ Failed to process data")
                                st.rerun()
                            else:
                                st.error("❌ Failed to create collection")
                    else:
                        st.error("❌ Failed to connect to Milvus")
            
        if st.session_state.collection and check_existing_data(st.session_state.collection) > 0:
            st.success("🟢 System Ready")
            
            # Enhanced suggestion buttons
            st.subheader("💡 Smart Query Suggestions")
            
            suggestions = [
                "What is the total revenue by customer?",
                "Which customers are at high churn risk?",
                "Show me customers with revenue over $50,000",
                "Which products generate the most revenue?",
                "Customer 360 view for Alpha01",
                "Revenue at risk from unhealthy customers",
                "Top 5 customers by support tickets",
                "API usage patterns and revenue correlation"
            ]
            
            for suggestion in suggestions:
                if st.button(suggestion, key=f"btn_{suggestion}"):
                    st.session_state.messages.append({"role": "user", "content": suggestion})
                    with st.spinner("Analyzing..."):
                        response = enhanced_chat_with_bot(suggestion, st.session_state.collection)
                        st.session_state.messages.append({"role": "assistant", "content": response})
                    st.rerun()
        else:
            st.info("🔵 Initialize system to enable smart suggestions")
    
    # Main chat interface
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # Chat input
        if prompt := st.chat_input("Ask about your financial data..."):
            if not st.session_state.collection or check_existing_data(st.session_state.collection) == 0:
                st.error("Please initialize the system first using the sidebar.")
            else:
                # Add user message
                st.session_state.messages.append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.markdown(prompt)
                
                # Generate response
                with st.chat_message("assistant"):
                    with st.spinner("Analyzing with enhanced AI..."):
                        response = enhanced_chat_with_bot(prompt, st.session_state.collection)
                        st.markdown(response)
                        st.session_state.messages.append({"role": "assistant", "content": response})
    
    with col2:
        st.subheader("📊 Dataset Overview")
        
        dataset_info = {
            "Revenue Data": "20 customers, $954K total",
            "Support Tickets": "20K tickets, 100 customers",
            "Customer Health": "1K records, health scores",
            "Product Usage": "1K records, API calls",
            "Jira Projects": "10K issues, development"
        }
        
        for dataset, info in dataset_info.items():
            st.metric(dataset, info)
        
        st.subheader("🎯 Query Types")
        query_types = [
            "Revenue Analysis",
            "Customer Health",
            "Support Analytics",
            "Usage Patterns",
            "Cross-functional"
        ]
        
        for query_type in query_types:
            st.write(f"• {query_type}")

if __name__ == "__main__":
    main()