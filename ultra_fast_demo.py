import streamlit as st
import pandas as pd
import numpy as np
import hashlib
import time
import concurrent.futures
from functools import lru_cache
import requests
import json

# Ultra-fast configuration
ULTRA_FAST_CONFIG = {
    'max_records_per_file': 50,  # Severely limit for demo speed
    'batch_size': 25,
    'max_workers': 4,
    'embedding_timeout': 5,
    'max_text_length': 200,
    'cache_size': 100
}

# Simple in-memory storage for ultra-fast demo
class FastMemoryStore:
    def __init__(self):
        self.data = []
        self.embeddings = {}
    
    def add_record(self, record):
        self.data.append(record)
    
    def search(self, query, top_k=3):
        # Simple keyword-based search for speed
        query_lower = query.lower()
        results = []
        
        for record in self.data:
            score = 0
            text_lower = record['text'].lower()
            
            # Simple scoring based on keyword matches
            if 'revenue' in query_lower and 'revenue' in text_lower:
                score += 2
            if 'customer' in query_lower and 'customer' in text_lower:
                score += 2
            if 'churn' in query_lower and 'churn' in text_lower:
                score += 2
            
            # Add partial matches
            for word in query_lower.split():
                if word in text_lower:
                    score += 1
            
            if score > 0:
                results.append({**record, 'score': score})
        
        # Sort by score and return top results
        results.sort(key=lambda x: x['score'], reverse=True)
        return results[:top_k]
    
    def get_stats(self):
        return {
            'total_records': len(self.data),
            'data_types': len(set(r['data_type'] for r in self.data)),
            'customers': len(set(r['customer'] for r in self.data if r['customer'])),
            'products': len(set(r['product'] for r in self.data if r['product']))
        }

# Ultra-fast data processing
def ultra_fast_process_data():
    """
    Process data with extreme speed optimizations for demo
    """
    store = FastMemoryStore()
    
    files_config = [
        {
            "path": "d:/Dev_space/chatbot/Excel/Uniform_Product revenue data.xlsx",
            "type": "revenue",
            "context": "revenue"
        },
        {
            "path": "d:/Dev_space/chatbot/Excel/uniform_gainsight_data.csv",
            "type": "customer_health", 
            "context": "customer"
        },
        {
            "path": "d:/Dev_space/chatbot/Excel/uniform_salesforce_data.csv",
            "type": "support",
            "context": "support"
        }
    ]
    
    total_processed = 0
    
    for config in files_config:
        try:
            # Load and sample data aggressively
            if config["path"].endswith('.xlsx'):
                df = pd.read_excel(config["path"])
            else:
                df = pd.read_csv(config["path"])
            
            # Take only first N records for speed
            df = df.head(ULTRA_FAST_CONFIG['max_records_per_file'])
            
            # Process each row quickly
            for _, row in df.iterrows():
                # Create simple text representation
                text_parts = []
                customer = ""
                product = ""
                revenue_amount = 0.0
                
                for col, val in row.items():
                    if pd.notna(val):
                        col_lower = col.lower()
                        if 'customer' in col_lower:
                            customer = str(val)
                            text_parts.append(f"Customer: {val}")
                        elif 'product' in col_lower:
                            product = str(val)
                            text_parts.append(f"Product: {val}")
                        elif 'revenue' in col_lower:
                            revenue_amount = float(val) if isinstance(val, (int, float)) else 0.0
                            text_parts.append(f"Revenue: ${val:,}")
                        elif col_lower in ['health score', 'churn risk', 'csat score']:
                            text_parts.append(f"{col}: {val}")
                        else:
                            text_parts.append(f"{col}: {str(val)[:30]}")
                
                # Create record
                text = ". ".join(text_parts[:5])[:ULTRA_FAST_CONFIG['max_text_length']]
                
                record = {
                    'text': text,
                    'data_type': config['type'],
                    'customer': customer,
                    'product': product,
                    'revenue_amount': revenue_amount,
                    'source_file': config["path"].split("/")[-1]
                }
                
                store.add_record(record)
                total_processed += 1
        
        except Exception as e:
            st.error(f"Error processing {config['path']}: {e}")
            continue
    
    return store, total_processed

# Ultra-fast chat function
def ultra_fast_chat(query, store):
    """
    Ultra-fast chat response without external API dependencies
    """
    results = store.search(query, top_k=3)
    
    if not results:
        return "No relevant data found for your query."
    
    # Create simple response based on data
    context_info = []
    for result in results:
        context_info.append(f"- {result['text'][:100]}...")
    
    # Generate simple response
    if 'revenue' in query.lower():
        revenues = [r['revenue_amount'] for r in results if r['revenue_amount'] > 0]
        if revenues:
            total_revenue = sum(revenues)
            response = f"Based on the data, I found revenue information:\\n\\n"
            response += "\\n".join(context_info)
            response += f"\\n\\nTotal revenue from matching records: ${total_revenue:,.2f}"
        else:
            response = "Revenue data found:\\n\\n" + "\\n".join(context_info)
    
    elif 'customer' in query.lower():
        customers = [r['customer'] for r in results if r['customer']]
        response = f"Customer information found:\\n\\n"
        response += "\\n".join(context_info)
        if customers:
            response += f"\\n\\nCustomers mentioned: {', '.join(set(customers))}"
    
    else:
        response = "Based on your query, here's what I found:\\n\\n"
        response += "\\n".join(context_info)
    
    return response

def main():
    st.set_page_config(page_title="⚡ Ultra-Fast Demo", layout="wide")
    
    st.title("⚡ Ultra-Fast Financial AI Demo")
    st.markdown("*Optimized for instant results - Demo Mode*")
    
    # Initialize session state
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'store' not in st.session_state:
        st.session_state.store = None
    if 'processed' not in st.session_state:
        st.session_state.processed = False
    
    # Ultra-fast sidebar
    with st.sidebar:
        st.header("⚡ Ultra-Fast Demo")
        
        if not st.session_state.processed:
            if st.button("🚀 Instant Setup", type="primary"):
                start_time = time.time()
                
                with st.spinner("Ultra-fast processing..."):
                    store, count = ultra_fast_process_data()
                    st.session_state.store = store
                    st.session_state.processed = True
                
                end_time = time.time()
                processing_time = end_time - start_time
                
                st.success(f"⚡ Ready in {processing_time:.1f} seconds!")
                st.metric("Records Processed", count)
                st.metric("Processing Speed", f"{processing_time:.1f}s")
                st.rerun()
        
        if st.session_state.processed and st.session_state.store:
            stats = st.session_state.store.get_stats()
            st.success("🟢 Ultra-Fast Mode Active")
            
            st.subheader("📊 Quick Stats")
            st.metric("Total Records", stats['total_records'])
            st.metric("Data Types", stats['data_types'])
            st.metric("Customers", stats['customers'])
            st.metric("Products", stats['products'])
            
            st.subheader("⚡ Quick Queries")
            quick_queries = [
                "Show me revenue data",
                "Which customers are mentioned?", 
                "What about product performance?",
                "Any customer health information?"
            ]
            
            for query in quick_queries:
                if st.button(query, key=f"quick_{query}"):
                    st.session_state.messages.append({"role": "user", "content": query})
                    response = ultra_fast_chat(query, st.session_state.store)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    st.rerun()
    
    # Main chat interface
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Display messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # Chat input
        if prompt := st.chat_input("Ask about your data..."):
            if not st.session_state.processed:
                st.error("Please run Instant Setup first!")
            else:
                st.session_state.messages.append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.markdown(prompt)
                
                with st.chat_message("assistant"):
                    with st.spinner("⚡ Ultra-fast analysis..."):
                        response = ultra_fast_chat(prompt, st.session_state.store)
                        st.markdown(response)
                        st.session_state.messages.append({"role": "assistant", "content": response})
    
    with col2:
        st.subheader("⚡ Performance Info")
        
        st.info("""
        **Ultra-Fast Demo Mode:**
        - Processes limited sample data
        - In-memory storage only
        - Keyword-based search
        - Instant responses
        
        **For Production:**
        - Use fast_streamlit_app.py
        - Full vector embeddings
        - Complete dataset processing
        - Advanced AI responses
        """)
        
        if st.session_state.processed:
            st.success("✅ Demo data loaded")
            st.write("*Ready for instant queries!*")

if __name__ == "__main__":
    main()