import sys
sys.path.append('d:/Dev_space/chatbot')

from enhanced_streamlit_app import connect_to_milvus, create_milvus_collection, insert_enhanced_data_to_milvus, search_enhanced_milvus, enhanced_chat_with_bot
import time

print("🚀 Enhanced RAG System Demonstration")
print("=" * 50)

# Connect to Milvus
print("\n1. Connecting to Milvus...")
if connect_to_milvus():
    print("✅ Connected successfully!")
    
    # Setup collection
    print("\n2. Setting up enhanced collection...")
    collection = create_milvus_collection()
    if collection:
        print("✅ Collection ready!")
        
        # Insert enhanced data
        print("\n3. Inserting enhanced data with business intelligence...")
        count = insert_enhanced_data_to_milvus(collection)
        print(f"✅ Inserted {count} enhanced records with metadata!")
        
        # Test enhanced queries
        print("\n4. Testing Enhanced Query Capabilities:")
        print("=" * 50)
        
        test_queries = [
            "What is the total revenue by customer?",
            "Which customers are at high churn risk with significant revenue?",
            "Show me the Customer 360 view for Alpha01",
            "Revenue at risk from unhealthy customers",
            "API usage patterns and revenue correlation"
        ]
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n🔍 Query {i}: {query}")
            print("-" * 60)
            
            # Test the enhanced search
            results = search_enhanced_milvus(collection, query, top_k=3)
            print(f"📊 Found {len(results)} enhanced results")
            
            for j, result in enumerate(results[:2], 1):
                print(f"  Result {j}:")
                print(f"    • Data Type: {result['data_type']}")
                print(f"    • Customer: {result['customer']}")
                print(f"    • Product: {result['product']}")
                print(f"    • Revenue: ${result['revenue_amount']:,.0f}")
                print(f"    • Enhanced Score: {result['score']:.3f}")
                print(f"    • Context: {result['context_type']}")
            
            # Test the enhanced chat
            print(f"\n🤖 Enhanced AI Response:")
            response = enhanced_chat_with_bot(query, collection)
            # Show first 200 characters of response
            print(f"    {response[:200]}...")
            print()
            
        print("\n🎉 Enhanced RAG System Demonstration Complete!")
        print("\nKey Improvements Demonstrated:")
        print("✅ Context-aware embeddings with business terms")
        print("✅ Revenue-weighted scoring and intelligent ranking")
        print("✅ Cross-dataset correlation and metadata enrichment")
        print("✅ Business intelligence integration")
        print("✅ Enhanced query processing with automatic context detection")
        
    else:
        print("❌ Failed to create collection")
else:
    print("❌ Failed to connect to Milvus")

print(f"\n🌐 Enhanced Streamlit app running at: http://localhost:8501")
print("📋 Try these enhanced queries in the web interface:")
for query in test_queries:
    print(f"   • {query}")