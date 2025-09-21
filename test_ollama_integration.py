import ollama

def test_ollama_integration():
    try:
        # Test chat completion
        response = ollama.chat(model='llama3.2:3b', messages=[
            {
                'role': 'user',
                'content': 'Analyze this financial data: Revenue $100M, Expenses $80M, Net Income $20M. What are the key insights?',
            },
        ])
        
        print("✅ Ollama integration successful!")
        print("\n📊 Financial Analysis Response:")
        print("-" * 50)
        print(response['message']['content'])
        
        return True
        
    except Exception as e:
        print(f"❌ Ollama integration failed: {e}")
        return False

if __name__ == "__main__":
    test_ollama_integration()