import ollama
import json

def debug_ollama():
    print("🔍 Debugging Ollama connection...")
    
    try:
        # Test 1: Basic connection
        print("\n1. Testing basic connection...")
        response = ollama.list()
        print(f"Response type: {type(response)}")
        print(f"Response: {response}")
        
        # Test 2: Check if llama3.2:3b exists
        print("\n2. Checking for llama3.2:3b model...")
        try:
            model_info = ollama.show('llama3.2:3b')
            print("✅ llama3.2:3b model found!")
            print(f"Model info: {model_info.get('modelfile', 'No modelfile info')}")
        except Exception as e:
            print(f"❌ Model check failed: {e}")
        
        # Test 3: Try a simple chat
        print("\n3. Testing chat functionality...")
        try:
            chat_response = ollama.chat(model='llama3.2:3b', messages=[
                {'role': 'user', 'content': 'Hello, just testing connection'}
            ])
            print("✅ Chat test successful!")
            print(f"Response: {chat_response['message']['content'][:100]}...")
        except Exception as e:
            print(f"❌ Chat test failed: {e}")
            
    except Exception as e:
        print(f"❌ Overall connection failed: {e}")

if __name__ == "__main__":
    debug_ollama()