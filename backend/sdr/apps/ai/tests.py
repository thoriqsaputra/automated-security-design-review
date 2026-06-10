from sdr.apps.ai.client import chat_completion, get_embedding

# --- 1. Test Chat Completion ---
print("Testing Chat Completion...")
chat_response = chat_completion(
    messages=[
        {"role": "system", "content": "You are a helpful security assistant. Answer in one sentence."},
        {"role": "user", "content": "What is CSRF?"}
    ],
    component="orchestrator",
)

if chat_response.error:
    print(f"❌ Chat Error: {chat_response.error}")
else:
    print(f"✅ Chat Success!")
    print(f"Response: {chat_response.content}")
    print(f"Usage: {chat_response.usage}")

print("-" * 40)

# --- 2. Test Embeddings ---
print("Testing Embeddings...")
embedding_vector = get_embedding("How to secure a React application?")

if not embedding_vector:
    print("❌ Embedding Error: Returned empty list")
else:
    print(f"✅ Embedding Success!")
    print(f"Vector Dimensions: {len(embedding_vector)}")
    print(f"First 5 floats: {embedding_vector[:5]}")
