import sys
sys.path.append('src')
from src.retrieval import RetrievalSystem
from src.embeddings import EmbeddingsManager

# Initialize components
retrieval_system = RetrievalSystem()
embeddings_manager = EmbeddingsManager()

# Load document
print("Loading documents...")
embeddings_manager.load_documents_from_directory('documents')

# Test grace period query specifically
query = 'What is the grace period for premium payment under the National Parivar Mediclaim Plus Policy?'
print(f"\nTesting query: {query}")

chunks = retrieval_system.retrieve_relevant_chunks(query, embeddings_manager, k=10)

print(f'\nFound {len(chunks)} chunks for grace period query:')
for i, chunk in enumerate(chunks[:5]):
    print(f'\n--- Chunk {i+1} (confidence: {chunk["confidence"]:.3f}) ---')
    content = chunk['content'][:400] + '...' if len(chunk['content']) > 400 else chunk['content']
    print(content)
    
    # Check for different grace period mentions
    content_lower = chunk['content'].lower()
    if '30' in chunk['content'] or 'thirty' in content_lower:
        print('*** CONTAINS 30/THIRTY ***')
    if '15' in chunk['content'] or 'fifteen' in content_lower:
        print('*** CONTAINS 15/FIFTEEN ***')
    if 'grace' in content_lower and 'period' in content_lower:
        print('*** CONTAINS GRACE PERIOD ***')

print("\n" + "="*50)
print("SEARCHING FOR ALL GRACE PERIOD MENTIONS:")

# Search for all chunks containing grace period
all_chunks = retrieval_system.retrieve_relevant_chunks('grace period premium payment', embeddings_manager, k=20)
grace_chunks = [chunk for chunk in all_chunks if 'grace' in chunk['content'].lower()]

for i, chunk in enumerate(grace_chunks[:3]):
    print(f'\n--- Grace Chunk {i+1} ---')
    print(chunk['content'][:500])
