import llama_cpp
from qdrant_client import QdrantClient


collection_name = 'source_data2'
embedding_llm = llama_cpp.Llama(
  model_path="models/llama-2-7b-chat.Q2_K.gguf",
  embedding=True,
  verbose=False
)

client = QdrantClient(path="embeddings")

"""
Below finds related context from input based on the query.
In order to do so, it uses Semantic Similarity.
"""
query = "Who won the game?"
query_vector = embedding_llm.create_embedding(query)['data'][0]['embedding']
related_context_from_inputs = client.search(
  collection_name=collection_name,
  query_vector=query_vector,
  limit=3
)


prompt_template = """You are a helpful assistant who answers questions using only the provided context.
If you don't know the answer, simply state that you don't know.

{context}

Question: {question}"""

llm = llama_cpp.Llama(
  model_path="./models/llama-2-7b-chat.Q2_K.gguf",
  verbose=False
)

stream = llm.create_chat_completion(
    messages=[
        {"role": "user", "content": prompt_template.format(
            context="\n\n".join([row.payload['text'] for row in related_context_from_inputs]),
            question=query
        )}
    ],
    stream=True
)

for chunk in stream:
    print(chunk['choices'][0]['delta'].get('content', ''), end='')
