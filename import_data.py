import uuid
import llama_cpp

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from langchain_text_splitters import RecursiveCharacterTextSplitter

from itertools import islice


def chunk(arr_range, chunk_size):
    arr_range = iter(arr_range)
    return iter(lambda: list(islice(arr_range, chunk_size)), [])


# obtain text source
data_source = 'inputs/source.txt'
collection_name = 'source_data2'
f = open(data_source, 'r')
text = f.read()
f.close()


# split text
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50,
    length_function=len,
    is_separator_regex=False,
)
documents = text_splitter.create_documents([text])


llm = llama_cpp.Llama(
  model_path="models/llama-2-7b-chat.Q2_K.gguf",
  embedding=True,
  verbose=False
)

batch_size = 100
documents_embeddings = []
batches = list(chunk(documents, batch_size))

for batch in batches:
    embeddings = llm.create_embedding([item.page_content for item in batch])
    documents_embeddings.extend(
        [
            (document, embeddings['embedding'])
            for document, embeddings in zip(batch, embeddings['data'])
        ]
    )


# Init client and create collection
client = QdrantClient(path="embeddings")
client.create_collection(
    collection_name=collection_name,
    vectors_config=VectorParams(size=4096, distance=Distance.COSINE),
)


# Store documents as points
points = [
    PointStruct(
        id=str(uuid.uuid4()),
        vector=embeddings,
        payload={
        "text": doc.page_content
        }
    )
    for doc, embeddings in documents_embeddings
]
operation_info = client.upsert(
    collection_name=collection_name,
    wait=True,
    points=points
)
