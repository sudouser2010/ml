import uuid
import llama_cpp

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from langchain_text_splitters import RecursiveCharacterTextSplitter

from utils import chunk

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

embedding_model = llama_cpp.Llama(
    model_path="models/llama-2-7b-chat.Q2_K.gguf",
    embedding=True,
    verbose=False
)

# for each document get vector representations
document_vectors = embedding_model.create_embedding([item.page_content for item in documents])

# create a data structure which pairs each document chuck to its corresponding vector
documents_embeddings_pair = [
    (document, vector['embedding'])
    for document, vector in zip(documents, document_vectors['data'])
]

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
        vector=vector,
        payload={
            'text': document.page_content
        }
    )
    for document, vector in document_vectors
]
operation_info = client.upsert(
    collection_name=collection_name,
    wait=True,
    points=points
)
