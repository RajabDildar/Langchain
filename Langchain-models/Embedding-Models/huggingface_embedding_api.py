from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpointEmbeddings

load_dotenv()

embeddings = HuggingFaceEndpointEmbeddings(
    model="sentence-transformers/all-MiniLM-L6-v2",
    task="feature-extraction"
)

# Embed your query
query_vector = embeddings.embed_query("How do I use Hugging Face?")

# Embed your documents
doc_vectors = embeddings.embed_documents(["Doc 1 text", "Doc 2 text"])

# print(query_vector)
print(doc_vectors)