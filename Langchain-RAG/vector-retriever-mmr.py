from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from dotenv import load_dotenv

load_dotenv()

docs = [
    (Document(page_content=("LangChain makes it easy to work with LLMs."))),
    (Document(page_content=("LangChain is used to build based applications."))),
    (
        Document(
            page_content=("Chroma is usecl tio store and search document embeddings.")
        )
    ),
    (Document(page_content=("are vector of text. - "))),
    (
        Document(
            page_content=(
                "W.R helps you get diverse results when doing similarity search."
            )
        )
    ),
    (Document(page_content=("LangChain supports Chroma, FAISS, Pinecone, and more. "))),
]

vector_store = FAISS.from_documents(embedding=HuggingFaceEmbeddings(), documents=docs)

retriever = vector_store.as_retriever(
    search_type="mmr",  # thois enables mmr
    search_kwargs={
        "k": 3,
        "lambda_mult": 0.5,  # range from 0 to 1. 0 means most diverse results. 1 is like semantic search
    },  # k for top results, lambda_mult = relevance diversity balance
)

query = "what is langchain?"
result = retriever.invoke(query)

for doc in enumerate(result):
    print(doc)
