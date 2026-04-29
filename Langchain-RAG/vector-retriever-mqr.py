from langchain_huggingface import (
    HuggingFaceEmbeddings,
    HuggingFaceEndpoint,
    ChatHuggingFace,
)
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
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

# setting an llm
llm = HuggingFaceEndpoint(repo_id="google/gemma-4-31B-it", task="conversational")
model = ChatHuggingFace(llm=llm)

retriever = MultiQueryRetriever.from_llm(
    retriever=vector_store.as_retriever(), llm=model
)

query = "What is LangChain?"
result = retriever.invoke(query)
print(result)
