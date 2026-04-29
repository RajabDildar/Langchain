from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from dotenv import load_dotenv

load_dotenv()

doc1 = Document(
    page_content="Virat Kohli is one of the most successful and consistent batsmen in IPL history. Known for his aggressive batting style.",
    metadata={"team": "Royal Challengers Bangalore"},
)
doc2 = Document(
    page_content="Rohit Sharma is the most successful captain in IPL history, leading Mumbai Indians to five titles. He's known for his calm",
    metadata={"team": "Mumbai Indians"},
)
doc3 = Document(
    page_content="MS Dhoni, famously known as Captain Cool, has led Chennai Super Kings to multiple IPL titles. His finishing skills, wicke",
    metadata={"team": "Chennai Super Kings"},
)
doc4 = Document(
    page_content="Jasprit Bumrah is considered one of the best fast bowlers in T20 cricket. Playing for Mumbai Indians, he is known for his",
    metadata={"team": "Mumbai Indians"},
)
doc5 = Document(
    page_content="Ravindra Jadeja is a dynamic all-rounder who contributes with both bat and ball. Representing Chennai Super Kings, his qu",
    metadata={"team": "Chennai Super Kings"},
)
docs = [doc1, doc2, doc3, doc4, doc5]

vector_store = Chroma(
    embedding_function=HuggingFaceEmbeddings(),
    persist_directory="chroma_db",
    collection_name="Players",
)

vector_store.add_documents(documents=docs)
data = vector_store.get()
# print(data)

# vectors have capability for semantic search. but still we make retrievers because retrievers are runnables, and can be used in complex workflows seamlessly. also they allow various other search strategies that are not available in this function.
result = vector_store.similarity_search(
    query="Who is the bowler among all players?",
    k=1,  # k defines how many similar documents to fetch
    # we can also write some filter here on metadata
    # filter={"team": "Chennai Super Kings"},
)
print(result)
# we can also delete and update documents
