from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()

embeddings = HuggingFaceEndpointEmbeddings(
    model="sentence-transformers/all-MiniLM-L6-v2",
    task="feature-extraction"
)

document =[
"Virat Kohli is an Indian cricketer known for his aggressive batting and leadership." ,
"MS Dhoni is a former Indian captain famous for his calm demeanor and fini hing skills." ,
"Sachin Tendulkar, also known as the 'God of Cricket',holds many batting ecords.",
"Rohit Sharma is known for his elegant batting and record-breaking double centuries." ,
"Jasprit Bumrah is an Indian fast bowler known for his unorthodox action and yorkers."]

query = "Tell me about Virat Kohli"

doc_embedding = embeddings.embed_documents(document)
query_embedding = embeddings.embed_query(query)

similarity_score = cosine_similarity([query_embedding],doc_embedding)
print(similarity_score)