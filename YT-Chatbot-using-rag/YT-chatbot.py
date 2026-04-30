from langchain_huggingface import (
    HuggingFaceEndpoint,
    HuggingFaceEmbeddings,
    ChatHuggingFace,
)
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import YoutubeLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from langchain_core.runnables import (
    RunnableParallel,
    RunnablePassthrough,
    RunnableLambda,
)
from dotenv import load_dotenv

load_dotenv()

# prompt templates
prompt = PromptTemplate(
    template="You are a helpful assistant. Answer the questions only from the provided video context(Its from Youtube video transcripts). If context is insufficient, mention user that it is not discussed in the video or point out any related thing discussed in the video.\n context:{context} \n query:{query}",
    input_variables=["context", "query"],
)

query = str(input("Write your query here: "))

# parser
parser = StrOutputParser()

# models
llm = HuggingFaceEndpoint(repo_id="google/gemma-4-31B-it", task="text-generation")
model = ChatHuggingFace(llm=llm)

# indexing
# doc loading
loader = YoutubeLoader.from_youtube_url(
    "https://www.youtube.com/watch?v=HAnw168huqA", add_video_info=False
)

docs = loader.load()
# docs = docs[0].page_content
# print(docs) # give list of documents

# text splitting
splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=10)
chunks = splitter.split_documents(docs)
# print(chunks)  # return list of chunks
# print(type(chunks))

# making vector store with chromadb
vector_store = Chroma(
    embedding_function=HuggingFaceEmbeddings(), collection_name="YT-videos"
)

vector_store.add_documents(documents=chunks)

# retrieving
retriever = MultiQueryRetriever.from_llm(
    retriever=vector_store.as_retriever(), llm=model
)

# result = retriever.invoke(query)
# print(result)


def formate_context(docs):
    res = ""
    for doc in docs:
        res += doc.page_content
        res += "\n"
    return res


# generating answers
context_chain = RunnableParallel(
    {
        "query": RunnablePassthrough(),
        "context": retriever | RunnableLambda(formate_context),
    }
)

Generator_chain = prompt | model | parser
chain = context_chain | Generator_chain
result = chain.invoke(query)
print(result)

chain.get_graph().print_ascii()


"""
prompts
output parsing
models
indexing -> doc loading, text splitting, vector store, retrieving
memory -> optional

retreive -> algorithms for retreiving mmr, mqr, ccr, e.t.c
augment -> context + query -> prompt
generation -> LLM  
"""

"""
input -> youtube video url + query

steps:
doc_loading -> yt transcript
text splitting -> split yt transcript based on topic. using different algorithms
vector store -> provide embedding model, 
retreiving related documents -> embed query, compare and retrieve related documents

now we have context and user query. make system prompt (prompt template) -> augmenting
send to llm/model
parse output and print
"""
