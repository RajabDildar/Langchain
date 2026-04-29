from langchain_community.retrievers import WikipediaRetriever

retriever = WikipediaRetriever(lang="en", top_k_results=2)

docs = retriever.invoke("Albert Einstein")
print(docs[0].page_content)
print(len(docs))

# retriever is like search engine. it fetches the relevant documents based on query using optimize algorithms.
