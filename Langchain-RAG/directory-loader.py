from langchain_community.document_loaders import UnstructuredPDFLoader, DirectoryLoader

loader = DirectoryLoader(
    path="books", glob="*.pdf", loader_cls=UnstructuredPDFLoader
)  # loading all pdf files

# docs = loader.load()  # loads all files at once
docs = loader.lazy_load()  # Use stream processing. loads a document, process it, release its ram usage and move to the next document. Used when we have large number of files to load.

for doc in docs:
    print(doc.page_content)
