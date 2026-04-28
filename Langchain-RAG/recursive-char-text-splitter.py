from langchain_text_splitters import Language, RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("dl-curriculum.pdf")
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=0)

chunks = splitter.split_documents(docs)

print(chunks)
# print(len(chunks))
# print(chunks[0].page_content)

# this algorithm can also be used for splitting of documents like markdown, python code or other languages.
splitter2 = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON, chunk_size=175, chunk_overlap=0
)
