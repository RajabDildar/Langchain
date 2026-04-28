from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(repo_id="deepseek-ai/DeepSeek-V4-Pro", task="text-generation")
model = ChatHuggingFace(llm=llm)
parser = StrOutputParser()

prompt = PromptTemplate(
    template="create a short summary for this document:{text}", input_variables=["text"]
)

# pdf loading
loader = PyPDFLoader("dl-curriculum.pdf")
docs = loader.load()

# print(docs)
# print(len(docs))

chain = prompt | model | parser
result = chain.invoke({"text": docs[0].page_content})

print(result)
