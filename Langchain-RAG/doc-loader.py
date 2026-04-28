from langchain_community.document_loaders import TextLoader
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(repo_id="deepseek-ai/DeepSeek-V4-Pro", task="text-generation")
model = ChatHuggingFace(llm=llm)
parser = StrOutputParser()

prompt = PromptTemplate(
    template="Explain a short summary for this poem:{poem}", input_variables=["poem"]
)

# loading document
loader = TextLoader("cricket.txt")
docs = loader.load()

chain = prompt | model | parser
result = chain.invoke({"poem": docs[0].page_content})

print(result)

# print(docs)
# print(type(docs))
# print(len(docs))
# print((docs[0]))
# print((docs[0]).page_content)
