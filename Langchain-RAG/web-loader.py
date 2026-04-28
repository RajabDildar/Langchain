from langchain_community.document_loaders import WebBaseLoader
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

url = "https://en.wikipedia.org/wiki/Elon_Musk"
loader = WebBaseLoader(
    url
)  # we can also pass list of urls to load data from multiple websites

docs = loader.load()  # loads all files at once

chain = prompt | model | parser
result = chain.invoke({"text": docs[0].page_content})

print(result)

# for javascript heavy websites, use seleniumurlloader.
# this loader is good for static sites
