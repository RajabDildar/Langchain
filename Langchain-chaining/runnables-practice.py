from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import (
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
)
from dotenv import load_dotenv

load_dotenv()

prompt = PromptTemplate(template="write a joke on {topic}", input_variables=["topic"])

llm = HuggingFaceEndpoint(repo_id="google/gemma-4-31B-it", task="text-generation")
model = ChatHuggingFace(llm=llm)
parser = StrOutputParser()

capitalize = RunnableLambda(lambda x: x.upper())


def word_count(text):
    return len(text.split())


word_count_runnable = RunnableLambda(word_count)

chain1 = prompt | model | parser | capitalize

chain2 = RunnableParallel(
    {"Joke": RunnablePassthrough(), "word_count": word_count_runnable}
)

chain = chain1 | chain2
result = chain.invoke({"topic": "cricket"})
print(result)

chain.get_graph().print_ascii()
