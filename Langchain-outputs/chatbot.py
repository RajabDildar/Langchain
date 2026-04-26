from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from typing import TypedDict,Annotated
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="zai-org/GLM-5.1",
    task="text-generation"
)
model = ChatHuggingFace(llm=llm)

class Review(TypedDict):
    summary:Annotated[str,"A brief summary of the review"]
    sentimant:Annotated[str,"either review is positive,negative or neutral"]

structured_model = model.with_structured_output(Review)

result = structured_model.invoke("There is  a slight problem with connecting it with samsung phone through bluetooth. I don't know what to do. I approached the seller but got no response. very disappointing!")

print(result)