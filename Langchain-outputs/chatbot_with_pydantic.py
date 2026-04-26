from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from typing import Optional
from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()

llm = HuggingFaceEndpoint(repo_id="zai-org/GLM-5.1", task="text-generation")
model = ChatHuggingFace(llm=llm)


class Review(BaseModel):
    key_themes: list[str] = Field(
        description="List all key themes discussed in review in a list"
    )
    summary: str = Field(description="A brief summary of the review")
    sentimant: str = Field(description="either review is positive,negative or neutral")
    pros: Optional[list] = Field(description="List all pros in the list")
    cons: Optional[list] = Field(description="List all cons in the list")


structured_model = model.with_structured_output(Review)

prompt = """I recently upgraded to the Samsung Galaxy S24 Ultra, and I must say, it's an absolute powerhouse! The Snapdragon 8 Gen 3 processor makes everything lightning fast—whether I'm gaming, multitasking, or editing photos. The SeeømAh battery easily 
lasts a full day even with heavy use, and the 45W fast charging is a lifesaver.
The S-Pen integration is a great touch for note-taking and quick sketches, though I don't use it often. What really blew me
away is the 2eemp camera—the night mode is stunning, capturing crisp, vibrant images even in low light. Zooming up to IOX
actually works well for distant objects, but anything beyond 30x loses quality.
However, the weight and size make it a bit uncomfortable for one-handed use. Also, Samsung's One UI still comes with
bloatware—why do I need five different Samsung apps for things Google already provides? The $1,30 price tag is also a hard
pill to swallow.
Pros :
Insanely powerful processor (great for gaming and productivity)
Stunning 20MP camera with incredible zoom capabilities
Long battery life with fast charging
S-Pen support is unique and useful
Cons :
Bulky and heavy—not great for one-handed use
Bloatware still exists in One UI
Expensive compared to competitors"""

result = structured_model.invoke(prompt)

print(result)

# pydantic function calling may not work with hugging face or langchain. but will work the same way with other models like openai
