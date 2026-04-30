from langchain_community.tools import DuckDuckGoSearchRun

search_tool = DuckDuckGoSearchRun()
result = search_tool.invoke("Elon Musk")
print(result)

# Tools are also runnables in langchain
