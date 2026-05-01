from langchain_community.tools import DuckDuckGoSearchRun, WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.tools.arxiv.tool import ArxivQueryRun
from langchain_community.utilities import ArxivAPIWrapper

# Initialize Wikipedia Wrapper
wikipedia_api_wrapper = WikipediaAPIWrapper(top_k_results=2, doc_content_chars_max=2000)
wikipedia_tool = WikipediaQueryRun(api_wrapper=wikipedia_api_wrapper)
wikipedia_tool.name = "wikipedia"

# Initialize DuckDuckGo Search
duckduckgo_tool = DuckDuckGoSearchRun()
duckduckgo_tool.name = "duckduckgo_search"

# Initialize Arxiv Search
arxiv_api_wrapper = ArxivAPIWrapper(top_k_results=2, load_max_docs=2)
arxiv_tool = ArxivQueryRun(api_wrapper=arxiv_api_wrapper)
arxiv_tool.name = "arxiv_search"

# Export the tools as a list
tools = [duckduckgo_tool, wikipedia_tool, arxiv_tool]
