from langchain_community.tools import tool


@tool  # this decorator tells that it is a tool, not a normal function. so this tool can be used by LLMs
def Multiply(a: int, b: int) -> int:
    """Multiplies two numbers"""
    return a * b


result = Multiply.invoke({"a": 5, "b": 5})
print(result)

print(Multiply.name)
print(Multiply.description)
print(Multiply.args)


@tool
def Add(a: int, b: int) -> int:
    """Adds two numbers"""
    return a + b


# Making toolkit -> bundling same kind of tools for reusability
class MathToolkit:
    def get_tools(self):
        return [Add, Multiply]


toolkit = MathToolkit()
tools = toolkit.get_tools()

for single_tool in tools:
    print(single_tool.name)
