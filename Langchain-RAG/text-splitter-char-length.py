from langchain_text_splitters import CharacterTextSplitter

text = """
Space exploration has led to incredible scientific discoveries. From landing
exploring mars, humanity continues to push the boundaries of wha€s possible
planet.
1
These missions have not only expanded our knowledge of the universe but have
on the Moon to
beyond our
also
contributed to advancements in technology here on Earth. Satellite cor•unications, GPS, and
even certain medical imaging techniques trace their roots back to innovations driven by space programs
"""
splitter = CharacterTextSplitter(separator=" ", chunk_size=100, chunk_overlap=0)

result = splitter.split_text(text)
print(result)
