
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

llm = ChatGroq(model="llama-3.3-70b-versatile",temperature=0.7)

result = llm.invoke("Write a poem about the beauty of nature.")

print("Poem : \n\n", result.content)