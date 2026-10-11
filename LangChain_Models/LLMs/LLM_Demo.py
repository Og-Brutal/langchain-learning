
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

llm = ChatGroq(model="llama-3.3-70b-versatile",temperature=0.7)

result = llm.invoke("Write a poem about the beauty of nature.")

print("Poem : \n\n", result.content)



# The Temperature argument controls the randomness of output like if you ask a question and in next prompt you ask the same question again it will give you different answer because of the randomness. The higher the temperature the more random the output will be and vice versa.