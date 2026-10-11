from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env filec

from langchain_google_genai import ChatGoogleGenerativeAI
chat_prompt = ChatPromptTemplate([
    ("system", "you are a helpful assistant that teaches {subject}"),
    ("human","Tell me about {topic}"),
])


prompt=chat_prompt.invoke({"subject":"programming", "topic":"recursion"})

llm=ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite", temperature=0.7)
result=llm.invoke(prompt)



print("Result : \n\n", result.content[0]['text'])