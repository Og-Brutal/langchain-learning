from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite", temperature=0.7)

history=[
    SystemMessage(content="You are a helpful assistant that teaches programming."),
    HumanMessage(content="Can you explain the concept of recursion in programming?"),
]


result=llm.invoke(history)

history.append(AIMessage(content=result.content[0]['text']))

print("History after first interaction:", history)