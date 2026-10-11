from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage  

from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

llm = HuggingFaceEndpoint(
    model="Qwen/Qwen2.5-7B-Instruct",
    provider="together",
    task="text-generation"  # try "together", "novita", or "fireworks-ai" — whichever shows as available on the model page
)

chatModel=ChatHuggingFace(llm=llm)

history=[
    SystemMessage(content="You are a helpful assistant.")]

while True:
    user_input = input("User: ")
    if user_input.lower() in ["exit", "quit"]:
        print("Exiting the chat.")
        break
    history.append(HumanMessage(content=user_input))
    
    response = chatModel.invoke(history)
    print("AI: ", response.content)
    
    history.append(AIMessage(content=response.content))



