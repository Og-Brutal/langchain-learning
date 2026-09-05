from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from dotenv import load_dotenv
from sympy import true
from sympy import true

load_dotenv()  # Load environment variables from .env file

prompt_template = ChatPromptTemplate([
    ("system", "You are a helpful assistant."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{querry}"),
])

history=[]

with open("history.txt", "r") as file:
    for line in file:
        if line.startswith("YOU:"):
            user_message = line[len("YOU:"):].strip()
            history.append(HumanMessage(content=user_message))
        elif line.startswith("AI:"):
            ai_message = line[len("AI:"):].strip()
            history.append(AIMessage(content=ai_message))


llm=ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite", temperature=0.7)

while true:
    user_input = input("YOU: ")
    if user_input.lower() == "exit":
        break
    history.append(HumanMessage(content=user_input))
    prompt = prompt_template.invoke({"history": history, "querry": user_input})
    result=llm.invoke(prompt)
    history.append(AIMessage(content=result.content[0]['text']))
    print("AI: ", result.content[0]['text'])


with open("history.txt", "w") as file:
    for message in history:
        if isinstance(message, HumanMessage):
            file.write(f"YOU: {message.content}\n")
        elif isinstance(message, AIMessage):
            file.write(f"AI: {message.content}\n")

