from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

load_dotenv()

model=ChatGoogleGenerativeAI(model='gemini-2.5-flash')
chat_history=[
    SystemMessage(content='you are a helpful assistant')
]
while True:
    prompt=input('You: ')
    chat_history.append(HumanMessage(content=prompt))
    if prompt=='exit':
        break

    output=model.invoke(chat_history)
    chat_history.append(AIMessage(content=output.content))
    print("AI: ",output.content)

print(chat_history)