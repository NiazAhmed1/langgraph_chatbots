from langgraph.graph import StateGraph, START,END
from typing import Type, Annotated
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os


load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")


class chatbot_state(dict):
    user_input:str
    response : str
    
    
#Load LLM
def chatbot_node(state:chatbot_state):
    llm = ChatGroq(
    model="gemma2-9b-it",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    )
    response = llm.invoke(state['user_input'])
    state["response"]=response.content
    return state



#create graph
graph = StateGraph(chatbot_state)
graph.add_node("chatbot",chatbot_node)
graph.add_edge(START,"chatbot")
graph.add_edge("chatbot",END)
app=graph.compile()



# 4. Run chatbot loop
if __name__ == "__main__":
    print("AI Chatbot (type 'exit' to quit)")
    while True:
        user_message = input("You: ")
        if user_message.lower() in ["quit", "exit"]:
            print("Chat ended.")
            break

        result = app.invoke({"user_input": user_message})
        print("Bot:", result["response"])    
