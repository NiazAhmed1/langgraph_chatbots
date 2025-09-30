import os
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq


# Load API key from .env 
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")


# Number of messages to keep in memory
MAX_MEMORY_MESSAGES = 10  


# Define chatbot state 
class ChatState(dict):
    messages: list  # conversation history
    user_input: str
    ai_response: str


# Define chatbot function
def chatbot_node(state: ChatState):
    llm = ChatGroq(model="llama-3.1-8b-instant", api_key=groq_api_key)

    
    if "messages" not in state:
        state["messages"] = []


    # Add latest user message
    state["messages"].append({"role": "user", "content": state["user_input"]})


    # Keep only the last N messages
    if len(state["messages"]) > MAX_MEMORY_MESSAGES:
        state["messages"] = state["messages"][-MAX_MEMORY_MESSAGES:]


    # Pass trimmed history to model
    response = llm.invoke(state["messages"])

    # Save AI response
    state["messages"].append({"role": "assistant", "content": response.content})


    # Again trim if exceeded
    if len(state["messages"]) > MAX_MEMORY_MESSAGES:
        state["messages"] = state["messages"][-MAX_MEMORY_MESSAGES:]

    state["ai_response"] = response.content
    return state


# Create graph
graph = StateGraph(ChatState)

graph.add_node("chatbot", chatbot_node)
graph.set_entry_point("chatbot")
graph.add_edge("chatbot", END)

app = graph.compile()


# 4. Run chatbot
if __name__ == "__main__":
    
    print("🤖 AI Chatbot with Sliding Window Memory (type 'exit' to quit)")
    memory = []  # store conversation history

    while True:
        user_message = input("You: ")
        if user_message.lower() in ["quit", "exit"]:
            print("Chat ended.")
            break

        result = app.invoke({
            "user_input": user_message,
            "messages": memory
        })

        memory = result["messages"]  # update memory
        print("Bot:", result["ai_response"])
