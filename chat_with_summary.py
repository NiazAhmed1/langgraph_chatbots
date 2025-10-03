import os
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq


# Load API key from .env
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")


MAX_MEMORY_MESSAGES = 10   # keep last 10 messages
SUMMARY_TRIGGER = 20       # when history exceeds (more than 10 messages), start summarizing


# 1. Define state
class ChatState(dict):
    messages: list
    summary: str
    user_input: str
    ai_response: str

# 2. Summary generator for messages 
def summarize_history(messages, current_summary, llm):
    """Summarize conversation history into a compact note."""
    summary_prompt = [
        {"role": "system", "content": "You are a helpful assistant that summarizes chat history."},
        {"role": "user", "content": f"Current summary:\n{current_summary}\n\n"
                                    f"New messages:\n{messages}\n\n"
                                    "Update the summary to include important facts and context."}
    ]
    summary_response = llm.invoke(summary_prompt)
    return summary_response.content


# 3. Chatbot node
def chatbot_node(state: ChatState):
    llm = ChatGroq(model="llama-3.1-8b-instant", api_key=groq_api_key)

    if "messages" not in state:
        state["messages"] = []
    if "summary" not in state:
        state["summary"] = ""

    # Add user message
    state["messages"].append({"role": "user", "content": state["user_input"]})

    # If too many messages, summarize
    if len(state["messages"]) > SUMMARY_TRIGGER:
        state["summary"] = summarize_history(state["messages"], state["summary"], llm)
        
        # Reset memory to summary + last N messages
        state["messages"] = [
            {"role": "system", "content": f"Conversation summary so far: {state['summary']}"}
        ] + state["messages"][-MAX_MEMORY_MESSAGES:]

    # Pass trimmed history to model
    response = llm.invoke(state["messages"])

    # Save AI response
    state["messages"].append({"role": "assistant", "content": response.content})
    state["ai_response"] = response.content
    return state

# 4. Build LangGraph
graph = StateGraph(ChatState)
graph.add_node("chatbot", chatbot_node)
graph.set_entry_point("chatbot")
graph.add_edge("chatbot", END)

app = graph.compile()


# 5. Run chatbot loop
if __name__ == "__main__":
    print("AI Chatbot with Summary Memory (type 'exit' to quit)")
    memory = []
    summary = ""

    while True:
        user_message = input("You: ")
        if user_message.lower() in ["quit", "exit"]:
            print("Chat ended.")
            break

        result = app.invoke({
            "user_input": user_message,
            "messages": memory,
            "summary": summary
        })

        memory = result["messages"]
        summary = result["summary"]  # keep updated summary
        print("Bot:", result["ai_response"])
