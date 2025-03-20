# run.py
import streamlit as st
from dotenv import load_dotenv
from langchain_core.agents import AgentFinish
from langgraph.graph import END, StateGraph
from nodes import execute_tools, run_agent_reasoning_engine
from state import AgentState
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv()

def create_app():
   AGENT_REASON = "agent_reason" 
   ACT = "act"

   def should_continue(state: AgentState) -> str:
       if isinstance(state["agent_outcome"], AgentFinish):
           return END
       return ACT

   flow = StateGraph(AgentState)
   flow.add_node(AGENT_REASON, run_agent_reasoning_engine)
   flow.set_entry_point(AGENT_REASON)
   flow.add_node(ACT, execute_tools)
   flow.add_conditional_edges(AGENT_REASON, should_continue)
   flow.add_edge(ACT, AGENT_REASON)
   
   return flow.compile()

st.title("AI Agent Orders")
st.write("Ask me about orders")

# Initialize conversation history
if "history" not in st.session_state:
    st.session_state.history = []

# Display existing conversation
for message in st.session_state.history:
    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.markdown(message.content)
    elif isinstance(message, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(message.content)

query = st.chat_input("Question?")

if query:
    st.session_state.history.append(HumanMessage(content=query))
    
    MAX_HISTORY = 10
    st.session_state.history = st.session_state.history[-MAX_HISTORY:]
    
    with st.chat_message("user"):
        st.markdown(query)
    
    try:
        app = create_app()
        
        context = ""
        for i, msg in enumerate(st.session_state.history[:-1]): 
            if isinstance(msg, HumanMessage):
                context += f"User: {msg.content}\n"
            elif isinstance(msg, AIMessage):
                context += f"Assistant: {msg.content}\n"
        
        if context:
            enriched_query = f"Previous conversation:\n{context}\n\nCurrent question: {query}"
        else:
            enriched_query = query
        
        result = app.invoke({"input": enriched_query})
        response = result["agent_outcome"].return_values["output"]
        
        st.session_state.history.append(AIMessage(content=response))
        
        with st.chat_message("assistant"):
            st.markdown(response)
            
    except Exception as e:
        error_message = f"An error occurred: {str(e)}"
        st.session_state.history.append(AIMessage(content=error_message))
                
        with st.chat_message("assistant"):
            st.markdown(error_message)

if __name__ == "__main__":
    pass