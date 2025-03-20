# run.py
import streamlit as st
from dotenv import load_dotenv
from langchain_core.agents import AgentFinish
from langgraph.graph import END, StateGraph
from nodes import execute_tools, run_agent_reasoning_engine
from state import AgentState

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

query = st.chat_input("Question?")

if query:
   app = create_app()
   result = app.invoke({"input": query})
   st.write(result["agent_outcome"].return_values["output"])

if __name__ == "__main__":
   pass