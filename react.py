# react.py
from dotenv import load_dotenv
from langchain import hub
from langchain.agents import create_react_agent
from langchain_core.prompts import PromptTemplate
from langchain_openai.chat_models import ChatOpenAI
from tools import query_document_knowledge_ar, query_order_details, query_salary_information, query_vacancy_balance, query_document_knowledge, format_final_response

load_dotenv()

# We can keep the standard ReAct prompt or customize it for our HR system
react_prompt = hub.pull("hwchase17/react")

# Define our tools
tools = [    
    query_salary_information,
    query_vacancy_balance,
    query_document_knowledge,
    query_document_knowledge_ar,
    format_final_response,
    query_order_details
]

llm = ChatOpenAI(model="gpt-4-turbo")

react_agent_runnable = create_react_agent(llm, tools, react_prompt)