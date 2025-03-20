import os
import json
import requests
import datetime
from typing import Optional
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.vectorstores import FAISS

# Import necessary services
from services.Intranet_repository import IntranetRepository
from services.Intranet_repository_ar import IntranetRepositoryAR

# Load environment variables
load_dotenv()

# Initialize LLM and vector store
llm = ChatOpenAI(model="gpt-4-turbo-preview")
intranet_repository = IntranetRepository()
vectorstore = intranet_repository.create_or_load_faiss_index()
intranet_repository_ar = IntranetRepositoryAR()
vectorstore_ar = intranet_repository_ar.create_or_load_faiss_index()

# @tool
# def classify_question(input: str) -> str:
#     """
#     Analyzes user input and classifies it as salary request, vacancy request, or global question.
#     This tool should be used first to determine which subsequent tool to use.
    
#     :param input: The user's message
#     :return: JSON string with request_type and employee_code if found
#     """
#     print("***** CLASSIFY QUESTION TOOL *****")
    
#     # Create the classifier prompt
#     actor_prompt_template = ChatPromptTemplate.from_messages([
#         (
#             "system",
#             """You are an advanced AI specialized in analyzing and extracting structured information from text.

#             Current time: {time}

#             Instructions:
#             1. Analyze the user's input and classify it into one of the following categories:
#             - 'salary_request': Queries related to salary.
#             - 'vacancy_request': Queries related to leave balances.
#             - 'global_question': General or unrelated queries.
#             2. Identify and extract the 'employeeCode' if present. The employee code can appear in various forms, including but not limited to:
#             - Phrases like "my number is xxx ", "my code is xxx", "employee ID is xxx", "ID: xxx", or any similar variation.
#             - Formats such as numeric codes (e.g., 12345) or alphanumeric (e.g., ABC123).
#             3. Return the response in JSON format with the following structure:
#             - 'request_type': The identified type of request.
#             - 'employee_code': The extracted employee code, or null if none is found.""",
#         ),
#         ("user", "{input}"),
#         ("system", "Analyze the above query and classify it accurately.")
#     ]).partial(
#         time=lambda: datetime.datetime.now().isoformat(),
#     )
    
#     # Process the input with the LLM
#     response = llm.invoke(actor_prompt_template.format(input=input))
    
#     # Extract the content
#     try:
#         # Get the classification information
#         result = {
#             "request_type": "global_question",
#             "employee_code": None
#         }
        
#         # Parse the response - look for JSON in the response or create a structured output
#         if "{" in response.content and "}" in response.content:
#             json_str = response.content[response.content.find("{"):response.content.rfind("}")+1]
#             try:
#                 parsed_json = json.loads(json_str)
#                 if "request_type" in parsed_json:
#                     result["request_type"] = parsed_json["request_type"]
#                 if "employee_code" in parsed_json:
#                     result["employee_code"] = parsed_json["employee_code"]
#             except:
#                 # Fallback to structured extraction if JSON parsing fails
#                 pass
        
#         return json.dumps(result)
#     except Exception as e:
#         return json.dumps({"request_type": "global_question", "employee_code": None, "error": str(e)})

@tool
def query_salary_information(employee_code: str) -> str:
    """
    Queries salary information for a specific employee.
    Use this tool when the user is asking about their salary information.
    
    :param employee_code: The employee's identification code
    :return: Salary information response
    """
    print("***** SALARY INFORMATION TOOL *****")
    print(f"Employee Code: {employee_code}")
    
    url = os.getenv("SALARY_ENDPOINT_URL")
    payload = {"employeeCode": employee_code}
    
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        api_result = response.json()
        
        salary_days = api_result.get('YTDPayroll', '-1')
        
        if salary_days == -1:
            message = "No salary data available for this employee."
        else:
            name = api_result.get('name', "Employee")
            message = f"{name}, your YTD salary is {salary_days}"
            
        return message
    except requests.RequestException as e:
        return f"Error retrieving salary information: {str(e)}"

@tool
def query_vacancy_balance(employee_code: str) -> str:
    """
    Queries vacation/leave balance for a specific employee.
    Use this tool when the user is asking about their vacation or leave balance.
    
    :param employee_code: The employee's identification code
    :return: Vacation/leave balance information response
    """
    print("***** VACANCY BALANCE TOOL *****")
    print(f"Employee Code: {employee_code}")
    
    url = os.getenv("VACANCY_ENDPOINT_URL")
    payload = {"employeeCode": employee_code}
    
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        api_result = response.json()
        
        vacancy_days = api_result.get('vacancyBalanceDays', '-1')
        
        if vacancy_days == -1:
            message = "No vacancy balance data available for this employee."
        else:
            name = api_result.get('name', "Employee")
            message = f"Your vacancy balance days is {vacancy_days}, {name}. Enjoy your time off!"
            
        return message
    except requests.RequestException as e:
        return f"Error retrieving vacancy balance information: {str(e)}"

@tool
def query_document_knowledge(query: str) -> str:
    """
    Queries the document knowledge base for information related to the user's question.
    Always use this tool to enrich information about orders, ensuring it works
    together with query_document_knowledge_ar.
    if conflicting information 
    is found in another tool, this tool must be used as default to resolve the conflict.
    
    :param query: The user's question
    :return: Combined information from both knowledge bases
    """
    print("***** DOCUMENT KNOWLEDGE TOOL *****")
    print(f"Query: {query}")
    
    # Retrieve relevant context from vector store
    def get_context(question, k=3):
        docs = vectorstore.similarity_search(question, k=k)
        if docs:
            results = []
            for doc in docs:
                source = doc.metadata.get('source', 'Unknown')
                content = doc.page_content
                results.append(f"[Source: {source}]\n{content}")
            
            return "\n\n".join(results)
        return "No relevant information found."
    
    # Build prompt with context
    def build_prompt_with_context(question, context):
        prompt = f"""
        You are an expert HR assistant. Use the context below to answer the
        user's question accurately and concisely.

        Context:
        {context}

        Question:
        {question}

        Answer:
        """
        return prompt
    
    try:
        # Get context and generate answer
        context = get_context(query)
        prompt = build_prompt_with_context(query, context)
        response = llm.invoke(prompt)
        
        return response.content
    except Exception as e:
        return f"Error retrieving information: {str(e)}"

@tool
def format_final_response(response_data: str) -> str:
    """
    Formats the final response to be presented to the user.
    This tool should be used as the final step to ensure consistent response formatting.
    

    :param response_data: JSON string containing the response data and any relevant metadata
    :return: Formatted response ready to be shown to the user
    """
    print("***** RESPONSE FORMATTER TOOL *****")
    
    try:
        # Parse the response data
        data = json.loads(response_data)
        
        # Extract the answer
        answer = data.get('answer', 'No response available.')
        
        # Format the answer for presentation
        formatted_response = llm.invoke(
            f"""
            Format the following HR assistant response to be clear, professional, and friendly:
            
            {answer}
            
            Do not add any disclaimers or additional information. Just refine the existing response.
            """
        )
        
        return formatted_response.content
    except Exception as e:
        return f"Error formatting response: {str(e)}"
    
@tool
def query_document_knowledge_ar(query: str) -> str:
    """
    Queries the document knowledge base (AR) for information related to the user's question.
    This tool is always used in combination with query_document_knowledge.
    Always use this tool to enrich information about orders, ensuring it works
    
    :param query: The user's question
    :return: Information from the AR knowledge base relevant to the question
    """
    print("***** DOCUMENT KNOWLEDGE AR TOOL  *****")
    print(f"Query: {query}")
    
    # Retrieve relevant context from vector store
    def get_context(question, k=3):
        docs = vectorstore_ar.similarity_search(question, k=k)
        if docs:
            results = []
            for doc in docs:
                source = doc.metadata.get('source', 'Unknown')
                content = doc.page_content
                results.append(f"[Source: {source}]\n{content}")
            
            return "\n\n".join(results)
        return "No relevant information found."
    
    # Build prompt with context
    def build_prompt_with_context(question, context):
        prompt = f"""
        You are an expert HR assistant. Use the context below to answer the
        user's question accurately and concisely.

        Context:
        {context}

        Question:
        {question}

        Answer:
        """
        return prompt
    
    try:
        # Get context and generate answer
        context = get_context(query)
        prompt = build_prompt_with_context(query, context)
        response = llm.invoke(prompt)
        
        return response.content
    except Exception as e:
        return f"Error retrieving information: {str(e)}"
    


@tool
def query_order_details(order_code: str) -> str:
    """
    Retrieves order details by querying the order API with an order code.
    Use this tool when a user asks about order status or details.
    
    :param order_code: The order code/number to look up
    :return: Order details information
    """
    print("***** ORDER DETAILS TOOL *****")
    print(f"Order Code: {order_code}")
    
    # Get endpoint from environment or use default
    url = os.getenv("ORDER_ENDPOINT_URL", "http://localhost:8000/order/details")
    
    # Prepare the payload
    payload = {"orderCode": order_code}
    
    try:
        # Send request to the endpoint
        response = requests.post(url, json=payload)
        response.raise_for_status()  # Raise an error for bad status codes
        
        # Parse the response
        order_data = response.json()
        
        # Format the response message
        message = f"Order Information:\n"
        message += f"Order Code: {order_data.get('orderCode', 'Unknown')}\n"
        message += f"Status: {order_data.get('status', 'Unknown')}\n"
        message += f"Details: {order_data.get('orderDetails', 'No details available')}\n"
        
        return message
    except requests.exceptions.RequestException as e:
        # Handle connection or API errors
        return f"Error retrieving order information: {str(e)}"
    except Exception as e:
        # Handle any other errors
        return f"Unexpected error processing order request: {str(e)}"