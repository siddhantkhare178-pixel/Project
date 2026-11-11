import re 
import operator 
import requests
from typing import TypedDict,Annotated
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool
from langchain_core.messages import (
    AnyMessage, HumanMessage, AIMessage, SystemMessage, ToolMessage
)

from langgraph.graph import StateGraph,START,END
from langgraph.errors import GraphRecursionError 

from rag_tool import seach_document  

load_dotenv() 

# --- Tool 2: web search ---
web_search = DuckDuckGoSearchRun()


# --- Tool 3: calculator (safe version, no eval on arbitrary text) ---
@tool
def calculator(expression:str)->str:
    """Evaluate basic math using numbers and + - * / % and parentheses,
    e.g. '12 * 3 + 7'. Powers are not supported.""" 

    if not re.fullmatch(r"[0-9+\-*/%(). ]+", expression) or "**" in expression:
        return "Error: only numbers and + - * / % () are allowed"
    try:
        return str(eval(expression,{"__builtins__": {}},{})) 
    except Exception as e:
        return f"Error: {e}" 


# --- Tool 4: live currency rate ---
@tool
def get_conversion_rate(base_currency:str,target_currency:str) -> str:
    """Get the live exchange rate from base_currency to target_currency, e.g. USD to INR."""
    base,target = base_currency.upper(),target_currency.upper()
    try:
        response = requests.get(
             f"https://api.exchangerate-api.com/v4/latest/{base}", timeout=10
        )
        response.raise_for_status()
        rate = response.json()['rates'][target] 
        return f"1{base} = {rate} {target}" 
    except Exception as e:
        return f"could not fetch the rate:{e}" 
    

tools = [seach_document,web_search,calculator,get_conversion_rate]
tool_map = {t.name: t for t in tools}  


llm = ChatGroq(model="openai/gpt-oss-20b")
llm_with_tools = llm.bind_tools(tools) 

SYSTEM_PROMPT = (
    "You are a helpful assistant with four tools.\n"
    "- Use seach_documents for questions about the uploaded PDF documents, and "
    "mention the source file and page in your answer.\n"
    "- If the passages returned do NOT actually answer the question, say the documents "
    "don't cover it. Do not guess.\n"
    "- Use the web search tool for current events or facts you are unsure about.\n"
    "- Use calculator for non-trivial arithmetic.\n"
    "- Use get_conversion_rate for live currency rates.\n"
    "- If no tool is needed, answer directly."
) 

class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage],operator.add] 


def call_model(state:AgentState):
    response = llm_with_tools.invoke([SystemMessage(content=SYSTEM_PROMPT)] + state["messages"])
    return {"messages":[response]} 


def call_tools(state:AgentState):
    last_message = state["messages"][-1]
    output = [] 

    for tool_call in last_message.tool_calls:
        selected_tool = tool_map.get(tool_call['name'])
        if selected_tool is None:
            result = f"Error:unkown tool {tool_call['name']}"
        else:
            try:
                result = selected_tool.invoke(tool_call["args"])
            except Exception as e:
                result = f"Error running tool : {e}"

        output.append(ToolMessage(
            content=str(result),
            tool_call_id = tool_call['id'],
            name=tool_call['name']

              
              
        ))
    return {"messages":output} 


def should_continue(state:AgentState) -> str:
    if state["messages"][-1].tool_calls:
        return "tools"
    return "end" 


graph = StateGraph(AgentState)
graph.add_node("agent",call_model)
graph.add_node("tools",call_tools)
graph.add_edge(START,"agent")
graph.add_conditional_edges("agent",should_continue,{"tools":"tools","end":END})
graph.add_edge("tools","agent") 



agent_graph = graph.compile() 

session:dict[str,list] = {} 
MAX_MESSAGES = 20 

def ask(question:str,session_id:str = "default") -> str:
    history = session.get(session_id,[])
    try:
        result = agent_graph.invoke(
           {"messages": history + [HumanMessage(content=question)]},
           config={"recursion_limit":12}
        )
        answer = result['messages'][-1].content
    except GraphRecursionError:
        answer = "I got stuck in a loop trying to answer that. Please rephrase the question."


    history = history + [HumanMessage(content=question), AIMessage(content=answer)]
    session[session_id] = history[-MAX_MESSAGES:]
    return answer 


if __name__ == "__main__":
    print("=" * 50)
    print("Agentic RAG Assistant. Type 'exit' to quit.")
    print("=" * 50 + "\n") 

    while True:
        question = input("you:"    ).strip()
        if question in ["exit","quit"]:
            break
        if not question:
            continue 
        print(f"\n Assitant : {ask(question,session_id='terminal')}\n")
