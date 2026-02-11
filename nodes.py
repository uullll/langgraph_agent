import json
import logging
from typing import Annotated, Literal
from langchain_core.messages import AIMessage, HumanMessage,  SystemMessage, ToolMessage
from langgraph.types import Command, interrupt
from langchain_openai import ChatOpenAI
from state import State
from prompts import *
from tools import *

os.environ["no_proxy"] = "localhost,127.0.0.1"
llm = ChatOpenAI(
  model="Qwen/Qwen2.5-VL-7B-Instruct-AWQ",
  base_url="http://127.0.0.1:8000/v1",
  api_key="EMPTY",
  model_kwargs={"tool_choice": "none"},
  temperature=0.0
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
hander = logging.StreamHandler()
hander.setLevel(logging.INFO)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
hander.setFormatter(formatter)
logger.addHandler(hander)

def extract_json(text):
    if '```json' not in text:
        return text
    text = text.split('```json')[1].split('```')[0].strip()
    return text

def extract_answer(text):
    if '</think>' in text:
        answer = text.split("</think>")[-1]
        return answer.strip()
    
    return text

def create_planner_node(state: State):
    
    logger.info("***正在运行Create Planner node***")
    messages = [SystemMessage(content=PLAN_SYSTEM_PROMPT), 
                HumanMessage(content=PLAN_CREATE_PROMPT.format(user_message = state['user_message']))]
    response = llm.invoke(messages)
    response = response.model_dump_json(indent=4, exclude_none=True)
    response = json.loads(response)
    raw = extract_json(extract_answer(response["content"]))
    print(raw)
    plan = json.loads(extract_json(extract_answer(response['content'])))
    state['messages'] += [AIMessage(content=json.dumps(plan, ensure_ascii=False))]
    return Command(goto="execute", update={"plan": plan})

def update_planner_node(state: State):
    logger.info("***正在运行Update Planner node***")
    plan = state['plan']
    goal = plan['goal']
    node_context = [
        SystemMessage(content=PLAN_SYSTEM_PROMPT),
        HumanMessage(content=UPDATE_PLAN_PROMPT.format(plan=plan, goal=goal))
    ]
    last_text = None
    for _ in range(5):
        response = None
        try:
            config = {"recursion_limit": 200}
            response = llm.invoke(node_context,config=config)
            content = response.content if hasattr(response, 'content') else str(response)
            last_text=content
            cleaned_json_str = extract_json(extract_answer(content))
            new_plan = json.loads(cleaned_json_str)
            final_ai_message = AIMessage(content=json.dumps(new_plan, ensure_ascii=False))
            
            return Command(
                goto="execute", 
                update={
                    "plan": new_plan,
                    "messages": [final_ai_message] # LangGraph 会自动将其 extend 到 state['messages']
                }
            )
        except Exception as e:
            logger.error(f"Update planner 解析失败")
            snippet = (last_text or "")[:1500]
            node_context.append
            (HumanMessage(
                content=(
                    "上次输出无法解析为严格 JSON。请只返回一个合法 JSON，不要任何额外文字。\n"
                    f"错误信息: {type(e).__name__}: {e}\n"
                    f"上次输出片段:\n{snippet}"
                )
            ))
            
def execute_node(state: State):
    logger.info("***正在运行execute_node***")
  
    plan = state['plan']
    steps = plan['steps']
    current_step = None
    current_step_index = 0
    
    # 获取第一个未完成STEP
    for i, step in enumerate(steps):
        status = step['status']
        if status == 'pending':
            current_step = step
            current_step_index = i
            break
        
    logger.info(f"当前执行STEP:{current_step}")
    
    ## 此处只是简单跳转到report节点，实际应该根据当前STEP的描述进行判断
    if current_step is None or current_step_index == len(steps)-1:
        return Command(goto='report')
    
    messages = state['observations'] + [SystemMessage(content=EXECUTE_SYSTEM_PROMPT), HumanMessage(content=EXECUTION_PROMPT.format(user_message=state['user_message'], step=current_step['description']))]
    
    tool_result = None
    while True:
        response = llm.bind_tools([create_file, str_replace, shell_exec]).invoke(messages)
        messages.append(response)
        #response = response.model_dump_json(indent=4, exclude_none=True)
        #response = json.loads(response)
        tools = {"create_file": create_file, 
                 "str_replace": str_replace, 
                 "shell_exec": shell_exec
                }     
        if response.tool_calls:
            for tool_call in response.tool_calls:
                tool_name = tool_call['name']
                tool_args = tool_call['args']
                tool_id   = tool_call["id"]
                tool_result = tools[tool_name].invoke(tool_args)
                logger.info(f"tool_name:{tool_name},tool_args:{tool_args}\ntool_result:{tool_result}")
                messages.append(
                    ToolMessage(
                        content=json.dumps(tool_result,ensure_ascii=False),
                        tool_call_id=tool_id
                    )
                )
            continue
        else:    
            break
        
    logger.info(f"当前STEP执行总结:{extract_answer(response.content)}")
    state['messages'] += [AIMessage(content=extract_answer(response.content))]
    state['observations'] += [AIMessage(content=extract_answer(response.content))]
    return Command(goto='update_planner', update={'plan': plan})
    

    
def report_node(state: State):
    """Report node that write a final report."""
    logger.info("***正在运行report_node***")
    
    observations = state.get("observations")
    messages = observations + [SystemMessage(content=REPORT_SYSTEM_PROMPT)]
    
    while True:
        response = llm.bind_tools([create_file, str_replace, shell_exec]).invoke(messages)
        messages.append(response)
        #response = response.model_dump_json(indent=4, exclude_none=True)
        #response = json.loads(response)
        tools = {"create_file": create_file, 
                 "str_replace": str_replace, 
                 "shell_exec": shell_exec}     
        if response.tool_calls:
            for tool_call in response.tool_calls:
                tool_name = tool_call['name']
                tool_args = tool_call['args']
                tool_id   = tool_call["id"]
                tool_result = tools[tool_name].invoke(tool_args)
                logger.info(f"tool_name:{tool_name},tool_args:{tool_args}\ntool_result:{tool_result}")
                messages.append(
                    ToolMessage(
                        content=json.dumps(tool_result,ensure_ascii=False),
                        tool_call_id=tool_id
                    )
                )
            continue
        else:
            break
            
    return {"final_report": response['content']}


