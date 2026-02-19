import json
import logging
import textwrap
from typing import Annotated, Literal
from pathlib import Path
from langchain_core.messages import AIMessage, HumanMessage,  SystemMessage, ToolMessage
from langgraph.types import Command, interrupt
from langchain_openai import ChatOpenAI
from state import State
from prompts import *
from tools import *
from config import WORKSPACE, get_setting
OPENAI_API_KEY = get_setting("OPENAI_API_KEY", "llm.api_key", required=True)
OPENAI_MODEL = get_setting("OPENAI_MODEL", "llm.model", default="gpt-4o-mini")
OPENAI_BASE_URL = get_setting("OPENAI_BASE_URL", "llm.base_url")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY is not set.")
llm = ChatOpenAI(
  model=OPENAI_MODEL,
  api_key=OPENAI_API_KEY,
  base_url=OPENAI_BASE_URL,
  temperature=0.0,
  request_timeout=60
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
hander = logging.StreamHandler()
hander.setLevel(logging.INFO)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
hander.setFormatter(formatter)
logger.addHandler(hander)

def load_hf_dataset_once(state: State):
    if state.get("file_path"):
        return state["file_path"]

    ds = load_hf_dataset.invoke({})
    save_dataset.invoke({"ds": ds})
    save_path = WORKSPACE / "dataset.parquet"

    state["file_path"] = str(save_path)
    return str(save_path)
def _build_pdf_report(report_text: str, workspace: Path) -> dict:
    """Build a PDF report from final text and all images under workspace/."""
    try:
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_pdf import PdfPages
    except Exception as e:
        return {"ok": False, "error": f"matplotlib unavailable: {e}"}


    workspace.mkdir(parents=True, exist_ok=True)
    pdf_path = workspace / "analysis_report.pdf"
    image_paths = []
    for pattern in ("*.png", "*.jpg", "*.jpeg", "*.webp"):
        image_paths.extend(sorted(workspace.glob(pattern)))

    try:
        with PdfPages(pdf_path) as pdf:
            # First page: text summary
            fig, ax = plt.subplots(figsize=(8.27, 11.69))  # A4 portrait
            ax.axis("off")
            safe_text = report_text if report_text else "No final report text generated."
            wrapped_text = "\n".join(textwrap.wrap(safe_text[:6000], width=90))
            ax.text(0.02, 0.98, wrapped_text, va="top", ha="left", fontsize=10)
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

            # Following pages: one image per page
            for image_path in image_paths:
                fig, ax = plt.subplots(figsize=(11.69, 8.27))  # A4 landscape
                ax.axis("off")
                ax.set_title(image_path.name)
                img = plt.imread(image_path)
                ax.imshow(img)
                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)
    except Exception as e:
        return {"ok": False, "error": str(e)}

    return {
        "ok": True,
        "pdf_path": str(pdf_path),
        "images_embedded": len(image_paths),
    }

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

def normalize_tool_call(tool_call: dict):
    """Normalize tool call fields from different providers/casings."""
    if tool_call is None:
        return None, None, None
    name = tool_call.get('name') or tool_call.get('Name') or tool_call.get('tool_name')
    args = tool_call.get('args') or tool_call.get('arguments') or tool_call.get('tool_args')
    tool_id = tool_call.get('id') or tool_call.get('tool_call_id')
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except Exception:
            pass
    return name, args, tool_id

def create_planner_node(state: State):
    logger.info("***正在下载数据集***")
    load_hf_dataset_once(state)
    logger.info("***正在运行Create Planner node***")
    messages = [SystemMessage(content=PLAN_SYSTEM_PROMPT), 
                HumanMessage(content=PLAN_CREATE_PROMPT.format(user_message = state['user_message']))]
    response = llm.invoke(messages)
    response = response.model_dump_json(indent=4, exclude_none=True)
    response = json.loads(response)
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
            response = llm.invoke(node_context)
            content = response.content if hasattr(response, 'content') else str(response)
            last_text=content
            cleaned_json_str = extract_json(extract_answer(content))
            new_plan = json.loads(cleaned_json_str)
            final_ai_message = AIMessage(content=json.dumps(new_plan, ensure_ascii=False))
            
            return Command(
                goto="execute", 
                update={
                    "plan": new_plan,
                    "messages": [final_ai_message]  # LangGraph auto-extends this into state['messages']
                }
            )
        except Exception as e:
            logger.error("Update planner JSON parse failed")
            snippet = (last_text or "")[:1500]
            node_context.append(HumanMessage(
                content=(
                    "Your previous output could not be parsed as strict JSON. "
                    "Return one valid JSON object only, with no extra text.\n"
                    f"Error: {type(e).__name__}: {e}\n"
                    f"Previous output snippet:\n{snippet}"
                )
            ))
            
def execute_node(state: State):
    logger.info("***正在运行execute_node***")
    plan = state['plan']
    steps = plan['steps']
    current_step = None
    current_step_index = 0
    
    # Get the first pending step
    for i, step in enumerate(steps):
        status = step['status']
        if status == 'pending':
            current_step = step
            current_step_index = i
            break
        
    logger.info(f"Current executing step: {current_step}")
    
    # Jump to report only when no pending step remains.
    if current_step is None:
        return Command(goto='report')
    
    messages = state['observations'] + [SystemMessage(content=EXECUTE_SYSTEM_PROMPT), HumanMessage(content=EXECUTION_PROMPT.format(user_message=state['user_message'], step=current_step['description']))]
    
    tool_result = None
    while True:
        response = llm.bind_tools([create_file, str_replace, shell_exec]).invoke(messages)
        messages.append(response)
        tools = {"create_file": create_file, 
                 "str_replace": str_replace, 
                 "shell_exec": shell_exec
                }     
        if response.tool_calls:
            for tool_call in response.tool_calls:
                tool_name, tool_args, tool_id = normalize_tool_call(tool_call)
                if not tool_name:
                    logger.warning(f"Tool call missing name, raw payload: {tool_call}")
                    continue
                tool_result = tools[tool_name].invoke(tool_args)
                logger.info(f"tool_name:{tool_name},tool_args:{tool_args}\ntool_result:{tool_result}")
                messages.append(
                    ToolMessage(
                        content=json.dumps(tool_result,ensure_ascii=False),
                        tool_call_id=tool_id
                    )
                )
            steps[current_step_index]['status'] = 'completed'
            continue
        else:    
            break
        
    logger.info(f"Current step summary: {extract_answer(response.content)}")
    # If no tool call happened but a step summary was produced, still mark step completed to avoid loops.
    if current_step is not None and steps[current_step_index].get('status') != 'completed':
        steps[current_step_index]['status'] = 'completed'
    
    state['messages'] += [AIMessage(content=extract_answer(response.content))]
    state['observations'] += [AIMessage(content=extract_answer(response.content))]
    return Command(goto='update_planner', update={'plan': plan})
    

    
def report_node(state: State):
    """Report node that write a final report."""
    logger.info("***正在运行report_node***")
    
    observations = state.get("observations") or []
    user_message = state.get("user_message", "")
    messages = observations + [
        SystemMessage(content=REPORT_SYSTEM_PROMPT)
    ]
    max_rounds = 8
    rounds = 0
    while rounds < max_rounds:
        rounds += 1
        response = llm.bind_tools([create_file, str_replace, shell_exec]).invoke(messages)
        messages.append(response)
        tools = {"create_file": create_file, 
                 "str_replace": str_replace, 
                 "shell_exec": shell_exec}     
        if response.tool_calls:
            for tool_call in response.tool_calls:
                tool_name, tool_args, tool_id = normalize_tool_call(tool_call)
                if not tool_name:
                    logger.warning(f"Tool call missing name, raw payload: {tool_call}")
                    continue
                tool_result = tools[tool_name].invoke(tool_args)
                logger.info(f"tool_name:{tool_name},tool_args:{tool_args}\ntool_result:{tool_result}")
                messages.append(
                    ToolMessage(
                        content=json.dumps(tool_result, ensure_ascii=False),
                        tool_call_id=tool_id
                    )
                )
                
            continue
        else:
            pdf_files = sorted(WORKSPACE.glob("*.pdf"))
            if not pdf_files:
                logger.warning("No PDF generated yet in workspace, ask model to continue.")
                messages.append(
                    HumanMessage(
                        content=(
                            "You have not created any .pdf file under workspace yet. "
                            "Please continue and create a real PDF file (not txt/md) in workspace, "
                            "then briefly report the file path."
                        )
                    )
                )
                continue
            break
        

    pdf_files = sorted(WORKSPACE.glob("*.pdf"))
    pdf_path = str(pdf_files[-1]) if pdf_files else ""
    return {
        "final_report": response.content,
        "final_report_pdf_path": pdf_path,
    }