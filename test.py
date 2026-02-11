# pip install -U langgraph langchain-openai
# 需要设置环境变量：OPENAI_API_KEY

from typing import TypedDict, List, Optional
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI

# --------- 1) 定义 State（图里的“全局状态”）---------
class State(TypedDict):
    messages: List[dict]          # 简化：用 dict 存 role/content
    tool_result: Optional[str]    # 工具结果

llm = ChatOpenAI(model="Qwen/Qwen2.5-3B-Instruct",
  base_url="http://127.0.0.1:8000/v1",
  api_key="EMPTY",
  model_kwargs={"tool_choice": "none"},
  temperature=0.0
)

# --------- 2) 一个“工具”（普通 Python 函数即可）---------
def fake_search_tool(query: str) -> str:
    # 你可以在这里换成真正的检索/RAG/数据库查询
    return f"[fake_search] 我查到：关于「{query}」的3条要点：A...B...C..."

# --------- 3) 节点：assistant（先决定要不要用工具）---------
def assistant_node(state: State) -> State:
    last = state["messages"][-1]["content"]

    # 简单策略：如果用户句子里包含“查/搜索/资料”，就走工具，否则直接回答并结束
    need_tool = any(k in last for k in ["查", "搜索", "资料", "检索"])
    if need_tool and state.get("tool_result") is None:
        # 让图跳到 tool 节点（用 conditional edge）
        state["messages"].append({"role": "assistant", "content": "我去查一下…"})
        return state

    # 如果已经有 tool_result，就把它喂给 LLM 生成最终回答
    if state.get("tool_result"):
        prompt = (
            "你是一个助手。下面是工具返回的信息，请基于它回答用户问题。\n\n"
            f"工具信息：{state['tool_result']}\n\n"
            f"用户问题：{last}"
        )
        resp = llm.invoke(prompt)
        state["messages"].append({"role": "assistant", "content": resp.content})
        return state

    # 不需要工具：直接让 LLM 回答
    resp = llm.invoke(last)
    state["messages"].append({"role": "assistant", "content": resp.content})
    return state

# --------- 4) 节点：tool（从用户问题提取 query，调用工具）---------
def tool_node(state: State) -> State:
    user_q = state["messages"][-2]["content"]  # 上一个 user 内容（因为 assistant 刚 append 了“我去查一下…”）
    state["tool_result"] = fake_search_tool(user_q)
    return state

# --------- 5) 条件跳转：assistant -> (tool 或 END)---------
def route_after_assistant(state: State) -> str:
    last_user = state["messages"][0]["content"] if state["messages"] else ""
    # 更稳：看 tool_result 是否已经存在；不存在且用户包含关键词 -> tool
    if state.get("tool_result") is None:
        # 取最初用户问题（你也可以取最新 user）
        u = state["messages"][0]["content"]
        if any(k in u for k in ["查", "搜索", "资料", "检索"]):
            return "tool"
    return END

# --------- 6) 组装 Graph ---------
g = StateGraph(State)
g.add_node("assistant", assistant_node)
g.add_node("tool", tool_node)

g.set_entry_point("assistant")
g.add_conditional_edges("assistant", route_after_assistant, {"tool": "tool", END: END})
g.add_edge("tool", "assistant")  # 工具结束后回到 assistant 生成最终回答

app = g.compile()

# --------- 7) 跑起来 ---------
if __name__ == "__main__":
    init_state: State = {
        "messages": [{"role": "user", "content": "帮我搜索一下 LangGraph demo 都在演示什么？"}],
        "tool_result": None,
    }
    out = app.invoke(init_state)
    for m in out["messages"]:
        print(f"{m['role']}: {m['content']}")
