from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from state import State
from nodes import (
    report_node,
    execute_node,
    create_planner_node,
    update_planner_node
)


def _build_base_graph():
    """Build and return the base state graph with all nodes and edges."""
    builder = StateGraph(State)
    builder.add_edge(START, "create_planner")
    builder.add_node("create_planner", create_planner_node)
    builder.add_node("update_planner", update_planner_node)
    builder.add_node("execute", execute_node)
    builder.add_node("report", report_node)
    builder.add_edge("report", END)
    return builder


def build_graph_with_memory():
    """Build and return the agent workflow graph with checkpoint memory."""
    memory = MemorySaver()
    builder = _build_base_graph()
    return builder.compile(checkpointer=memory)


def build_graph():
    """Build and return the agent workflow graph. Defaults to memory-enabled graph."""
    return build_graph_with_memory()



graph = build_graph()


if __name__ == "__main__":
    inputs = {
        "user_id": "demo_user",
        "user_message": "对所给文档进行分析，生成一份分析报告，需要用图表为结论证明，不需要分析太多内容，只需要分析成绩与什么正相关即可,文档名称为dataset.parquet",
        "plan": None,
        "observations": [],
        "final_report": "",
    }
    graph.invoke(inputs, {"recursion_limit": 100, "configurable": {"thread_id": "demo_user_thread"}})
