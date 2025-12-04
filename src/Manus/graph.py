from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from state import State
from nodes import (
    report_node,
    execute_node,
    create_planner_node,
    update_planner_node
)
import os
import sys

# 获取当前文件所在的目录(src/Manus)
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取src目录
src_dir = os.path.dirname(current_dir)
# 将src目录添加到系统路径中
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

def _build_base_graph():
    """构建并返回包含所有节点和边的基础状态图
    
    图结构说明:
    START -> create_planner -> update_planner <-> execute -> report -> END
    
    节点说明:
    - START: 图的起始节点
    - create_planner: 创建初始计划节点，负责根据用户请求生成初步的执行计划
    - update_planner: 更新计划节点，根据执行结果更新和完善计划
    - execute: 执行节点，负责执行计划中的具体步骤
    - report: 报告节点，生成最终的执行报告
    - END: 图的结束节点
    """
    builder = StateGraph(State)
    # 添加从起始节点到创建计划节点的边
    builder.add_edge(START, "create_planner")
    # 添加创建计划节点
    builder.add_node("create_planner", create_planner_node)
    # 添加更新计划节点
    builder.add_node("update_planner", update_planner_node)
    # 添加执行节点
    builder.add_node("execute", execute_node)
    # 添加报告节点
    builder.add_node("report", report_node)
    # 添加从报告节点到结束节点的边
    builder.add_edge("report", END)
    return builder


def build_graph_with_memory():
    """构建并返回带内存的代理工作流图"""
    memory = MemorySaver()
    builder = _build_base_graph()
    return builder.compile(checkpointer=memory)


def build_graph():
    """构建并返回不带内存的代理工作流图"""
    # 构建状态图
    builder = _build_base_graph()
    return builder.compile()


graph = build_graph()  


inputs = {"user_message": "我需要从常宁市宜阳小学开车前往长沙市火车站，请你分别给出现在出发的多种路线方案，然后进行各种指标的对比，并给出一个md详细对比的高质量报告。", 
          "plan": None,
          "observations": [], 
          "final_report": ""}


graph.invoke(inputs, {"recursion_limit":100})