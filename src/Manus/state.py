from langgraph.graph import MessagesState
from typing import Optional, List, Dict, Literal

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


class Step(BaseModel):
    """步骤模型，表示计划中的单个步骤"""
    title: str = ""  # 步骤标题
    description: str = ""  # 步骤详细描述
    status: Literal["pending", "completed"] = "pending"  # 步骤状态：待处理或已完成


class Plan(BaseModel):
    """计划模型，包含目标、思考过程和具体步骤"""
    goal: str = ""  # 计划目标
    thought: str = ""  # 思考过程
    steps: List[Step] = []  # 步骤列表


class State(MessagesState):
    """
    应用状态类，继承自MessagesState
    
    Attributes:
        plan: 执行计划
        user_message: 用户输入的消息
        observations: 观察结果列表
        final_report: 最终报告
    """
    plan: Optional[Plan] = None
    user_message: str
    observations: List = []
    final_report: str = ""