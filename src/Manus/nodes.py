import os
import sys
# 获取当前文件所在的目录(src/Manus)
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取src目录
src_dir = os.path.dirname(current_dir)
# 将src目录添加到系统路径中
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

import json
import logging
from typing import Annotated, Literal
from langchain_core.messages import AIMessage, HumanMessage,  SystemMessage, ToolMessage
from langgraph.types import Command, interrupt
from langchain_openai import ChatOpenAI
from state import State
from prompts import *
from Agent.tools import *
from Models import ModelManager
from pprint import pformat
# 从mcp_tool_wrapper导入mcp_assistant工具函数
from Agent.tools.mcp_tool_wrapper import mcp_assistant

model_manager = ModelManager()
# llm = model_manager.get_qwen_model()
llm = model_manager.get_deepseek_model()

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# 创建控制台处理器
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

# 创建文件处理器
file_handler = logging.FileHandler('log.txt', mode='w', encoding='utf-8')
file_handler.setLevel(logging.INFO)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

# 自定义pprint函数，使其输出也记录到日志文件中
def pprint(obj, width=80, depth=None):
    formatted_obj = pformat(obj, width=width, depth=depth)
    logger.info(formatted_obj)
    # 同时在控制台打印
    print(formatted_obj)

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
    """创建计划节点函数，用于生成初始任务计划"""
    # 记录日志，标识当前正在运行Create Planner节点
    logger.info("***正在运行Create Planner node***")
    
    # 构建消息列表，包含系统消息和人类消息
    # 系统消息包含计划系统的提示词
    # 人类消息包含创建计划的提示词，并将用户消息填入其中
    messages = [SystemMessage(content=PLAN_SYSTEM_PROMPT), HumanMessage(content=PLAN_CREATE_PROMPT.format(user_message = state['user_message']))]
    
    # 打印消息内容，便于调试和查看
    logger.info("创建计划节点的输入消息:")
    pprint(messages, width=40, depth=3)
    
    # 调用大语言模型处理消息，生成响应
    logger.info("调用大语言模型生成计划...")
    response = llm.invoke(messages)
    logger.info("大语言模型调用完成")
    logger.info(f"模型响应内容: {response.content[:200]}...")
    
    # 将响应转换为JSON格式的字符串
    response_json = response.model_dump_json(indent=4, exclude_none=True)
    
    # 将JSON字符串解析为Python字典
    response_dict = json.loads(response_json)
    
    # 从响应内容中提取JSON格式的计划
    plan = json.loads(extract_json(extract_answer(response_dict['content'])))

    logger.info("生成的计划:")
    pprint(plan, width=40, depth=3)
    
    # 将生成的计划作为AI消息添加到状态的消息列表中
    plan_json = json.dumps(plan, ensure_ascii=False)
    logger.info(f"将计划添加到state['messages']中: {plan_json[:100]}...")
    state['messages'] += [AIMessage(content=plan_json)]
    
    # 返回命令，指示下一步转向execute节点，并更新状态中的计划
    logger.info("Create Planner node执行完成，转向execute节点")
    return Command(goto="execute", update={"plan": plan})

def update_planner_node(state: State):
    logger.info("***正在运行Update Planner node***")
    plan = state['plan']
    goal = plan['goal']
    logger.info(f"当前计划目标: {goal}")
    
    state['messages'].extend([SystemMessage(content=PLAN_SYSTEM_PROMPT), HumanMessage(content=UPDATE_PLAN_PROMPT.format(plan = plan, goal=goal))])
    messages = state['messages']
    
    logger.info("更新计划节点的输入消息:")
    pprint(messages, width=40, depth=3)
    
    while True:
        try:
            logger.info("调用大语言模型更新计划...")
            response = llm.invoke(messages)
            logger.info("大语言模型调用完成")
            logger.info(f"模型响应内容: {response.content[:200]}...")
            
            response_json = response.model_dump_json(indent=4, exclude_none=True)
            response_dict = json.loads(response_json)
            plan = json.loads(extract_json(extract_answer(response_dict['content'])))
            
            logger.info("更新后的计划:")
            pprint(plan, width=40, depth=3)
            
            plan_json = json.dumps(plan, ensure_ascii=False)
            logger.info(f"将更新后的计划添加到state['messages']中: {plan_json[:100]}...")
            state['messages']+=[AIMessage(content=plan_json)]
            
            logger.info("Update Planner node执行完成，转向execute节点")
            return Command(goto="execute", update={"plan": plan})
        except Exception as e:
            logger.error(f"更新计划时发生错误: {e}")
            error_message = f"json格式错误:{e}"
            logger.info(f"添加错误消息到messages中: {error_message}")
            messages += [HumanMessage(content=error_message)]
            
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
        logger.info("没有更多待处理步骤，转向report节点")
        return Command(goto='report')
    
    # 移除消息历史长度限制，使用所有观察结果
    messages = state['observations'] + [SystemMessage(content=EXECUTE_SYSTEM_PROMPT), HumanMessage(content=EXECUTION_PROMPT.format(user_message=state['user_message'], step=current_step['description']))]
    
    logger.info("初始消息列表:")
    pprint(messages, width=40, depth=3)
    
    tool_result = None
    iteration = 0
    max_iterations = 5  # 限制最大迭代次数以避免无限循环
    while iteration < max_iterations:
        iteration += 1
        logger.info(f"=== 第 {iteration} 轮工具调用循环 ===")
        logger.info("调用模型前的消息列表:")
        pprint(messages, width=40, depth=3)
        
        # 绑定工具到LLM并调用
        tools_list = [create_file, str_replace, shell_exec, mcp_assistant]
        logger.info(f"绑定的工具列表: {[tool.name if hasattr(tool, 'name') else tool.__class__.__name__ for tool in tools_list]}")
        response = llm.bind_tools(tools_list).invoke(messages)
        logger.info("模型调用完成，返回响应")
        logger.info(f"模型响应内容: {response.content[:200]}...")
        logger.info(f"模型响应tool_calls: {response.tool_calls}")
        
        response_json = response.model_dump_json(indent=4, exclude_none=True)
        response_dict = json.loads(response_json)
        # 包含所有工具
        tools = {"create_file": create_file, "str_replace": str_replace, "shell_exec": shell_exec, "mcp_assistant": mcp_assistant}     
        if response_dict['tool_calls']:
            logger.info(f"检测到 {len(response_dict['tool_calls'])} 个工具调用")
            # 先将模型的完整响应作为AIMessage添加到消息列表中（包括tool_calls）
            logger.info("添加包含tool_calls的AIMessage到消息列表")
            messages += [AIMessage(content=response.content, tool_calls=response.tool_calls)]
            
            for i, tool_call in enumerate(response_dict['tool_calls']):
                tool_name = tool_call['name']
                tool_args = tool_call['args']
                tool_call_id = tool_call['id']
                logger.info(f"执行第 {i+1} 个工具调用: {tool_name}")
                logger.info(f"工具参数: {tool_args}")
                
                try:
                    if tool_name == "mcp_assistant":
                        # 对于MCP工具，我们将查询参数作为字符串传递
                        query = tool_args.get("query", "")
                        tool_result = tools[tool_name].invoke({"query": query})
                    else:
                        tool_result = tools[tool_name].invoke(tool_args)
                    logger.info(f"工具 {tool_name} 执行成功")
                    logger.info(f"工具执行结果长度: {len(str(tool_result))} 字符")
                except Exception as e:
                    logger.error(f"执行工具 {tool_name} 时发生错误: {str(e)}")
                    tool_result = f"执行工具时发生错误: {str(e)}"
                
                # 再添加ToolMessage（工具执行结果）
                tool_message_content = f"tool_name:{tool_name},tool_args:{tool_args}\ntool_result:{tool_result}"
                logger.info(f"添加ToolMessage: {tool_message_content[:100]}...")
                messages += [ToolMessage(content=tool_message_content, tool_call_id=tool_call_id)]
        
        else:    
            logger.info("没有检测到工具调用，退出循环")
            break
        
    logger.info(f"当前STEP执行总结: {extract_answer(response_dict['content'])}")
    logger.info("向state['messages']和state['observations']添加执行总结")
    summary_content = extract_answer(response_dict['content'])
    state['messages'] += [AIMessage(content=summary_content)]
    state['observations'] += [AIMessage(content=summary_content)]
    logger.info("execute_node执行完成，转向update_planner")
    return Command(goto='update_planner', update={'plan': plan})

def report_node(state: State):
    """Report node that write a final report."""
    logger.info("***正在运行report_node***")
    
    observations = state.get("observations")
    # 移除消息历史长度限制，使用所有观察结果
    messages = observations + [SystemMessage(content=REPORT_SYSTEM_PROMPT)]
    
    logger.info("report_node初始消息列表:")
    pprint(messages, width=40, depth=3)
    
    iteration = 0
    while True:
        iteration += 1
        logger.info(f"=== report_node第 {iteration} 轮工具调用循环 ===")
        logger.info("调用模型前的消息列表:")
        pprint(messages, width=40, depth=3)
        
        tools_list = [create_file, shell_exec]
        logger.info(f"绑定的工具列表: {[tool.name if hasattr(tool, 'name') else tool.__class__.__name__ for tool in tools_list]}")
        response = llm.bind_tools(tools_list).invoke(messages)
        logger.info("模型调用完成，返回响应")
        logger.info(f"模型响应内容: {response.content[:200]}...")
        logger.info(f"模型响应tool_calls: {response.tool_calls}")
        
        response_json = response.model_dump_json(indent=4, exclude_none=True)
        response_dict = json.loads(response_json)
        tools = {"create_file": create_file, "shell_exec": shell_exec} 
        if response_dict['tool_calls']:    
            logger.info(f"检测到 {len(response_dict['tool_calls'])} 个工具调用")
            # 先将模型的完整响应作为AIMessage添加到消息列表中（包括tool_calls）
            logger.info("添加包含tool_calls的AIMessage到消息列表")
            ai_message = AIMessage(content=response.content, tool_calls=response.tool_calls)
            messages += [ai_message]
            
            for i, tool_call in enumerate(response_dict['tool_calls']):
                tool_name = tool_call['name']
                tool_args = tool_call['args']
                tool_call_id = tool_call['id']
                logger.info(f"执行第 {i+1} 个工具调用: {tool_name}")
                logger.info(f"工具参数: {tool_args}")
                
                try:
                    tool_result = tools[tool_name].invoke(tool_args)
                    logger.info(f"工具 {tool_name} 执行成功")
                    logger.info(f"工具执行结果: {tool_result}")
                except Exception as e:
                    logger.error(f"执行工具 {tool_name} 时发生错误: {str(e)}")
                    tool_result = f"执行工具时发生错误: {str(e)}"
                
                # 再添加ToolMessage（工具执行结果）
                tool_message_content = f"tool_name:{tool_name},tool_args:{tool_args}\ntool_result:{tool_result}"
                logger.info(f"添加ToolMessage: {tool_message_content[:100]}...")
                messages += [ToolMessage(content=tool_message_content, tool_call_id=tool_call_id)]
                
        else:
            logger.info("没有检测到工具调用，退出循环")
            break
            
    logger.info("report_node执行完成")
    logger.info(f"最终报告内容预览: {response_dict['content'][:200]}...")
    return {"final_report": response_dict['content']}
