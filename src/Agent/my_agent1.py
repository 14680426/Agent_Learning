import os
import sys

# 获取当前文件所在的目录(src/Agent)
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取src目录
src_dir = os.path.dirname(current_dir)
# 将src目录添加到系统路径中
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from langchain.agents.factory import create_agent
from langchain_core.tools import tool
from Models import ModelManager


# 初始化模型管理器并获取模型实例
print("🔄 初始化模型管理器...")
model_manager = ModelManager()
llm = model_manager.get_qwen_model()
print("✅ 模型初始化完成")


@tool
def send_email(to: str, subject: str, body: str):
    """发送邮件 - 该工具可以发送电子邮件给指定收件人
    
    Args:
        to: 收件人邮箱地址或姓名
        subject: 邮件主题
        body: 邮件正文内容
    """
    email = {
        "to": to,
        "subject": subject,
        "body": body
    }
    # ...邮件发送逻辑
    print(f"📧 工具执行: send_email(to='{to}', subject='{subject}', body='{body}')")

    return f"邮件已发送至 {to}"


# 创建 React Agent
agent_executor = create_agent(
    model=llm,
    tools=[send_email],
    system_prompt="你是一个邮件助手。"
)
print("✅ Agent创建完成!")


# 添加测试输入
if __name__ == "__main__":
    import asyncio
    
    async def main():
        print("\n" + "="*60)
        print("🚀 开始执行Agent测试")
        print("="*60)
        
        # 测试输入
        inputs = {
            "messages": [
                {"role": "user", "content": "请帮我给张三发一封邮件，告诉他会议时间改到明天下午3点了，主题是项目进度同步。"}
            ]
        }
       
        # 异步流式执行
        print("📡 开始流式执行...")
        async for chunk in agent_executor.astream(inputs, stream_mode="updates"):
            print(chunk)
        print("\n执行完成")
                
    asyncio.run(main())
    print("🎊 程序执行完毕")