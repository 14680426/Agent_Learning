import openai
import json
from typing import List, Dict, Any
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

class DeepSeekReasoner:
    def __init__(self, api_key: str, base_url: str, model: str = "DeepSeek-R1-0528"):
        self.client = openai.OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
    
    def invoke_with_thinking(self, messages: List[BaseMessage], **kwargs) -> AIMessage:
        """
        调用DeepSeek模型并显示思考过程
        
        Args:
            messages: 消息列表
            **kwargs: 其他参数
            
        Returns:
            AIMessage: 包含最终回答的AI消息
        """
        # 将LangChain消息转换为OpenAI格式
        openai_messages = self._convert_to_openai_messages(messages)
        
        print("=" * 60)
        print("🤔 DeepSeek 思考过程:")
        print("=" * 60)
        
        try:
            # 启用思考功能
            response = self.client.chat.completions.create(
                model=self.model,
                messages=openai_messages,
                extra_body={
                    "enable_thinking": True  # 关键：启用思考功能
                },
                **kwargs
            )
            
            # 打印完整的响应结构（用于调试）
            print("\n📋 完整响应结构:")
            print(json.dumps(response.model_dump(), indent=2, ensure_ascii=False))
            
            # 提取思考过程和最终回答
            thinking_content, final_answer = self._extract_thinking_and_answer(response)
            
            # 打印思考过程
            if thinking_content:
                print("\n💭 模型思考过程:")
                print("-" * 40)
                print(thinking_content)
                print("-" * 40)
            
            # 打印最终答案
            print(f"\n✅ 最终答案: {final_answer}")
            print("=" * 60)
            
            # 返回AIMessage
            return AIMessage(content=final_answer)
            
        except Exception as e:
            print(f"❌ 调用失败: {e}")
            # 返回错误信息
            return AIMessage(content=f"调用失败: {str(e)}")
    
    def _convert_to_openai_messages(self, messages: List[BaseMessage]) -> List[Dict[str, str]]:
        """将LangChain消息转换为OpenAI格式"""
        openai_messages = []
        for msg in messages:
            if hasattr(msg, 'type'):
                # HumanMessage 或 AIMessage
                role = "user" if msg.type == "human" else "assistant"
                openai_messages.append({"role": role, "content": msg.content})
            else:
                # 其他类型的消息
                openai_messages.append({"role": "user", "content": str(msg)})
        return openai_messages
    
    def _extract_thinking_and_answer(self, response) -> tuple:
        """
        从响应中提取思考过程和最终答案
        
        Returns:
            tuple: (thinking_content, final_answer)
        """
        try:
            # 获取响应内容
            message_content = response.choices[0].message.content
            
            # 检查是否有思考痕迹（DeepSeek的思考通常包含特定的标记）
            if "∴" in message_content or "因为" in message_content or "所以" in message_content:
                # 尝试分割思考过程和最终答案
                lines = message_content.split('\n')
                thinking_lines = []
                answer_lines = []
                
                in_thinking = True
                for line in lines:
                    if any(marker in line for marker in ["最终答案", "答案:", "因此", "所以"]):
                        in_thinking = False
                    if in_thinking:
                        thinking_lines.append(line)
                    else:
                        answer_lines.append(line)
                
                thinking_content = '\n'.join(thinking_lines).strip()
                final_answer = '\n'.join(answer_lines).strip()
                
                # 如果没有明显分割，返回整个内容作为答案
                if not final_answer:
                    final_answer = message_content
                    
                return thinking_content, final_answer
            else:
                # 没有明显思考过程，整个内容作为答案
                return "", message_content
                
        except Exception as e:
            print(f"解析响应时出错: {e}")
            return "", response.choices[0].message.content
    
    def stream_with_thinking(self, messages: List[BaseMessage], **kwargs):
        """
        流式输出思考过程
        """
        openai_messages = self._convert_to_openai_messages(messages)
        
        print("🔄 流式思考过程:")
        print("=" * 40)
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=openai_messages,
                stream=True,
                extra_body={"enable_thinking": True},
                **kwargs
            )
            
            full_content = ""
            for chunk in response:
                if chunk.choices[0].delta.content is not None:
                    content = chunk.choices[0].delta.content
                    full_content += content
                    print(content, end="", flush=True)
            
            print("\n" + "=" * 40)
            return AIMessage(content=full_content)
            
        except Exception as e:
            print(f"流式调用失败: {e}")
            return AIMessage(content=f"调用失败: {str(e)}")



# 运行测试
if __name__ == "__main__":
    # 确保环境变量已设置
    import os
    from dotenv import load_dotenv
    load_dotenv(override=True)
    
    BingXing_API_KEY = os.getenv('BingXing_API_KEY')
    BingXing_BASE_URL = os.getenv('BINGXING_BASE_URL')
    
    if BingXing_API_KEY and BingXing_BASE_URL:
        test_deepseek_reasoner()
    else:
        print("❌ 请先设置 BingXing_API_KEY 和 BINGXING_BASE_URL 环境变量")