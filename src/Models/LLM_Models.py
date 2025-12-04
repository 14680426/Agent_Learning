# model_manager.py
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# 加载.env文件中的环境变量
load_dotenv()

class ModelManager:
    """模型管理器，用于创建和管理不同的LLM实例"""
    
    def __init__(self):
        self.api_key = os.getenv('BINGXING_API_KEY')
        self.base_url = os.getenv('BINGXING_BASE_URL')
        self.zhipuai_api_key = os.getenv('ZHIPUAI_API_KEY')
        
        if not self.api_key or not self.base_url:
            raise ValueError("请先设置 BINGXING_API_KEY 和 BINGXING_BASE_URL 环境变量")
    
    def get_model(self, model_name: str, **kwargs) -> ChatOpenAI:
        """
        获取指定模型的ChatOpenAI实例
        
        Args:
            model_name: 模型名称，如 "Qwen3-32B", "DeepSeek-R1-0528"
            **kwargs: 其他模型参数，如 temperature, max_tokens 等
            
        Returns:
            ChatOpenAI: 配置好的模型实例
        """
        default_params = {
            "model": model_name,
            "api_key": self.api_key,
            "base_url": self.base_url,
            "default_headers": {"Accept": "application/json"},
        }
        
        # 合并默认参数和自定义参数
        model_params = {**default_params, **kwargs}
        
        return ChatOpenAI(**model_params)
    
    def get_qwen_model(self, **kwargs) -> ChatOpenAI:
        """快速获取Qwen模型实例"""
        # 为Qwen模型添加enable_thinking参数
        default_kwargs = {
            "model_kwargs": {
                "extra_body": {"enable_thinking": False}
            }
        }
        
        # 如果kwargs中已包含model_kwargs，则合并它们
        if "model_kwargs" in kwargs:
            if "extra_body" in kwargs["model_kwargs"]:
                default_kwargs["model_kwargs"]["extra_body"].update(
                    kwargs["model_kwargs"]["extra_body"]
                )
            # 合并其他model_kwargs参数
            default_kwargs["model_kwargs"].update(kwargs["model_kwargs"])
            # 移除kwargs中的model_kwargs以避免重复
            kwargs.pop("model_kwargs")
        
        # 合并传入的参数
        merged_kwargs = {**default_kwargs, **kwargs}
        return self.get_model("Qwen3-32B", **merged_kwargs)
        # return self.get_model("Qwen3-Coder-Plus", **merged_kwargs)
    
    def get_deepseek_model(self, **kwargs) -> ChatOpenAI:
        """快速获取DeepSeek模型实例"""
        return self.get_model("DeepSeek-R1-0528", **kwargs)
    
    def get_zhipuai_model(self, **kwargs):
        """快速获取Zhipuai模型实例"""
        if not self.zhipuai_api_key:
            raise ValueError("请先设置 ZHIPUAI_API_KEY 环境变量")
        from zhipuai import ZhipuAI
        return ZhipuAI(api_key=self.zhipuai_api_key)

