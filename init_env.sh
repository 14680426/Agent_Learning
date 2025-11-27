#!/bin/bash
# macOS/Linux环境初始化脚本

echo "正在初始化项目环境..."

# 设置PYTHONPATH环境变量
export PYTHONPATH="$(pwd)/src"
echo "PYTHONPATH已设置为: $PYTHONPATH"

# 激活虚拟环境（如果存在）
if [ -f ".venv/bin/activate" ]; then
    echo "激活虚拟环境..."
    source .venv/bin/activate
fi

echo ""
echo "环境初始化完成！"
echo "现在可以运行项目命令了。"
echo ""
echo "示例："
echo "  运行RAG模块: python src/RAG/rag.py"
echo "  启动开发服务器: langgraph dev"
echo ""