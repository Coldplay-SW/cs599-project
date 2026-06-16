# cs599-project

## 项目简介
一个基于 LangGraph 和 DeepSeek 大模型构建的智能医疗诊断辅助系统。该项目模拟了完整的医疗诊断流程，从初步评估到最终报告生成，为医生和患者提供初步的诊断参考和治疗建议。

## 方向
方向一：Agentic AI 原生开发

## 技术栈
- **后端框架**: LangGraph (工作流编排)
- **AI 模型**: DeepSeek API
- **前端界面**: Streamlit
- **编程语言**: Python 3.8+

## 目录结构
- `app.py`: Streamlit 主程序入口，负责渲染用户界面及处理前端交互逻辑。
- `requirements.txt`: 项目依赖包清单，包含 LangGraph、DeepSeek SDK 等必要库。
- `.env`: 环境变量配置文件，用于安全存储 API Key 等敏感凭证。

## 环境搭建
1. 依赖安装
在项目src 目录下打开终端，执行以下命令安装所需依赖：
```bash
pip install -r requirements.txt

```
2. 环境变量配置
在项目src目录创建 `.env` 文件，并填入你的 DeepSeek API Key：
```
DEEPSEEK_API="your_deepseek_api_key_here"
```
请将 `your_deepseek_api_key_here` 替换为您自己的 API 密钥。

3. 启动步骤
确保你在包含 app.py 的目录下，运行以下命令启动应用：
    ```bash
    streamlit run app.py
    ```
应用启动后，它会自动在您的默认浏览器中打开 `http://localhost:8501`。

## 项目状态
- [x] Proposal
- [x] MVP
- [x] Final
