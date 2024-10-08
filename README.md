# 简介
多样的大模型推理工具，包含
1. 离线推理：
   - **本地大模型**：基于vLLM引擎对本地文件夹内prompt list进行批处理，速度最快，但每次处理需要启动时间。
   - **API接口**：使用OpenAI标准接口，但无法进行批处理。
2. 在线推理：问答式，使用进程通信

# 入口参数
- offline_online    离线批处理还是在线QA
- local_api         使用本地权重还是远程API
- model_path        本地模型路径，或者远程模型名
- gpu_list_str      GPU列表，可选参数