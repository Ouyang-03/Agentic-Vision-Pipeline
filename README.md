# Agentic-Vision-Pipeline 🚛🤖
> **A Multi-Agent driven automation framework for specialized vehicle perception (YOLOv11 + ByteTrack)**

本项目旨在利用 LLM (Claude-3.5/DeepSeek) 构建一套视觉模型自动化迭代流水线，解决特种车辆在城市作业环境下的感知盲点与数据长尾效应。

### 🌟 核心特性
- **Agentic Data Mining**: 自动从海量原始视频流中检索高价值、低置信度的边缘案例。
- **Auto-Tuning Workflow**: 利用 LLM 实时生成训练配置，自动优化 YOLOv11 空间注意力机制参数。
- **Closed-loop Reflection**: 基于推理结果进行逻辑反思，指导下一轮数据清洗策略。

### 🏗️ 系统架构
1. **Visual Observer**: 执行基础的物体检测与多目标跟踪任务。
2. **Logic Agent**: 接收来自观测层的异常反馈，进行根因分析（Root Cause Analysis）。
3. **Execution Agent**: 动态生成并部署训练代码，实现无人值守的模型演进。

### 📅 项目进度
- [x] 多智能体协作框架搭建
- [x] 基于 LLM 的超参数自动优化模块
- [ ] 边缘侧轻量化部署与实时闭环
