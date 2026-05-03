
import os
import json
import yaml
from typing import Dict, Any, Optional
from datetime import datetime

# 假设项目中使用了统一的日志模块，增加真实感
try:
    from rich.console import Console
    console = Console()
except ImportError:
    class Console:
        def print(self, msg, style=None): print(msg)
    console = Console()

class HyperTunerAgent:
    """
    超参编排智能体 (Hyper-Parameter Orchestrator Agent)
    职责: 读取上一轮 YOLOv11 训练的 mAP 与 Loss 指标，
    利用 LLM 进行推理分析，并动态生成下一轮的数据增强策略与 ByteTrack 匹配阈值。
    """

    def __init__(self, model_name: str = "claude-3-5-sonnet", api_key: Optional[str] = None):
        self.model_name = model_name
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self.base_config_path = "configs/default_yolo.yaml"
        
        if not self.api_key:
            console.print("[yellow]Warning: ANTHROPIC_API_KEY 未设置，Agent 将运行在 Mock 模式。[/yellow]")

    def _build_tuning_prompt(self, current_metrics: Dict[str, float], target_class: str) -> str:
        """构建用于指导 LLM 进行超参调优的系统级提示词"""
        
        prompt = f"""
        You are an expert AI agent specializing in Computer Vision model tuning, specifically for YOLOv11 and ByteTrack tracking systems.
        
        Current Iteration Metrics:
        - mAP50-95: {current_metrics.get('map_50_95', 0.0)}
        - Recall on target '{target_class}': {current_metrics.get('target_recall', 0.0)}
        - False Positive Rate (FPR): {current_metrics.get('fpr', 0.0)}
        
        Context: 
        We are optimizing a perception model for a specialized operational vehicle (sprinkler truck) in complex urban environments. 
        The current bottleneck is severe occlusion caused by water mist, leading to ID switches in ByteTrack and low recall in YOLOv11.

        Task:
        Analyze the metrics and adjust the hyperparameters. You must output a valid JSON containing:
        1. 'reasoning': Your step-by-step logic for the parameter changes.
        2. 'hsv_s': New HSV saturation augmentation fraction (current: 0.7).
        3. 'mosaic': New Mosaic augmentation probability (current: 1.0).
        4. 'track_thresh': New ByteTrack tracking threshold (current: 0.5).

        Output JSON ONLY.
        """
        return prompt

    def analyze_and_tune(self, metrics_report: Dict[str, float]) -> Dict[str, Any]:
        """核心执行链路：调用大模型生成新配置"""
        
        console.print(f"[cyan][{datetime.now().strftime('%H:%M:%S')}] HyperTunerAgent: 正在分析验证集指标...[/cyan]")
        
        prompt = self._build_tuning_prompt(metrics_report, target_class="water_mist_occlusion")
        
        # ---------------------------------------------------------------------
        # 在真实代码中，这里是实际调用 LLM API 的地方。
        # 例如: response = anthropic_client.messages.create(messages=[{"role": "user", "content": prompt}])
        # 为了演示和申请展示，这里模拟 LLM 的结构化 JSON 返回
        # ---------------------------------------------------------------------
        
        # Mock LLM Response
        mock_llm_response = {
            "reasoning": "The low recall combined with water mist occlusion suggests the model is relying too much on color/saturation which is distorted by water. Decreasing 'hsv_s' will force the model to learn shape features. Reducing 'mosaic' slightly prevents overly complex synthetic scenes that confuse the baseline. Lowering 'track_thresh' helps maintain IDs when confidence drops temporarily due to mist.",
            "hsv_s": 0.45,
            "mosaic": 0.8,
            "track_thresh": 0.4
        }
        
        console.print(f"[green]LLM 推理完成:[/green] {mock_llm_response['reasoning']}")
        return mock_llm_response

    def update_yaml_config(self, new_params: Dict[str, Any], output_path: str = "configs/auto_tuned.yaml"):
        """将 LLM 决策出的参数写入新的 YAML 配置文件，供下一轮训练读取"""
        
        # 读取基础配置
        if os.path.exists(self.base_config_path):
            with open(self.base_config_path, 'r') as f:
                config = yaml.safe_load(f)
        else:
            config = {"model": {}, "tracker": {}, "augmentation": {}}

        # 注入 LLM 生成的新参数
        config.setdefault('augmentation', {})['hsv_s'] = new_params['hsv_s']
        config.setdefault('augmentation', {})['mosaic'] = new_params['mosaic']
        config.setdefault('tracker', {})['track_thresh'] = new_params['track_thresh']

        # 保存新配置
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
            
        console.print(f"[blue]已生成下一轮迭代配置文件: {output_path}[/blue]")

if __name__ == "__main__":
    # 测试代码，展示智能体工作流
    agent = HyperTunerAgent()
    
    # 模拟从上一个训练 Epoch 获取的性能指标
    mock_metrics = {
        "map_50_95": 0.42,
        "target_recall": 0.31,
        "fpr": 0.15
    }
    
    # 1. 智能体分析指标并决定参数调整方向
    optimized_params = agent.analyze_and_tune(mock_metrics)
    
    # 2. 智能体将决策落地为物理配置文件
    agent.update_yaml_config(optimized_params)
    
    console.print("[bold green]HyperTuner Agent 任务执行完毕，等待 Training Agent 接管。[/bold green]")
