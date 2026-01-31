# 抽象基类与类型
from abc import abstractmethod, ABC
from typing import Dict, Union, List
import torch
import pytorch_lightning as pl

from navsim.common.dataclasses import AgentInput, Trajectory, SensorConfig
from navsim.planning.training.abstract_feature_target_builder import AbstractFeatureBuilder, AbstractTargetBuilder


class AbstractAgent(torch.nn.Module, ABC):
    """Agent 抽象基类：定义训练/推理所需接口。"""
    def __init__(
        self,
        requires_scene: bool = False,
    ):
        """
        抽象智能体基类的构造函数
        
        Args:
            requires_scene: 是否需要场景信息，默认为False
                          当为True时，表示该智能体在推理时需要完整的场景数据
        """
        super().__init__()
        self.requires_scene = requires_scene

    @abstractmethod
    def name(self) -> str:
        """
        :return: 智能体名称字符串。
        """
        pass
    
    @abstractmethod
    def get_sensor_config(self) -> SensorConfig:
        """
        :return: 传感器配置（相机与 LiDAR 等）。
        """
        pass

    @abstractmethod
    def initialize(self) -> None:
        """
        初始化智能体（如加载权重等）。
        """
        pass

    def forward(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        模型前向。
        :param features: 特征字典。
        :return: 预测结果字典。
        """
        raise NotImplementedError

    def get_feature_builders(self) -> List[AbstractFeatureBuilder]:
        """
        :return: 特征构建器列表。
        """
        raise NotImplementedError("No feature builders. Agent does not support training.")

    def get_target_builders(self) -> List[AbstractTargetBuilder]:
        """
        :return: 目标构建器列表。
        """
        raise NotImplementedError("No target builders. Agent does not support training.")

    def compute_trajectory(self, agent_input: AgentInput) -> Trajectory:
        """
        计算自车未来轨迹。
        :param current_input: 智能体输入数据结构。
        :return: 预测的未来轨迹。
        """
        # 推理模式（不更新梯度）
        self.eval()
        features : Dict[str, torch.Tensor] = {}
        # 构建特征
        for builder in self.get_feature_builders():
            features.update(builder.compute_features(agent_input))

        # 添加 batch 维
        features = {k: v.unsqueeze(0) for k, v in features.items()}

        # 前向推理
        with torch.no_grad():
            predictions = self.forward(features)
            poses = predictions["trajectory"].squeeze(0).numpy()

        # 组装轨迹结果
        return Trajectory(poses)
    
    def compute_trajectory_gpu(self, agent_input: AgentInput) -> Trajectory:
        """
        计算自车未来轨迹（GPU）。
        :param current_input: 智能体输入数据结构。
        :return: 预测的未来轨迹。
        """
        # 推理模式（GPU）
        self.eval()
        features : Dict[str, torch.Tensor] = {}
        # 构建特征
        for builder in self.get_feature_builders():
            features.update(builder.compute_features(agent_input))

        # add batch dimension + 移动到 GPU
        features = {k: v.unsqueeze(0).to('cuda') for k, v in features.items()}

        # 前向推理
        with torch.no_grad():
            predictions = self.forward(features)
            poses = predictions["trajectory"].squeeze(0).cpu().numpy()

        # 组装轨迹结果
        return Trajectory(poses)
    
    def compute_loss(
        self,
        features: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        predictions: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        根据特征、目标与预测计算反向传播的损失。
        """
        raise NotImplementedError("No loss. Agent does not support training.")
    
    def get_optimizers(
        self
    ) -> Union[
        torch.optim.Optimizer,
        Dict[str, Union[
            torch.optim.Optimizer,
            torch.optim.lr_scheduler.LRScheduler]
        ]
    ]:
        """
        返回用于 Lightning Trainer 的优化器配置。
        可以是单个优化器，也可以是包含优化器与学习率调度器的字典。
        """
        raise NotImplementedError("No optimizers. Agent does not support training.")
    
    def get_training_callbacks(
        self
    ) -> List[pl.Callback]:
        """
        返回训练用的 PyTorch Lightning 回调列表。
        可参考 navsim.planning.training.callbacks。
        """
        return []
