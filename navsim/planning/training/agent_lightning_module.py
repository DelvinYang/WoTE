# Lightning 封装训练流程
import pytorch_lightning as pl
from torch import Tensor
from typing import Dict, Tuple
import torch
from navsim.agents.abstract_agent import AbstractAgent

class AgentLightningModule(pl.LightningModule):
    """将 Agent 封装为 LightningModule 以统一训练/验证流程。"""
    def __init__(
        self,
        agent: AbstractAgent,
    ):
        super().__init__()
        self.agent = agent

    def _step(
        self,
        batch: Tuple[Dict[str, Tensor], Dict[str, Tensor]],
        logging_prefix: str,
    ):
        """共享的 train/val 步骤逻辑。"""
        features, targets = batch

        # 某些模型需要同时输入 targets
        input_target = self.agent.config.input_target if hasattr(self.agent.config, 'input_target') else False
        if input_target:
            prediction = self.agent.forward(features, targets)
        else:
            prediction = self.agent.forward(features)

        # 计算损失（可返回 dict 或 Tensor）
        loss_dict = self.agent.compute_loss(features, targets, prediction)
        if isinstance(loss_dict, Tensor):
            loss_dict = {"traj_loss": loss_dict}
            
        total_loss = 0.0
        for loss_key, loss_value in loss_dict.items():
            # 记录日志，acc 类指标不参与总 loss
            self.log(f"{logging_prefix}/{loss_key}", loss_value, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
            if 'acc' in loss_key:
                continue
            total_loss = total_loss + loss_value
        return total_loss
    
    def training_step(
        self,
        batch: Tuple[Dict[str, Tensor], Dict[str, Tensor]],
        batch_idx: int
    ):
        """Lightning 训练步骤。"""
        return self._step(batch, "train")

    def validation_step(
        self,
        batch: Tuple[Dict[str, Tensor], Dict[str, Tensor]],
        batch_idx: int
    ):
        """Lightning 验证步骤。"""
        return self._step(batch, "val")

    def configure_optimizers(self):
        """返回优化器与调度器配置。"""
        return self.agent.get_optimizers()
    
    def backward(self, loss):
        # print('set detect anomaly')
        # torch.autograd.set_detect_anomaly(True)
        # 自定义 backward（保留注释的异常检测选项）
        loss.backward()
