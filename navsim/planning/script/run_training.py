# 用于类型标注的元组类型
from typing import Tuple
# hydra 用于配置管理与实例化
import hydra
from hydra.utils import instantiate
# 日志与 torch（torch 在此文件中未直接使用，但可能被其他模块需要）
import logging, torch
from omegaconf import DictConfig
from pathlib import Path
import pytorch_lightning as pl
from torch.utils.data import DataLoader

# 数据集与 Lightning 模块
from navsim.planning.training.dataset import CacheOnlyDataset, Dataset
from navsim.planning.training.agent_lightning_module import AgentLightningModule
# 场景加载器与过滤器
from navsim.common.dataloader import SceneLoader
from navsim.common.dataclasses import SceneFilter
# 抽象智能体接口
from navsim.agents.abstract_agent import AbstractAgent

# 当前文件的日志记录器
logger = logging.getLogger(__name__)

# Hydra 配置路径与默认配置名
CONFIG_PATH = "config/training"
CONFIG_NAME = "default_training"

def build_datasets(cfg: DictConfig, agent: AbstractAgent) -> Tuple[Dataset, Dataset]:
    """根据配置与智能体构建训练/验证数据集。"""
    # 训练集过滤器（从配置实例化）
    train_scene_filter: SceneFilter = instantiate(cfg.scene_filter)
    if train_scene_filter.log_names is not None:
        # 如果过滤器中已有 log 列表，则与训练日志取交集
        train_scene_filter.log_names = list(set(train_scene_filter.log_names) & set(cfg.train_logs))
    else:
        # 否则直接使用配置中的训练日志
        train_scene_filter.log_names = cfg.train_logs

    # 验证集过滤器（从配置实例化）
    val_scene_filter: SceneFilter = instantiate(cfg.scene_filter)
    if val_scene_filter.log_names is not None:
        # 如果过滤器中已有 log 列表，则与验证日志取交集
        val_scene_filter.log_names = list(set(val_scene_filter.log_names) & set(cfg.val_logs))
    else:
        # 否则直接使用配置中的验证日志
        val_scene_filter.log_names = cfg.val_logs

    # 数据与传感器缓存路径
    data_path = Path(cfg.navsim_log_path)
    sensor_blobs_path = Path(cfg.sensor_blobs_path)
    # 训练调试开关（可选字段）
    train_debug = cfg.train_debug if hasattr(cfg, "train_debug") else False

    # 训练集场景加载器
    train_scene_loader = SceneLoader(
        sensor_blobs_path=sensor_blobs_path,
        data_path=data_path,
        scene_filter=train_scene_filter,
        sensor_config=agent.get_sensor_config(),
        train_debug=train_debug,
    )

    # 验证集场景加载器
    val_scene_loader = SceneLoader(
        sensor_blobs_path=sensor_blobs_path,
        data_path=data_path,
        scene_filter=val_scene_filter,
        sensor_config=agent.get_sensor_config(),
    )

    # 是否使用未来帧（由 agent 配置决定）
    use_fut_frames = agent.config.use_fut_frames if hasattr(agent.config, "use_fut_frames") else False
    # 训练数据集
    train_data = Dataset(
        scene_loader=train_scene_loader,
        feature_builders=agent.get_feature_builders(),
        target_builders=agent.get_target_builders(),
        cache_path=cfg.cache_path,
        force_cache_computation=cfg.force_cache_computation,
        use_fut_frames=use_fut_frames,
    )

    # 验证数据集
    val_data = Dataset(
        scene_loader=val_scene_loader,
        feature_builders=agent.get_feature_builders(),
        target_builders=agent.get_target_builders(),
        cache_path=cfg.cache_path,
        force_cache_computation=cfg.force_cache_computation,
        use_fut_frames=use_fut_frames,
    )

    # 返回训练/验证数据集
    return train_data, val_data


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME)
def main(cfg: DictConfig) -> None:
    """训练入口：构建数据、模型与 Trainer 并启动训练。"""
    # 固定全局随机种子以保证可复现
    logger.info("Global Seed set to 0")
    pl.seed_everything(0, workers=True)

    # 输出目录
    logger.info(f"Path where all results are stored: {cfg.output_dir}")

    # 构建 Agent
    logger.info("Building Agent")
    agent: AbstractAgent = instantiate(cfg.agent)

    # 构建 Lightning 模块
    logger.info("Building Lightning Module")
    lightning_module = AgentLightningModule(
        agent=agent,
    )

    # 若仅使用缓存（不构建 SceneLoader）
    if cfg.use_cache_without_dataset:
        logger.info("Using cached data without building SceneLoader")
        # 仅缓存模式下，必须禁用强制缓存计算且提供 cache_path
        assert cfg.force_cache_computation==False, "force_cache_computation must be False when using cached data without building SceneLoader"
        assert cfg.cache_path is not None, "cache_path must be provided when using cached data without building SceneLoader"
        # 训练/验证数据直接从缓存构建
        train_data = CacheOnlyDataset(
            cache_path=cfg.cache_path,
            feature_builders=agent.get_feature_builders(),
            target_builders=agent.get_target_builders(),
            log_names=cfg.train_logs,
        )
        val_data = CacheOnlyDataset(
            cache_path=cfg.cache_path,
            feature_builders=agent.get_feature_builders(),
            target_builders=agent.get_target_builders(),
            log_names=cfg.val_logs,
        )
    else:
        # 正常路径：构建 SceneLoader 与完整数据集
        logger.info("Building SceneLoader")
        train_data, val_data = build_datasets(cfg, agent)

    # 构建 DataLoader
    logger.info("Building Datasets")
    train_dataloader = DataLoader(train_data, **cfg.dataloader.params, shuffle=True)
    logger.info("Num training samples: %d", len(train_data))
    val_dataloader = DataLoader(val_data, **cfg.dataloader.params, shuffle=False)
    logger.info("Num validation samples: %d", len(val_data))

    # 构建 Trainer
    logger.info("Building Trainer")
    trainer_params = cfg.trainer.params
    # trainer_params['strategy'] = "ddp_find_unused_parameters_true" # TODO: 分布式参数查找
    trainer = pl.Trainer(
                **trainer_params, 
                callbacks=agent.get_training_callbacks(),
                )

    # 启动训练
    logger.info("Starting Training")
    # ckpt_path = '/data2/yingyan_li/repo/WoTE//exp/training_transfuser_agent/10ep/lightning_logs/version_0/checkpoints/epoch=10-step=3663.ckpt'
    trainer.fit(
        model=lightning_module,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
        # ckpt_path=ckpt_path,
    )

if __name__ == "__main__":
    # 作为脚本执行时的入口
    main()
