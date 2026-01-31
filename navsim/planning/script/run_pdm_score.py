# 结果统计与进度条
import pandas as pd
from tqdm import tqdm
# 异常打印
import traceback

# Hydra 配置管理
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig

# 路径与类型
from pathlib import Path
from typing import Any, Dict, List, Union, Tuple
from dataclasses import asdict
from datetime import datetime
# 日志与序列化
import logging
import lzma
import pickle
import os
import uuid

# nuPlan 日志与多进程工具
from nuplan.planning.script.builders.logging_builder import build_logger
from nuplan.planning.utils.multithreading.worker_utils import worker_map

# 本项目组件
from navsim.planning.script.builders.worker_pool_builder import build_worker
from navsim.common.dataloader import MetricCacheLoader
from navsim.agents.abstract_agent import AbstractAgent
from navsim.evaluate.pdm_score import pdm_score
from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import (
    PDMSimulator
)
from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorer
from navsim.common.dataloader import SceneLoader, SceneFilter
from navsim.planning.metric_caching.metric_cache import MetricCache
from navsim.common.dataclasses import SensorConfig

# 日志对象
logger = logging.getLogger(__name__)

# Hydra 配置路径与默认配置名
CONFIG_PATH = "config/pdm_scoring"
CONFIG_NAME = "default_run_pdm_score"

@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME)
def main(cfg: DictConfig) -> None:
    """评测入口：构建 worker，按场景分发任务并汇总 PDM 评分。"""
    # 构建日志与 worker pool
    build_logger(cfg)
    worker = build_worker(cfg)

    # Extract scenes based on scene-loader to know which tokens to distribute across workers
    # TODO: infer the tokens per log from metadata, to not have to load metric cache and scenes here
    # 只用 SceneLoader 统计 token 分布，不加载传感器
    scene_loader = SceneLoader(
        sensor_blobs_path=None,
        data_path=Path(cfg.navsim_log_path),
        scene_filter=instantiate(cfg.scene_filter),
        sensor_config=SensorConfig.build_no_sensors(),
    )
    metric_cache_loader = MetricCacheLoader(Path(cfg.metric_cache_path))

    # 仅评测有 metric cache 的 token
    tokens_to_evaluate = list(set(scene_loader.tokens) & set(metric_cache_loader.tokens))
    num_missing_metric_cache_tokens = len(set(scene_loader.tokens) - set(metric_cache_loader.tokens))
    num_unused_metric_cache_tokens = len(set(metric_cache_loader.tokens) - set(scene_loader.tokens))
    if num_missing_metric_cache_tokens > 0:
        logger.warning(f"Missing metric cache for {num_missing_metric_cache_tokens} tokens. Skipping these tokens.")
    if num_unused_metric_cache_tokens > 0:
        logger.warning(f"Unused metric cache for {num_unused_metric_cache_tokens} tokens. Skipping these tokens.")
    logger.info("Starting pdm scoring of %s scenarios...", str(len(tokens_to_evaluate)))
    # 将 token 分组为每个 log 的任务，交给 worker_map
    data_points = [
        {
            "cfg": cfg,
            "log_file": log_file,
            "tokens": tokens_list,
        }
        for log_file, tokens_list in scene_loader.get_tokens_list_per_log().items()
    ]

    single_eval = getattr(cfg, 'single_eval', False)
    # 单线程评测（便于调试）
    if single_eval:
        print("Running single-threaded worker_map")
        score_rows = run_pdm_score(data_points)
    else:
    # 多线程/多进程评测
        score_rows: List[Tuple[Dict[str, Any], int, int]] = worker_map(worker, run_pdm_score, data_points)
    

    # 汇总为 DataFrame
    pdm_score_df = pd.DataFrame(score_rows)
    num_sucessful_scenarios = pdm_score_df["valid"].sum()
    num_failed_scenarios = len(pdm_score_df) - num_sucessful_scenarios
    average_row = pdm_score_df.drop(columns=["token", "valid"]).mean(skipna=True)
    average_row["token"] = "average"
    average_row["valid"] = pdm_score_df["valid"].all()
    pdm_score_df.loc[len(pdm_score_df)] = average_row

    # 写入结果 CSV（带时间戳）
    save_path = Path(cfg.output_dir)
    timestamp = datetime.now().strftime("%Y.%m.%d.%H.%M.%S")
    pdm_score_df.to_csv(save_path / f"{timestamp}.csv")

    logger.info(f"""
        Finished running evaluation.
            Number of successful scenarios: {num_sucessful_scenarios}. 
            Number of failed scenarios: {num_failed_scenarios}.
            Final average score of valid results: {pdm_score_df['score'].mean()}.
            Results are stored in: {save_path / f"{timestamp}.csv"}.
    """)

def run_pdm_score(args: List[Dict[str, Union[List[str], DictConfig]]]) -> List[Dict[str, Any]]:
    """worker 侧评测函数：对给定 token 列表计算 PDM 评分。"""
    node_id = int(os.environ.get("NODE_RANK", 0))
    thread_id = str(uuid.uuid4())
    logger.info(f"Starting worker in thread_id={thread_id}, node_id={node_id}")

    log_names = [a["log_file"] for a in args]
    tokens = [t for a in args for t in a["tokens"]]
    cfg: DictConfig = args[0]["cfg"]

    # 构建 simulator 与 scorer
    simulator: PDMSimulator = instantiate(cfg.simulator)
    scorer: PDMScorer = instantiate(cfg.scorer)
    assert simulator.proposal_sampling == scorer.proposal_sampling, "Simulator and scorer proposal sampling has to be identical"
    # 构建 agent 并切到评测模式
    agent: AbstractAgent = instantiate(cfg.agent)
    agent.initialize()
    agent.is_eval = True

    metric_cache_loader = MetricCacheLoader(Path(cfg.metric_cache_path))
    # 限定本 worker 的 log 与 token
    scene_filter: SceneFilter =instantiate(cfg.scene_filter)
    scene_filter.log_names = log_names
    scene_filter.tokens = tokens
    scene_loader = SceneLoader(
        sensor_blobs_path=Path(cfg.sensor_blobs_path),
        data_path=Path(cfg.navsim_log_path),
        scene_filter=scene_filter,
        sensor_config=agent.get_sensor_config(),
    )

    # 仅评测 cache 中存在的 token
    tokens_to_evaluate = list(set(scene_loader.tokens) & set(metric_cache_loader.tokens))
    pdm_results: List[Dict[str, Any]] = []
    for idx, (token) in enumerate(tokens_to_evaluate):
        logger.info(
            f"Processing scenario {idx + 1} / {len(tokens_to_evaluate)} in thread_id={thread_id}, node_id={node_id}"
        )
        score_row: Dict[str, Any] = {"token": token, "valid": True}
        try:
            # 读取 metric cache
            metric_cache_path = metric_cache_loader.metric_cache_paths[token]
            with lzma.open(metric_cache_path, "rb") as f:
                metric_cache: MetricCache = pickle.load(f)
            
            # 构建 agent 输入（可选未来帧）
            use_future_frames = agent.config.use_fut_frames if hasattr(agent.config, 'use_fut_frames') else False
            agent_input = scene_loader.get_agent_input_from_token(token, use_fut_frames=use_future_frames)
            if agent.requires_scene:
                # 部分模型需要完整 Scene
                scene = scene_loader.get_scene_from_token(token)
                trajectory = agent.compute_trajectory(agent_input, scene)
            else:
                trajectory = agent.compute_trajectory(agent_input)
            
            # PDM 评分
            pdm_result = pdm_score(
                metric_cache=metric_cache,
                model_trajectory=trajectory,
                future_sampling=simulator.proposal_sampling,
                simulator=simulator,
                scorer=scorer,
            )
            score_row.update(asdict(pdm_result))
        except Exception as e:
            logger.warning(f"----------- Agent failed for token {token}:")
            traceback.print_exc()
            score_row["valid"] = False

        pdm_results.append(score_row)
    return pdm_results

if __name__ == "__main__":
    main()
