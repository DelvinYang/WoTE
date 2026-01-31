# WorkerPool 构建工具
import logging

from hydra.utils import instantiate
from omegaconf import DictConfig

from nuplan.planning.script.builders.utils.utils_type import is_target_type, validate_type
from nuplan.planning.utils.multithreading.worker_pool import WorkerPool
from nuplan.planning.utils.multithreading.worker_parallel import SingleMachineParallelExecutor
from nuplan.planning.utils.multithreading.worker_sequential import Sequential

# 日志
logger = logging.getLogger(__name__)


def build_worker(cfg: DictConfig) -> WorkerPool:
    """
    Builds the worker.
    :param cfg: DictConfig. Configuration that is used to run the experiment.
    :return: Instance of WorkerPool.
    """
    logger.info('Building WorkerPool...')
    # Parallel / Sequential 两种 worker 的构建方式不同
    worker: WorkerPool = (
        instantiate(cfg.worker)
        if (
            is_target_type(cfg.worker, SingleMachineParallelExecutor)
            or is_target_type(cfg.worker, Sequential)
        )
        else instantiate(cfg.worker, output_dir=cfg.output_dir)
    )
    # 确保类型正确
    validate_type(worker, WorkerPool)

    logger.info('Building WorkerPool...DONE!')
    return worker
