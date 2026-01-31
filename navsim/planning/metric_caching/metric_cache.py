# Metric cache 数据结构
from __future__ import annotations

import lzma
import pickle
from dataclasses import dataclass

from typing import List
from pathlib import Path
from nuplan.planning.simulation.trajectory.interpolated_trajectory import InterpolatedTrajectory
from nuplan.common.actor_state.ego_state import EgoState

from navsim.planning.simulation.planner.pdm_planner.observation.pdm_observation import (
    PDMObservation,
)
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_path import PDMPath
from navsim.planning.simulation.planner.pdm_planner.observation.pdm_occupancy_map import (
    PDMDrivableMap,
)

from nuplan.common.utils.io_utils import save_buffer


@dataclass
class MetricCache:
    """用于 PDM 评测的缓存内容。"""

    file_path: Path
    trajectory: InterpolatedTrajectory
    ego_state: EgoState

    observation: PDMObservation
    centerline: PDMPath
    route_lane_ids: List[str]
    drivable_area_map: PDMDrivableMap

    def dump(self) -> None:
        """将 MetricCache 压缩序列化到 file_path。"""
        # TODO: check if file_path must really be pickled
        pickle_object = pickle.dumps(self, protocol=pickle.HIGHEST_PROTOCOL)
        save_buffer(self.file_path, lzma.compress(pickle_object, preset=0))
