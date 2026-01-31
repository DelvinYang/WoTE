# WoTE 特征构建器
from typing import Any, List, Dict, Union
import torch
import numpy as np
from torchvision import transforms

# 依赖的枚举与采样
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from navsim.common.enums import BoundingBoxIndex, LidarIndex

# 数据结构与基类
from navsim.agents.abstract_agent import AbstractAgent
from navsim.common.dataclasses import AgentInput, SensorConfig
from navsim.planning.training.abstract_feature_target_builder import (
    AbstractFeatureBuilder,
    AbstractTargetBuilder,
)
from navsim.common.dataclasses import Scene
import timm, cv2

class WoTEFeatureBuilder(AbstractFeatureBuilder):
    """WoTE 特征构建：相机拼接图像 + LiDAR 直方图 + 车辆状态。"""
    def __init__(
            self,
            slice_indices=[3],
            config=None,
        ):
        self.slice_indices = slice_indices
        self._config = config

    def get_unique_name(self) -> str:
        return "wote_feature"

    def compute_features(self, agent_input: AgentInput) -> Dict[str, torch.Tensor]:
        """Inherited, see superclass."""
        features = {}

        # 相机 + LiDAR + 状态
        features["camera_feature"] = self._get_camera_feature(agent_input)
        features["lidar_feature"] = self._get_lidar_feature(agent_input)
        features["status_feature"] = torch.concatenate(
            [
                torch.tensor(agent_input.ego_statuses[-1].driving_command, dtype=torch.float32),
                torch.tensor(agent_input.ego_statuses[-1].ego_velocity, dtype=torch.float32),
                torch.tensor(agent_input.ego_statuses[-1].ego_acceleration, dtype=torch.float32),
            ],
        )

        return features
    
    def _get_camera_feature(self, agent_input: AgentInput) -> torch.Tensor:
        """
        Extract stitched camera from AgentInput
        :param agent_input: input dataclass
        :return: stitched front view image as torch tensor
        """

        cameras = agent_input.cameras[-1]

        # 裁剪三路前向相机，保证 4:1 视野比例
        l0 = cameras.cam_l0.image[28:-28, 416:-416]
        f0 = cameras.cam_f0.image[28:-28]
        r0 = cameras.cam_r0.image[28:-28, 416:-416]

        # 拼接为单张宽图
        stitched_image = np.concatenate([l0, f0, r0], axis=1)
        resized_image = cv2.resize(stitched_image, (1024, 256))
        tensor_image = transforms.ToTensor()(resized_image)

        return tensor_image

    def _get_lidar_feature(self, agent_input: AgentInput) -> torch.Tensor:
        """
        Compute LiDAR feature as 2D histogram, according to Transfuser
        :param agent_input: input dataclass
        :return: LiDAR histogram as torch tensors
        """

        # 只取 (x,y,z)，并转为 (N,3)
        lidar_pc = agent_input.lidars[-1].lidar_pc[LidarIndex.POSITION].T

        # NOTE: Code from
        # https://github.com/autonomousvision/carla_garage/blob/main/team_code/data.py#L873
        def splat_points(point_cloud):
            # 生成固定网格的 BEV 直方图
            xbins = np.linspace(
                self._config.lidar_min_x,
                self._config.lidar_max_x,
                (self._config.lidar_max_x - self._config.lidar_min_x)
                * int(self._config.pixels_per_meter)
                + 1,
            )
            ybins = np.linspace(
                self._config.lidar_min_y,
                self._config.lidar_max_y,
                (self._config.lidar_max_y - self._config.lidar_min_y)
                * int(self._config.pixels_per_meter)
                + 1,
            )
            hist = np.histogramdd(point_cloud[:, :2], bins=(xbins, ybins))[0]
            hist[hist > self._config.hist_max_per_pixel] = self._config.hist_max_per_pixel
            overhead_splat = hist / self._config.hist_max_per_pixel
            return overhead_splat

        # 去除过高点并按高度分层
        lidar_pc = lidar_pc[lidar_pc[..., 2] < self._config.max_height_lidar]
        below = lidar_pc[lidar_pc[..., 2] <= self._config.lidar_split_height]
        above = lidar_pc[lidar_pc[..., 2] > self._config.lidar_split_height]
        above_features = splat_points(above)
        if self._config.use_ground_plane:
            below_features = splat_points(below)
            features = np.stack([below_features, above_features], axis=-1)
        else:
            features = np.stack([above_features], axis=-1)
        features = np.transpose(features, (2, 0, 1)).astype(np.float32)

        return torch.tensor(features)
    
