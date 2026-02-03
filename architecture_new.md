---
config:
  layout: elk
  look: neo
---
flowchart TB
 subgraph s2["输入层"]
        A3["相机特征 张量 3 256 1024"]
        A2["裁剪 拼接 缩放 大小 1024 256"]
        A1["相机图像 输入 cam_l0 cam_f0 cam_r0 当前帧"]
        B4["LiDAR 特征 张量"]
        B3["BEV直方图 网格"]
        B1["LiDAR 点云 输入 当前帧"]
        C3["状态特征 张量 8"]
        C2["驾驶指令 4 速度 2 加速度 2"]
        C1["自车状态 当前帧"]
  end
 subgraph s3["监督目标"]
        T2["轨迹目标 张量 B 1 8 3"]
        T1["未来轨迹 采样 num_poses 由 time_horizon 4 和 interval 0.5 推导 为 8"]
        T4["仿真奖励目标 张量 B 1 5"]
        T3["预计算仿真奖励 use_sim_reward True sim_reward_dict_path"]
        T6["车辆检测目标 状态与标签"]
        T5["当前帧标注"]
        T8["语义BEV目标"]
        T7["当前帧标注 与 地图"]
        T10["未来车辆目标 状态与标签"]
        T9["未来帧标注 变换到当前自车坐标"]
        T12["未来语义BEV目标"]
        T11["未来标注 与 采样轨迹锚点"]
        T14["sampled_trajs_index"]
        T13["采样轨迹索引 num_sampled_trajs 1"]
  end
 subgraph s4["主干网络与BEV表示"]
        D1["TransfuserBackbone 输入 相机与激光"]
        D2["backbone_bev_feature 通道 512 高 8 宽 8"]
        D3["一乘一卷积 降到 tf_d_model 256"]
        D4["展平 HW 形成 token 数 64"]
        D5["keyval_embedding 数量 num_keyval 64"]
        D6["bev_tokens_per_scene @t 形状 B 64 256"]
  end
 subgraph subGraph4["Trajectory Anchors 与自车编码"]
        E1a["维度说明 256 是轨迹簇数 8 是时间步数 3 是每步状态 x y heading"]
        E1["离线聚类 trajectory anchors 文件 cluster_file_path numpy npy 结构 256 8 3 数值为自车坐标 x y heading"]
        E2["traj_anchors_global 张量 num_traj 256 num_poses 8 3"]
        E3["MLP 输入 24 输出 hidden_dim 256"]
        E4["Transformer Encoder 层数 2 头数 8 dim 256"]
        E5["线性层 状态编码到 tf_d_model 256"]
        E6["轨迹锚点特征 B 256 256"]
        E7["自车状态特征 B 1 256"]
        E8["按轨迹复制 到 B 256 256"]
        E9["拼接 自车与轨迹特征"]
        E10["encode_ego_feat_mlp 得到 ego_feat_per_traj @t B 256 1 256"]
  end
 subgraph s5["轨迹偏移分支"]
        F1["offset_tf_decoder Transformer Decoder 层数 3 头数 8 FFN 1024 dropout 0"]
        F2["TrajectoryOffsetHead 输出 轨迹偏移 B 256 8 3"]
        F3["offset_score_head 输出 轨迹偏移奖励 B 256 softmax"]
  end
 subgraph subGraph6["Latent World Model"]
        G1["BEV token 按轨迹复制 B 256 64 256"]
        G2["注入当前自车到 BEV 空间 @t"]
        G3["拼接 ego token 与 BEV tokens @t"]
        G4["scene_position_embedding 大小 num_plan_queries 64 加 1"]
        G5["latent_world_model Transformer Encoder 层数 2 头数 8 dim 256"]
        G6["未来自车特征 fut_ego_feat @t+Δ"]
        G7["未来 BEV tokens @t+Δ"]
        G8["注入未来自车到 BEV 依据轨迹锚点坐标 @t+Δ"]
        G9["for-loop 迭代 num_fut_timestep 1 输入输出 token 形状一致 参数共享"]
  end
 subgraph s6["奖励特征构建"]
        H1["收集各时间 BEV tokens"]
        H2["收集各时间 ego 特征"]
        H3["按通道拼接 多时刻 BEV 输入通道 num_fut_timestep 1 加 1 乘 256 等于 512"]
        H4["RewardConvNet 两层卷积 输出通道 256"]
        H5["reward_conv_output"]
        H6["拼接 ego 特征 与 reward_conv_output"]
        H7["reward_cat_head 输出 reward_feature_per_traj B 256 256"]
        H8["reward 含义 聚合 ego 动态 BEV 未来占用 与 场景一致性"]
  end
 subgraph s7["奖励头与组合"]
        I1["模仿奖励头 输出 im_rewards softmax"]
        I2["五个指标奖励头 输出 sim_rewards sigmoid"]
        I3["对数空间加权求和"]
        I4["final_rewards"]
  end
 subgraph subGraph10["车辆检测分支 代码默认启用"]
        J1["agent_query_embedding 数量 num_bounding_boxes 30"]
        J2["agent_tf_decoder 层数 3 头数 8 FFN 1024 dropout 0"]
        J3["AgentHead 输出 状态 5 维 与 置信度"]
        J4["agent_states B 30 5 agent_labels B 30"]
  end
 subgraph subGraph11["语义BEV分支 use_map_loss True"]
        K1["bev_upscale 与 BEVUpsampleHead"]
        K2["bev_semantic_head"]
        K3["bev_semantic_map 预测 类别 8 尺寸 128 256"]
  end
 subgraph subGraph12["未来语义BEV分支 仅训练"]
        L1["按 sampled_trajs_index 选择轨迹"]
        L2["new_scene_bev_feature_pos_embed"]
        L3["bev_upscale 与 BEVUpsampleHead"]
        L4["bev_semantic_head"]
        L5["fut_bev_semantic_map 预测 类别 8 尺寸 128 256"]
  end
 subgraph s8["训练输出"]
        M1["trajectory_offset"]
        M2["trajectory_offset_rewards 训练专用"]
        M3["im_rewards"]
        M4["sim_rewards"]
        M5["agent_states 与 agent_labels"]
        M6["bev_semantic_map"]
        M7["fut_bev_semantic_map 训练专用"]
        M8["traj_anchors_global"]
  end
 subgraph s9["损失函数"]
        N1["轨迹偏移损失 WTA L1 权重 1.0"]
        N2["偏移模仿奖励损失 软标签交叉熵 权重 0.1"]
        N3["模仿奖励损失 软标签交叉熵 权重 1.0"]
        N4["仿真奖励损失 BCE 五指标 权重 1.0"]
        N5["车辆检测损失 分类 BCE 与 回归 L1 权重未在配置中定义"]
        N6["语义BEV损失 Focal alpha 0.5 gamma 2.0 权重 10.0"]
        N7["未来语义BEV损失 Focal alpha 0.5 gamma 2.0 权重 0.1"]
  end
 subgraph subGraph15["推理输出 推理专用"]
        P1["轨迹锚点加偏移 得到候选轨迹"]
        P2["取最大 final_rewards"]
        P3["select_best_trajectory 输出 轨迹 B 8 3"]
  end
 subgraph s10["轨迹几何语义"]
        Q1["Anchor 离线轨迹原型"]
        Q2["Offset 小幅修正"]
        Q3["Refined 候选轨迹集合"]
  end
    A1 --> A2
    A2 --> A3
    B3 --> B4
    C1 --> C2
    C2 --> C3
    T1 --> T2
    T3 --> T4
    T5 --> T6
    T7 --> T8
    T9 --> T10
    T11 --> T12
    T13 --> T14
    A3 --> D1
    B4 --> D1
    D1 --> D2
    D2 --> D3
    D3 --> D4
    D4 --> D5
    D5 --> D6
    E1 --> E1a
    E1a --> E2
    E2 --> E3 & M8 & Q1
    E3 --> E4
    C3 --> E5
    E4 --> E6
    E5 --> E7
    E7 --> E8
    E8 --> E9
    E9 --> E10
    E10 --> F1 & G2
    D6 --> F1 & G1 & J1 & J2 & K1
    F1 --> F2 & F3
    G1 --> G2
    G2 --> G3
    G3 --> G4
    G4 --> G5
    G5 --> G6 & G7
    G6 --> G8
    G7 --> G8
    G8 --> G9
    G9 --> H1 & H2 & L1
    H1 --> H3
    H3 --> H4
    H4 --> H5
    H2 --> H6
    H6 --> H7
    H7 --> H8 & I1 & I2
    I1 --> I3 & M3
    I2 --> I3 & M4
    I3 --> I4
    J1 --> J2
    J2 --> J3
    J3 --> J4
    K1 --> K2
    K2 --> K3
    L1 --> L2
    L2 --> L3
    L3 --> L4
    L4 --> L5
    F2 --> M1 & P1 & Q2
    F3 --> M2
    J4 --> M5
    K3 --> M6
    L5 --> M7
    M1 --> N1
    M2 --> N2
    M3 --> N3
    M4 --> N4
    M5 --> N5
    M6 --> N6
    M7 --> N7
    I4 --> P2
    P1 --> P3 & Q3
    P2 --> P3
    Q1 --> Q2
    Q2 --> Q3
    T2 --> N1 & N2 & N3
    T4 --> N4
    T6 --> N5
    T8 --> N6
    T12 --> N7
    T14 --> L1
    B1 --> B3