---
config:
  layout: elk
---
flowchart TB

%% ===============================
%% 输入层
%% ===============================
subgraph s0["0 输入层"]
  I1["camera_feature 相机特征<br>shape 3 256 1024"]
  I2["lidar_feature LiDAR特征<br>shape C 256 256"]
  I3["status_feature 自车状态拼接(驾驶指令4+速度2+加速度2)<br>shape 8"]
end

%% ===============================
%% 1. 离线聚类轨迹锚点
%% ===============================
subgraph s1["1 轨迹锚点与编码"]
  A1["trajectory_anchors 轨迹锚点<br>shape 256 8 3"]
  A2("unsqueeze_repeat_batch")
  A3["init_trajectory_anchor 初始锚点<br>shape B 256 8 3"]
  A4("flatten_anchors")
  A5["traj_anchors_flat 展平锚点<br>shape B 256 24"]
  A6("mlp_planning_vb")
  A7["trajectory_anchors_feat_pre 锚点MLP特征<br>shape B 256 256"]
  A8("cluster_encoder")
  A9["trajectory_anchors_feat 锚点编码特征<br>shape B 256 256"]

  A1 --> A2 --> A3 --> A4 --> A5 --> A6 --> A7 --> A8 --> A9
end

%% ===============================
%% 2. 当前时刻自车状态编码
%% ===============================
subgraph s2["2 自车状态编码"]
  B1("status_encoding_linear")
  B2["status_encoding 状态编码<br>shape B 256"]
  B3("unsqueeze_status")
  B4["ego_status_feat 自车编码特征<br>shape B 1 256"]

  B1 --> B2 --> B3 --> B4
end

%% ===============================
%% 3. 轨迹条件化特征构建
%% ===============================
subgraph s3["3 轨迹条件化特征构建"]
  C1("repeat_ego_status_feat")
  C2["ego_status_feat_repeat 轨迹级自车特征<br>shape B 256 256"]
  C3("concat_ego_and_traj")
  C4["ego_traj_concat 轨迹条件化特征<br>shape B 256 512"]
  C5("encode_ego_feat_mlp")
  C6["ego_feat_per_traj 轨迹级ego特征<br>shape B 256 256"]
  C7("unsqueeze_token_dim")
  C8["ego_feat 轨迹级ego token<br>shape B 256 1 256"]

  B4 --> C1 --> C2
  A9 --> C3
  C2 --> C3 --> C4 --> C5 --> C6 --> C7 --> C8
end

I3 --> B1

%% ===============================
%% 4. 主干网络与 BEV 表示
%% ===============================
subgraph s4["4 主干网络与BEV表示"]
  D1("TransfuserBackbone")
  D2["backbone_bev_feature 主干BEV特征<br>shape B 512 8 8"]
  D3("bev_downscale_1x1")
  D4("flatten_8x8")
  D5["bev_feature BEV特征<br>shape B 64 256"]

  D6["keyval_embedding BEV身份嵌入<br>shape 64 256"]
  D7("add_keyval_embedding")
  D8["flatten_bev_feature BEV tokens<br>shape B 64 256"]

  I1 --> D1
  I2 --> D1
  D1 --> D2 --> D3 --> D4 --> D5
  D5 --> D7 --> D8
  D6 --> D7
end

%% ===============================
%% 5. Latent World Model
%% ===============================
subgraph s5["5 Latent World Model"]
  E1("copy_bev_by_traj 每一条采样轨迹都需要当前场景的 BEV 特征")
  E2["flatten_bev_feature_multi_trajs 逐轨迹BEV<br>shape B 256 64 256"]
  E3("inject_cur_ego_into_bev")
  E4["flatten_bev_feature_injected 注入ego后BEV<br>shape B 256 64 256"]
  E5("reshape_bev_for_lwm")
  E6["bev_tokens_lwm BEV tokens<br>shape BxT 64 256"]

  E7("reshape_ego_for_lwm")
  E8["ego_feat_lwm ego token<br>shape BxT 1 256"]

  E9("concat_ego_bev_tokens")
  E10["scene_feature 条件序列<br>shape BxT 65 256"]
  E11("add_scene_pos_embedding")
  E12["scene_feature_pos 加位置编码<br>shape BxT 65 256"]
  E13("latent_world_model_encoder loop_num_fut_timestep")
  E14["fut_ego_feat 未来ego token<br>shape BxT 1 256"]
  E15["fut_flatten_bev_feature 未来BEV tokens<br>shape BxT 64 256"]
  E16("inject_fut_ego_into_bev")
  E17["fut_flatten_bev_feature_injected 未来BEV注入ego<br>shape BxT 64 256"]

  D8 --> E1 --> E2 --> E3 --> E4 --> E5 --> E6
  C8 --> E7 --> E8
  E6 --> E9
  E8 --> E9 --> E10 --> E11 --> E12 --> E13
  E13 --> E14
  E13 --> E15 --> E16 --> E17
end

%% ===============================
%% 6. 车辆检测分支 训练辅助
%% ===============================
subgraph s6["6 车辆检测分支 训练辅助"]
  F1("agent_query_embedding")
  F2["agent_query agent查询<br>shape B 30 256"]
  F3("agent_tf_decoder")
  F4("agent_head")
  F5["agent_states 车辆状态<br>shape B 30 5"]
  F6["agent_labels 车辆存在标签<br>shape B 30"]

  F1 --> F2 --> F3
  D8 --> F3
  F3 --> F4 --> F5
  F4 --> F6
end

%% ===============================
%% 7. 轨迹偏移分支
%% ===============================
subgraph s7["7 轨迹偏移分支"]
  G1("squeeze_ego_feat")
  G2["ego_feat_squeezed ego特征<br>shape B 256 256"]
  G3("offset_tf_decoder")
  G4("TrajectoryOffsetHead")
  G5("offset_score_head")
  G6["trajectory_offset 轨迹偏移<br>shape B 256 8 3"]
  G7["trajectory_offset_rewards 轨迹评分<br>shape B 256"]

  C8 --> G1 --> G2
  G2 --> G3
  D8 --> G3
  G3 --> G4 --> G6
  G3 --> G5 --> G7
end

%% ===============================
%% 8. 奖励特征与奖励头
%% ===============================
subgraph s8["8 奖励特征与奖励头"]
  R1("collect_bev_over_time")
  R2["bev_feat_list 多时刻BEV序列"]
  R3("reward_convnet")
  R4["reward_conv_output<br>shape BxT 1 256"]

  R5("collect_ego_over_time")
  R6["ego_feat_list 多时刻ego序列"]

  R7("reward_cat_head")
  R8["reward_feature 轨迹reward特征<br>shape B 256 256"]

  R9("reward_head")
  R10["im_rewards 模仿奖励<br>shape B 256"]

  R11("sim_reward_heads")
  R12["sim_rewards 指标奖励<br>shape B 5 256"]

  R13("weighted_reward_calculation")
  R14["final_rewards 最终奖励<br>shape B 256"]

  E15 --> R1 --> R2 --> R3 --> R4
  E14 --> R5 --> R6
  R4 --> R7
  R6 --> R7 --> R8
  R8 --> R9 --> R10
  R8 --> R11 --> R12
  R10 --> R13
  R12 --> R13 --> R14
end

%% ===============================
%% 9. 最终轨迹生成
%% ===============================
subgraph s9["9 最终轨迹生成"]
  H1("add_anchor_offset")
  H2["refined_traj 精修轨迹<br>shape B 256 8 3"]
  H3("argmax_reward")
  H4("select_best_traj")
  H5["best_traj 最优轨迹<br>shape B 8 3"]

  A1 --> H1
  G6 --> H1 --> H2
  R14 --> H3
  H2 --> H4
  H3 --> H4 --> H5
end

%% ===============================
%% 10. 未来语义 BEV 分支 训练辅助
%% ===============================
subgraph s10["10 未来语义BEV分支 训练辅助"]
  S0["sampled_trajs_index 采样索引<br>shape B K"]
  S1("sample_future_bev_feature")
  S2["new_scene_bev_feature 采样BEV特征<br>shape BxK 64 256"]
  S3["new_scene_bev_feature_pos_embed 位置嵌入<br>shape 64 256"]
  S4("add_new_scene_pos")
  S5["new_scene_bev_feature_with_pos 加位置BEV<br>shape BxK 64 256"]
  S6("BEVUpsampleHead")
  S7("bev_semantic_head")
  S8["fut_bev_semantic_map 未来语义BEV<br>shape BxK 8 128 256"]

  E15 --> S1 --> S2 --> S4 --> S5 --> S6 --> S7 --> S8
  S0 --> S1
  S3 --> S4
end

%% ===============================
%% 11. 当前时刻语义 BEV 分支
%% ===============================
subgraph s11["11 当前时刻语义BEV分支"]
  T1("bev_upscale")
  T2("BEVUpsampleHead")
  T3("bev_semantic_head")
  T4["bev_semantic_map 语义BEV@t<br>shape B 8 128 256"]

  D8 --> T1 --> T2 --> T3 --> T4
end
