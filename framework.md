%%{init: {
  'theme': 'base',
  'themeVariables': {
    'fontFamily': 'Arial, Helvetica, sans-serif',
    'fontSize': '14px',
    'primaryTextColor': '#212121',
    'lineColor': '#546e7a',
    'background': '#ffffff'
  },
  'flowchart': {
    'defaultRenderer': 'elk',
    'nodeSpacing': 40,
    'rankSpacing': 60,
    'padding': 20,
    'curve': 'linear'
  }
}}%%

flowchart TB

    %% =========================================
    %% 样式定义 (Nature Journal Style)
    %% =========================================
    classDef subgraphContainer fill:#fafafa,stroke:#cfd8dc,stroke-width:1.5px,rx:8,ry:8,color:#455a64,font-weight:bold;
    classDef inputNode fill:#e3f2fd,stroke:#64b5f6,stroke-width:1.2px,rx:4,ry:4;
    classDef processNode fill:#e8f5e9,stroke:#81c784,stroke-width:1.2px,rx:4,ry:4;
    classDef dataNode fill:#fff3e0,stroke:#ffb74d,stroke-width:1.2px,rx:4,ry:4;
    classDef outputNode fill:#f3e5f5,stroke:#ba68c8,stroke-width:1.2px,rx:4,ry:4;
    classDef memoryNode fill:#eceff1,stroke:#b0bec5,stroke-width:1.2px,rx:4,ry:4;

    linkStyle default stroke:#546e7a,stroke-width:1.5px,fill:none;

    %% =========================================
    %% Inputs Subgraph
    %% =========================================
    subgraph s0["Inputs"]
      I1["CameraFeature"]:::inputNode
      I2["LidarFeature"]:::inputNode
      I3["StatusFeature"]:::inputNode
    end

    %% =========================================
    %% Backbone Subgraph
    %% =========================================
    subgraph s1["Backbone"]
      B1("BEVBackbone"):::processNode
      B2["FlattenBevFeature"]:::dataNode
    end

    I1 --> B1
    I2 --> B1
    B1 --> B2

    %% =========================================
    %% Trajectory Sampling Subgraph
    %% =========================================
    subgraph s2["TrajectorySampling"]
      A0("Diffv2PlannerSampling"):::processNode
      A1["TrajectorySamples"]:::dataNode
      A2("EgoTrajEncoder"):::processNode
      A3["EgoFeatToken"]:::dataNode
    end

    I3 --> A2
    A0 --> A1 --> A2 --> A3

    %% =========================================
    %% Offset Branch Subgraph
    %% =========================================
    subgraph s3["OffsetBranch"]
      C1("OffsetHead"):::processNode
      C2["TrajectoryOffset"]:::dataNode
    end

    A3 --> C1
    B2 --> C1
    C1 --> C2

    %% =========================================
    %% Latent World Model Subgraph
    %% =========================================
    subgraph s4["LatentWorldModel"]
      D1("LatentWorldModel"):::processNode
      D2["FutEgoFeat"]:::dataNode
    end

    A3 --> D1
    B2 --> D1
    D1 --> D2

    %% =========================================
    %% Memory Subgraph
    %% =========================================
    subgraph s6["Memory"]
      M1("MemoryWrite"):::processNode
      M2["MemoryPool"]:::memoryNode
      M3("RagRetriever"):::processNode
      M4["RetrievedMemory"]:::dataNode
    end

    B2 --> M1
    A3 --> M1
    B2 --> M3
    M1 --> M2 --> M3 --> M4

    %% =========================================
    %% Reward And Selection Subgraph
    %% =========================================
    subgraph s5["RewardAndSelection"]
      E1("RewardHead"):::processNode
      E2["FinalRewards"]:::dataNode
      E5("AddAnchorOffset"):::processNode
      E6["RefinedTraj"]:::dataNode
      E3("SelectBestTraj"):::processNode
      E4["BestTraj"]:::outputNode
    end

    M4 --> E1
    D2 --> E1 --> E2
    A1 --> E5
    C2 --> E5 --> E6
    E2 --> E3
    E6 --> E3 --> E4

    %% =========================================
    %% Feedback Loop (Cross-subgraph)
    %% =========================================
    E4 --> M1

    %% 应用子图样式
    style s0 fill:#fafafa,stroke:#cfd8dc,stroke-width:1.5px,rx:8,ry:8,color:#455a64
    style s1 fill:#fafafa,stroke:#cfd8dc,stroke-width:1.5px,rx:8,ry:8,color:#455a64
    style s2 fill:#fafafa,stroke:#cfd8dc,stroke-width:1.5px,rx:8,ry:8,color:#455a64
    style s3 fill:#fafafa,stroke:#cfd8dc,stroke-width:1.5px,rx:8,ry:8,color:#455a64
    style s4 fill:#fafafa,stroke:#cfd8dc,stroke-width:1.5px,rx:8,ry:8,color:#455a64
    style s6 fill:#fafafa,stroke:#cfd8dc,stroke-width:1.5px,rx:8,ry:8,color:#455a64
    style s5 fill:#fafafa,stroke:#cfd8dc,stroke-width:1.5px,rx:8,ry:8,color:#455a64