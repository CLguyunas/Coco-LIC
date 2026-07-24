# QI-CTDR 论文实验执行手册

本文档只验证当前已完成的两部分工作，不改变方法主体：

1. QI：观测可信度建模与信息保持选择；
2. CTDR：连续时间 LiDAR 弱方向检测，以及由该方向驱动的真实视觉观测补充。

主轨迹表不设置 `QI+CT-Shadow` 方法，也不把 CT 诊断结果作为主表的一列。
Shadow 仅用于不改轨迹的机理审计；退化检测单独构成实验 E3。

## 1. 实验结论与证据

| 实验 | 核心问题 | 主要对照 | 可支撑的结论 |
|---|---|---|---|
| E1 总体精度 | 完整方法是否改善轨迹 | Original、QI、QI-CTDR | 完整方法的 ATE、RPE、终点漂移收益 |
| E2 QI 消融 | 可信度与信息选择是否互补 | O、Q、I、QI | 两个 QI 子模块的独立贡献与组合收益 |
| E3 CT 检测 | CT 检测是否比离散扫描检测更贴合误差 | CT-Spline、Scan-6DoF | 连续点时刻和样条支撑的必要性 |
| E4 恢复机理 | 弱方向选择是否优于非定向补点 | QI、Random、Global-D、CTDR | 改善来自方向匹配，而非单纯增加视觉点 |
| E5 参数与效率 | 结论是否依赖单一参数，代价是否可接受 | 参数扫描、模块计时 | 稳定区间、观测开销、运行时间开销 |

E3 的轨迹误差只是外部代理，不应在论文中称为“退化真值”。它用于检验：
低信息分数和持续触发是否更集中地对应局部 RPE 增大。若数据集没有可靠 GT，
该序列只能用于内部一致性与触发统计，不能用于误差相关性结论。

## 2. 固定实验纪律

- 所有方法使用同一代码提交、编译选项、数据起止区间和基础传感器参数。
- 主实验采用 SE(3) 对齐，不允许 Sim(3) 尺度对齐；Sim(3) 只能作为附录诊断。
- 每个确定性方法至少重复 3 次，报告均值和标准差。
- Random 使用 5 个固定种子：`0, 1, 2, 3, 4`，报告均值和标准差。
- 先完成预注册的主参数实验，再做参数敏感性；不得按测试集逐序列挑最优参数。
- 每次运行后立即复制输出并写入唯一目录，避免同名 bag 的 CSV 被下一次覆盖。
- `ct_degeneracy.apply` 始终为 `false`。真正的估计器写开关只有
  `ct_visual_recovery.apply_to_estimator`。

建议优先使用具有可靠 GT 的序列完成 E1/E2/E4；当前已知可用的 GEODE
`Tunneling_tunnel4_gamma` 和 `flat_surfaces_smooth` 可纳入。其余
`LiDAR_Degenerate`、`Visual_Challenge`、`degenerate_seq_02`
若无可靠 GT，则纳入 E3 的内部统计和 E4 的 Shadow 信息审计。

## 3. 新增实验开关

在现有 `ct_odometry_geode.yaml` 中使用以下字段：

```yaml
ct_degeneracy:
  enabled: true
  apply: false
  output_csv: true
  detector_mode: continuous_time   # continuous_time | scan_6d
  weak_eigenvalue_ratio: 3.0e-3

ct_visual_recovery:
  enabled: true
  apply_to_estimator: false
  output_csv: true
  selection_policy: weak_subspace  # weak_subspace | random | global_d
  random_seed: 42
  max_additional_visual_observations: 64

experiment_profile:
  enabled: true
```

`scan_6d` 只移除逐点时刻和跨结点支撑：它在统一参考时刻计算完全相同的
LiDAR 几何残差，复用同一关联、权重、特征数量、阈值与三帧迟滞。因此它是
针对“连续时间建模是否必要”的单变量对照。

三种恢复策略使用同一真实视觉候选池。程序先计算 CTDR 在该帧需要的附加数量
\(K\)，然后 Random 和 Global-D 也严格选择 \(K\) 个：

- `weak_subspace`：在 LiDAR 弱子空间最大化边际 log-det；
- `random`：按 `random_seed` 和帧时间戳确定性抽取；
- `global_d`：在完整 6DoF 空间最大化边际 log-det。

输出 CSV 同时保存三种策略在同一帧、同一候选池、同一 \(K\) 下的信息指标。
选择非 `weak_subspace` 时，仅对应策略的索引进入估计器。

## 4. 通用编译、运行和保存

### 4.0 由当前 GEODE 配置生成方法组

先确认现有 `ct_odometry_geode.yaml` 的数据路径、话题、外参和基础 Coco-LIC
参数能够正常运行。随后一次性生成全部方法配置，避免手工漏改开关：

```bash
cd ~/catkin_coco/src/Coco-LIC
python3 tools/generate_qi_ctdr_experiment_configs.py \
  --base config/ct_odometry_geode.yaml \
  --output-dir config/experiments/geode \
  --random-seeds 0,1,2,3,4
```

生成器不修改基础配置，只复制它并改写 QI/CTDR 实验字段。主要文件为
`E1_original.yaml`、`E1_E2_qi.yaml`、`E1_E4_ctdr.yaml`、
`E2_q.yaml`、`E2_i.yaml`、`E3_ct_spline.yaml`、
`E3_scan_6d.yaml`、`E4_shadow.yaml`、`E4_global_d.yaml` 和五个
`E4_random_seed_*.yaml`。运行时把对应文件传给 `config_path`；若 launch
要求 `/config` 下的相对路径，可将输出目录设为 `config/experiments/geode`
并使用 `/config/experiments/geode/文件名.yaml`。

### 4.1 编译

```bash
source /opt/ros/noetic/setup.bash
cd ~/catkin_coco
catkin_make
source devel/setup.bash
```

### 4.2 运行 GEODE 序列

先在 `ct_odometry_geode.yaml` 中设置 bag 路径与本次方法开关，然后执行：

```bash
cd ~/catkin_coco
set -o pipefail
roslaunch cocolic odometry.launch \
  config_path:=/config/ct_odometry_geode.yaml 2>&1 | \
  tee src/Coco-LIC/data/Tunneling_tunnel4_gamma_terminal.txt
```

不依赖 `/usr/bin/time`。模块时间由
`*_experiment_profile.csv` 直接记录。

### 4.3 每次运行后保存

以 `Tunneling_tunnel4_gamma` 的 `QI_CTDR_R1` 为例：

```bash
cd ~/catkin_coco/src/Coco-LIC
RUN_DIR=data/experiments/Tunneling_tunnel4_gamma/QI_CTDR_R1
mkdir -p "$RUN_DIR"
cp data/Tunneling_tunnel4_gamma_LICO.txt "$RUN_DIR/"
cp data/Tunneling_tunnel4_gamma_qi_observation.csv "$RUN_DIR/" 2>/dev/null || true
cp data/Tunneling_tunnel4_gamma_ct_lidar_observability.csv "$RUN_DIR/" 2>/dev/null || true
cp data/Tunneling_tunnel4_gamma_ct_visual_recovery.csv "$RUN_DIR/" 2>/dev/null || true
cp data/Tunneling_tunnel4_gamma_experiment_profile.csv "$RUN_DIR/" 2>/dev/null || true
cp data/Tunneling_tunnel4_gamma_terminal.txt "$RUN_DIR/"
cp config/ct_odometry_geode.yaml "$RUN_DIR/config_used.yaml"
```

`2>/dev/null || true` 只用于某组按设计不生成对应 CSV 的情况，例如
Original 不生成 QI/CT CSV。轨迹和本次配置不允许缺失。

## 5. E1：总体轨迹估计精度

### 目的

验证完整 QI-CTDR 相对原始 Coco-LIC 以及仅 QI 的总体精度收益。

### 方法组

| 方法 | QI quality | QI selection | CT detector | CT recovery |
|---|---:|---:|---:|---:|
| Original | false | false | false | false |
| QI | true | true | false | false |
| QI-CTDR | true | true | CT-Spline | armed, weak-subspace |

QI-CTDR 配置：

```yaml
qi_quality_enable: true
qi_selection_enable: true
ct_degeneracy:
  enabled: true
  apply: false
  output_csv: true
  detector_mode: continuous_time
  weak_eigenvalue_ratio: 3.0e-3
ct_visual_recovery:
  enabled: true
  apply_to_estimator: true
  output_csv: true
  selection_policy: weak_subspace
  random_seed: 42
  max_additional_visual_observations: 64
experiment_profile:
  enabled: true
```

### 步骤

1. 每个 GT 序列依次运行 Original、QI、QI-CTDR。
2. 每种方法重复 3 次并按 4.3 节保存。
3. 对每次轨迹计算 SE(3) 对齐的 ATE、1 s RPE 和终点漂移：

```bash
python3 tools/evaluate_trajectory.py \
  --gt data/ground_truth/Tunneling_tunnel4_gamma_GT.txt \
  --run Original_R1=data/experiments/Tunneling_tunnel4_gamma/Original_R1/Tunneling_tunnel4_gamma_LICO.txt \
  --run QI_R1=data/experiments/Tunneling_tunnel4_gamma/QI_R1/Tunneling_tunnel4_gamma_LICO.txt \
  --run QI_CTDR_R1=data/experiments/Tunneling_tunnel4_gamma/QI_CTDR_R1/Tunneling_tunnel4_gamma_LICO.txt \
  --align se3 --rpe-delta 1.0 \
  --output-dir data/evaluation/tunnel4_r1
```

4. 建立重复实验清单 `data/evaluation/trajectory_manifest.csv`：

```csv
sequence,method,repeat,summary_json,run_label
Tunneling_tunnel4_gamma,Original,1,data/evaluation/tunnel4_r1/trajectory_summary.json,Original_R1
Tunneling_tunnel4_gamma,QI,1,data/evaluation/tunnel4_r1/trajectory_summary.json,QI_R1
Tunneling_tunnel4_gamma,QI-CTDR,1,data/evaluation/tunnel4_r1/trajectory_summary.json,QI_CTDR_R1
```

汇总：

```bash
python3 tools/summarize_experiment_repeats.py \
  --manifest data/evaluation/trajectory_manifest.csv \
  --baseline Original \
  --output data/evaluation/E1_trajectory_summary.csv
```

### 成果体现

主表报告每序列 ATE RMSE、RPE RMSE、终点漂移的“均值 ± 标准差”，并给出
跨序列平均排名。只有 QI-CTDR 相对 QI 的增量才属于 CTDR 的净贡献；
QI-CTDR 相对 Original 的差值是完整系统收益。

## 6. E2：QI 的 2×2 消融

### 目的

分离“观测可信度重标定 Q”和“信息保持选择 I”的作用，并判断两者是否互补。

| 组 | `qi_quality_enable` | `qi_selection_enable` |
|---|---:|---:|
| O | false | false |
| Q | true | false |
| I | false | true |
| QI | true | true |

四组均设置：

```yaml
ct_degeneracy:
  enabled: false
ct_visual_recovery:
  enabled: false
experiment_profile:
  enabled: true
```

### 步骤与成果

按 E1 的保存、轨迹评测和重复汇总流程运行。除 ATE/RPE 外，使用：

```bash
python3 tools/analyze_qi_observation.py \
  --run Q=data/experiments/SEQUENCE/Q_R1/SEQUENCE_qi_observation.csv \
  --run I=data/experiments/SEQUENCE/I_R1/SEQUENCE_qi_observation.csv \
  --run QI=data/experiments/SEQUENCE/QI_R1/SEQUENCE_qi_observation.csv
```

报告 LiDAR/视觉候选数、保留数、D-efficiency、最弱方向保持率和残差
median/MAD。若 Q 改善残差稳健性、I 降低观测数且保持信息、QI 同时取得两者，
即可支持“可信度与信息互补”；不能只凭 ATE 单项下结论。

## 7. E3：CT LiDAR 退化检测有效性

### 目的

验证持续低信息状态与局部轨迹误差的关系，并检验连续点时刻/样条支撑相对
离散 Scan-6DoF 的增量价值。

### 两组只读配置

两组均固定 QI 设置，关闭恢复写入：

```yaml
qi_quality_enable: true
qi_selection_enable: true
ct_visual_recovery:
  enabled: false
experiment_profile:
  enabled: true
```

仅切换：

```yaml
ct_degeneracy:
  enabled: true
  apply: false
  output_csv: true
  detector_mode: continuous_time  # CT-Spline
  weak_eigenvalue_ratio: 3.0e-3
```

或：

```yaml
  detector_mode: scan_6d
```

### 步骤

1. 运行 CT-Spline，保存检测 CSV 为 `CT_SPLINE/...`。
2. 运行 Scan-6DoF，保存检测 CSV 为 `SCAN_6D/...`。
3. 用两组任一只读轨迹生成局部误差 CSV：

```bash
python3 tools/evaluate_trajectory.py \
  --gt data/ground_truth/Tunneling_tunnel4_gamma_GT.txt \
  --run detector_base=data/experiments/Tunneling_tunnel4_gamma/CT_SPLINE/Tunneling_tunnel4_gamma_LICO.txt \
  --align se3 --rpe-delta 1.0 \
  --output-dir data/evaluation/tunnel4_detector
```

4. 联合审计：

```bash
python3 tools/analyze_ct_detector_validity.py \
  --run CT_Spline=data/experiments/Tunneling_tunnel4_gamma/CT_SPLINE/Tunneling_tunnel4_gamma_ct_lidar_observability.csv,data/evaluation/tunnel4_detector/detector_base_trajectory_errors.csv \
  --run Scan_6DoF=data/experiments/Tunneling_tunnel4_gamma/SCAN_6D/Tunneling_tunnel4_gamma_ct_lidar_observability.csv,data/evaluation/tunnel4_detector/detector_base_trajectory_errors.csv \
  --error-field rpe_translation_m \
  --max-time-difference 0.05 \
  --output-json data/evaluation/E3_tunnel4_detector.json
```

### 成果体现

报告：

- 持续触发比例、区段数和最长区段；
- 触发帧/健康帧的 RPE median、p90 及中位数比值；
- 低信息分数识别 RPE 最高 20% 帧的 AUROC；
- 触发对高误差帧的 precision/recall；
- CT-Spline 与 Scan-6DoF 的 active Jaccard 和各自独有触发帧。

有效性要求不是“触发越多越好”，而是 CT-Spline 的误差集中度、AUROC 或
高误差召回在多个有 GT 序列上稳定优于 Scan-6DoF，同时不过度降低 precision。

## 8. E4：弱方向视觉补充机理

### 目的

排除“只要增加视觉点就会改善”的替代解释，验证 LiDAR 弱方向驱动的选择价值。

### 阶段 A：同帧 Shadow 信息审计

```yaml
qi_quality_enable: true
qi_selection_enable: true
ct_degeneracy:
  enabled: true
  apply: false
  output_csv: true
  detector_mode: continuous_time
  weak_eigenvalue_ratio: 3.0e-3
ct_visual_recovery:
  enabled: true
  apply_to_estimator: false
  output_csv: true
  selection_policy: weak_subspace
  random_seed: 42
  max_additional_visual_observations: 64
```

一次 Shadow 运行会同时计算 CTDR、Random 和 Global-D 的同帧反事实指标：

```bash
python3 tools/analyze_ct_recovery_policies.py \
  --run tunnel4=data/experiments/Tunneling_tunnel4_gamma/SHADOW/Tunneling_tunnel4_gamma_ct_visual_recovery.csv \
  --output-json data/evaluation/E4_tunnel4_shadow.json
```

重点报告同一 \(K\) 下的：

- weak D-efficiency；
- weak minimum-direction retention；
- global D-efficiency；
- CTDR 相对 Random/Global-D 的逐帧胜率；
- 附加视觉观测相对 QI 基线的百分比。

### 阶段 B：Armed 轨迹对照

共同配置与 Shadow 相同，只设置：

```yaml
ct_visual_recovery:
  enabled: true
  apply_to_estimator: true
  selection_policy: weak_subspace  # 或 random / global_d
  random_seed: 0
  max_additional_visual_observations: 64
```

运行 QI、QI+Random、QI+Global-D、QI+CTDR。Random 对种子
`0,1,2,3,4` 分别运行；其他方法至少 3 次。每次执行结构审计：

```bash
python3 tools/analyze_ct_visual_recovery.py \
  --run CTDR=data/experiments/SEQUENCE/CTDR_R1/SEQUENCE_ct_visual_recovery.csv
```

再按 E1 计算轨迹指标。

### 成果体现

论文应同时给出两层证据：

1. Shadow：同帧同预算下 CTDR 对弱方向信息的恢复更强；
2. Armed：QI+CTDR 的 ATE/RPE 优于 QI，并优于或更稳定于
   QI+Random、QI+Global-D。

若只有 Shadow 信息改善而轨迹不改善，只能声称“方向选择机理成立”，不能声称
整体定位精度提高。若 Armed 改善但同帧信息不占优，则需要重新检查因果解释。

## 9. E5：参数敏感性与效率

### 参数敏感性

以统一主参数为中心，一次只改变一个参数：

| 参数 | 主值 | 扫描值 |
|---|---:|---|
| `weak_eigenvalue_ratio` | 0.003 | 0.001、0.003、0.006 |
| `qi_selection_d_efficiency` | 0.95 | 0.90、0.95、0.98 |
| `max_additional_visual_observations` | 64 | 16、32、64、128 |

先在一个开发序列上确定主值，然后锁定主值完成其他测试序列。敏感性图报告
ATE/RPE、触发率、平均附加观测数和运行时间；不从每条测试序列单独选最优点。

### 运行时间

各方法配置 `experiment_profile.enabled: true`，然后执行：

```bash
python3 tools/analyze_experiment_profile.py \
  --run Original=data/experiments/SEQUENCE/Original_R1/SEQUENCE_experiment_profile.csv \
  --run QI=data/experiments/SEQUENCE/QI_R1/SEQUENCE_experiment_profile.csv \
  --run QI_CTDR=data/experiments/SEQUENCE/QI_CTDR_R1/SEQUENCE_experiment_profile.csv \
  --output-json data/evaluation/E5_profile.json
```

报告每帧 mean/median/p90：

- CT detector；
- QI LiDAR 与视觉；
- CT recovery；
- LIC solver；
- 核心总时间；
- LiDAR/视觉候选数、保留数与 CT 附加数。

相对运行时间必须用同一机器、相同 ROS/编译配置和无其他重负载进程的连续实验。
模块计时用于解释开销来源，端到端总时间用于报告实际代价。

## 10. 完整执行顺序

1. 固定代码提交和三个主参数。
2. 在全部序列运行 E3 的 CT-Spline 与 Scan-6DoF 只读检测。
3. 在有 GT 序列运行 E2，确认 QI 基线。
4. 在全部序列运行 E4-A Shadow，检查真实视觉候选与方向信息恢复。
5. 只有 E4-A 通过后，在有 GT 序列运行 E4-B Armed。
6. 运行 E1 三组主比较。
7. 运行 E5 单参数敏感性和统一机器计时。
8. 汇总重复实验，制作主表、消融表、检测 ROC/分组箱线图、恢复信息
   CDF/箱线图和效率表。

这套顺序先验证检测与选择机理，再允许估计器写入，最后给出总体精度结论；
每项论文主张都有独立的对应证据。
