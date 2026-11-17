# 训练归档

这个目录保存 `renju_v2` 这次 AlphaZero 训练的原始产物，便于核对 README 中的数字。

| 文件 | 内容 |
|---|---|
| `run_config.json` | 启动时的完整配置快照（32 worker、256 模拟、RIF 开局等）。 |
| `training.log` | 从 2026-11-16 00:18 到 2026-11-17 05:45 的完整训练日志，约 5 MB。 |
| `supervision.jsonl` | 每 5 小时自动写入的训练状态与评估记录。 |
| `supervision.json` | 最近一次监督状态。 |
| `strength_goal_status.json` | 强度目标达成记录，指向两次 100 局验收结果。 |
| `EXPERIMENT.md` | 本次实验的设定摘要。 |
| `STOP` | 触发优雅停止的标记文件内容。 |
| `evaluations/` | 第 2,500 / 5,000 / 10,500 / 15,500 / 20,500 步的原始评估 JSON，以及 20,500 vs 27,702 的 20 局对战结果。 |

## 与仓库目录的对应关系

原始运行目录是 `outputs/renju_v2/`，归档时把最终权重移到了 `weights/`：

| 原始路径 | 仓库路径 |
|---|---|
| `outputs/renju_v2/model_weights/model_weights_step_20500.pt` | `weights/alphazero_20500.pt` |
| `outputs/renju_v2/training_state.db`（step 27,702 的 `model_state_dict`） | `weights/alphazero_27702.pt` |
| `outputs/renju_v2/initial_model.pt` | `weights/initial_model.pt` |
| `outputs/renju_v2/run_config.json` | `training/run_config.json` |
| `outputs/renju_v2/evaluations/*.json` | `training/evaluations/*.json` |
| `outputs/renju_v2/training.log` | `training/training.log` |

原始路径仍保留在 `run_config.json`、`supervision.jsonl` 和评估 JSON 的 `checkpoint` 字段中，这是刻意保留的可追溯信息；`weights/*.json` 是与权重同名的结构/规则元数据，不包含本机路径。

`alphazero_27702.pt` 是从 `training_state.db` 的 `trainer_state.singleton_state` 反序列化后取出的 `model_state_dict`，对应训练停止时的最终步数 27,702；它不是周期检查点文件，所以仓库里没有同名的原始 `.pt` 路径。

## 复现说明

- 训练是单机、随机初始化、纯自我对弈；没有人类棋谱或预训练权重。
- 由于 GPU 型号、驱动、worker 数和 PyTorch 版本不同，具体步数与墙钟时间会有差异；趋势应一致：先学会规则与基本吃子，再在 1 万步左右学会挡四/成五，2 万步左右超过 greedy 基准。
- `training.log` 末尾的 “did not exit gracefully / Terminating” 是主进程在 `STOP` 后强制回收未及时退出的 worker，不影响已保存的权重；最终状态在第 27,702 步保存成功。
