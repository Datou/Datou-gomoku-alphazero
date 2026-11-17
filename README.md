# 五子棋 AlphaZero（Renju 禁手规则）

从零开始、只用自我对弈训练的 15×15 五子棋 AlphaZero。完整实现 RIF 禁手规则（黑棋长连、四四、三三），并使用 26 种 RIF 平衡开局 + 20% 自由开局训练。

本仓库只包含 **AlphaZero** 代码、最终权重、训练过程和结果。MuZero 实验代码、权重和日志不在这里。

<p align="center">
  <img src="docs/preview.png" alt="浏览器对战页" width="900">
</p>

## 目录

- [结果速览](#结果速览)
- [在线试玩（浏览器）](#在线试玩浏览器)
- [训练方法](#训练方法)
- [训练过程](#训练过程)
- [最终权重](#最终权重)
- [代码结构](#代码结构)
- [本地运行](#本地运行)
- [评估与复现](#评估与复现)
- [浏览器对战页原理](#浏览器对战页原理)
- [已知限制](#已知限制)

## 结果速览

训练从随机初始化开始，没有使用任何人类棋谱、开局库落子提示或战术规则覆盖。共训练 **27,702 步**，约 **29 小时**（单张 RTX 4090，32 个自对弈 worker，每步 256 次 MCTS 模拟）。

| 训练步数 | 对 greedy 基准（8 局） | 对随机初始模型（8 局） |
|---:|---:|---:|
| 2,500 | 0 胜 0 和 8 负 | 8 胜 0 和 0 负 |
| 5,000 | 0 胜 0 和 8 负 | 8 胜 0 和 0 负 |
| 10,500 | 6 胜 0 和 2 负（75%） | 8 胜 0 和 0 负 |
| 15,500 | 7 胜 0 和 1 负（87.5%） | 8 胜 0 和 0 负 |
| **20,500** | **8 胜 0 和 0 负（100%）** | **8 胜 0 和 0 负（100%）** |

第 20,500 步的模型随后做了更大规模的验收（256 次模拟、50 个固定开局、双方交换执子，共 100 局）：

- 对 greedy 基准：**87 胜 1 和 12 负，得分率 87.5%**，配对 bootstrap 95% 区间 **[81.5%, 93.0%]**；
- 对随机初始模型：**100 胜 0 和 0 负**。

达到预设强度目标后，训练在第 27,702 步被 `STOP` 优雅停止并保存最终状态。第 20,500 步是经过完整验收的推荐模型；第 27,702 步是最终训练状态，从 `trainer_state` 中导出为权重。

## 在线试玩（浏览器）

部署到 GitHub Pages 后，直接打开：

**https://datou.github.io/Datou-gomoku-alphazero/**

页面完全静态：模型在浏览器本地通过 ONNX Runtime Web 推理，不需要后端服务器，也不会把棋局上传到任何地方。

- 可选择 **20,500（验收版）** 或 **27,702（最终训练版）** 模型；
- 模型加载时显示下载进度、百分比和初始化状态；
- 可选择执黑（先手，有禁手）或执白；
- 每步 64 / 128 / 256 次 MCTS 模拟；
- 优先使用 **WebGPU**，不可用时自动回退到单线程 **WASM**；
- 黑棋下出长连、四四、三三会立即判负；白棋连成五子或以上获胜。

首次打开需要加载本地 ONNX Runtime 运行时并下载约 10 MB 的模型，通常 10–30 秒；之后会使用浏览器缓存。128 次模拟在普通桌面 CPU 上约 10 秒/步，WebGPU 会快很多。

### 启用 GitHub Pages

仓库已把静态页面放在 `docs/`：

1. 打开仓库 **Settings → Pages**；
2. **Build and deployment → Source** 选择 **Deploy from a branch**；
3. Branch 选择 `main`，目录选择 **`/docs`**，保存；
4. 等待 Pages 构建完成，访问上面的地址。

本地预览：

```bash
python tools/serve_docs.py
# 打开 http://127.0.0.1:8000/
```

> 用 `python -m http.server` 也可以，但某些 Python 版本会把 `.mjs` 当成 `text/plain`，导致 ONNX Runtime 的模块加载失败；`tools/serve_docs.py` 已修正 MIME 类型。

> `docs/models/*.onnx` 是最终权重的推理版本，和 `weights/*.pt` 一一对应。

## 训练方法

### 规则

- 15×15 棋盘，黑先。
- 黑棋必须**恰好**连成五子才能获胜；长连（六子及以上）判负。
- 黑棋四四、三三为禁手，落子即判负。
- 白棋连成五子或以上即获胜，没有禁手。
- 棋盘下满判和。

禁手判定在 `renju_rules.py` 中递归实现：一个“三”只有在它的两个直四延伸点都能形成合法五连时才算真三；延伸点本身是禁手时，这个三不算数。这样能正确处理“假三”“四四解禁”“长连解禁”等边界情况。

### 网络

- 输入：3×15×15 的规范平面 —— 当前行动方棋子、对手棋子、最后一手位置。
- 主干：8 个 SE-ResBlock，128 通道。
- 策略头：225 维落子 logits。
- 价值头：3 个支撑点 `{-1, 0, +1}` 的分布，推理时取期望。
- 辅助头：预测剩余步数（训练时做 log 归一化回归，权重 0.05）。

### 搜索

- AlphaZero PUCT，`c_puct = 2.0`。
- 每步 256 次模拟；只搜索已有棋子附近半径 2 的候选点。
- 根节点 Dirichlet 噪声：仅加在先验最高的 32 个着法上，`alpha = 0.3`，`epsilon = 0.25`。
- 前 8 手按访问次数分布采样，之后取访问次数最多的着法。
- 终局节点不展开：按“轮到走棋一方”的胜负记 +1 / 0 / -1，沿树回溯时逐层变号。

### 自我对弈与开局

- 每局从 26 种 RIF 三手开局中随机选一种，并随机做 8 种对称变换（旋转/镜像）。
- 20% 的自我对弈局只固定黑棋天元，从自由开局开始，避免模型只会背 26 个开局。
- 开局的三手棋作为“棋谱着法”记录，策略目标为 one-hot；从白棋第四手开始由 MCTS 正常搜索。
- 不使用任何强制成五/强制挡四的战术覆盖；所有落子都由策略网络 + PUCT 决定。

### 优化器与回放

- Adam，学习率 `2e-4`，余弦衰减到 `1e-7`（200,000 步）。
- 物理 batch 128；每个新产生的局面最多被采样 8 次，防止对同一批数据反复过拟合。
- 回放缓冲区 100,000 个样本；目标网络软更新 `tau = 0.995`。
- 价值目标始终是最终胜负 `z`，不从模型自身价值 bootstrap。
- 数据增强：随机旋转 0/90/180/270 度 + 水平翻转，策略和动作索引同步变换。

## 训练过程

完整日志在 [`training/training.log`](training/training.log)，运行配置在 [`training/run_config.json`](training/run_config.json)，每次强度评估的原始对局在 [`training/evaluations/`](training/evaluations/)。

时间线（Asia/Shanghai）：

- **2026-11-16 00:18**：从随机初始化开始训练（`renju_v2`）。
- **第 2,500–5,000 步**：对 greedy 基准 0/8，但已经能 8/8 击败随机初始模型；说明学会了基本规则和吃子，但还不会处理成五威胁。
- **第 10,500 步**：对 greedy 6/8，首次超过 50%。
- **第 15,500 步**：对 greedy 7/8；只差一局达到目标。
- **第 20,500 步**：对 greedy 8/8；随后完成 100 局验收，87.5% 得分率，写入 `strength_goal_status.json`（`status: reached`）。
- **2026-11-17 05:45**：训练继续到第 27,702 步后，`STOP` 文件触发优雅停止；最终状态保存。

为什么第 20,500 步之后还继续训练？因为强度验收是异步的：训练进程不会等待评估结果，验收完成时模型已经继续前进了。训练停止前的最后一个周期检查点是第 27,500 步，最终状态在第 27,702 步保存；两者都没有再做 100 局验收，所以 README 把第 20,500 步标为“验收版”，把第 27,702 步标为“最终训练版”。

## 最终权重

| 文件 | 说明 |
|---|---|
| [`weights/alphazero_20500.pt`](weights/alphazero_20500.pt) | **推荐模型**。第 20,500 步，通过 100 局验收（87.5% vs greedy，100% vs 初始模型）。 |
| [`weights/alphazero_27702.pt`](weights/alphazero_27702.pt) | 第 27,702 步，训练停止时保存的最终状态；未做完整 100 局验收。 |
| [`weights/initial_model.pt`](weights/initial_model.pt) | 随机初始化参考模型（第 0 步），只用于评估基线。 |

每个 `.pt` 旁边都有同名 `.json`，记录棋盘大小、网络结构、规则和搜索参数。权重是 PyTorch `state_dict`，可以直接这样加载：

```python
import torch
from config import config
from network import GomokuNetEZ

model = GomokuNetEZ(config)
state = torch.load("weights/alphazero_20500.pt", map_location="cpu", weights_only=True)
model.load_state_dict(state, strict=True)
model.eval()
```

浏览器使用的是对应导出的 ONNX：

- `docs/models/alphazero_20500.onnx`
- `docs/models/alphazero_27702.onnx`

重新导出：

```bash
python tools/export_onnx.py \
  --checkpoint weights/alphazero_20500.pt \
  --output docs/models/alphazero_20500.onnx
```

## 代码结构

```text
.
├── main.py                 # 训练入口：启动 worker、推理服务、DataLoader、Trainer
├── config.py               # 所有超参数与运行开关
├── game.py                 # 棋盘状态、合法着法、终局判定
├── renju_rules.py          # 黑棋禁手判定（递归真三检测）
├── rif_openings.py         # 26 种 RIF 开局与 8 种对称变换
├── network.py              # SE-ResNet、策略头、价值头、辅助头
├── mcts.py                 # AlphaZero PUCT 搜索
├── loss.py                 # policy / value / moves-left 损失
├── workers.py              # 自对弈、推理服务、DataLoader、Trainer
├── db_manager.py           # SQLite 回放与训练状态持久化
├── inference_backend.py    # 批量推理、fp16、CUDA Graph
├── shared_inference.py     # 共享内存推理传输
├── training_runtime.py     # 运行锁、STOP/PAUSE、预算和断点恢复
├── checkpoint_metadata.py  # 权重与规则/结构元数据校验
├── docs/                   # GitHub Pages 静态对战页
│   ├── index.html
│   ├── css/style.css
│   ├── js/renju.js         # 浏览器版禁手规则
│   ├── js/mcts.js          # 浏览器版 PUCT
│   ├── js/app.js           # 棋盘 UI 与模型加载
│   ├── vendor/             # ONNX Runtime Web 运行时与 wasm（离线可用）
│   └── models/*.onnx
├── weights/                # 最终 PyTorch 权重
├── training/               # 训练日志、配置、评估原始结果
├── tools/                  # 评估、arena、ONNX 导出
└── tests/                  # 规则、搜索、损失、训练管线测试
```

## 本地运行

### 环境

- Python 3.12
- PyTorch 2.x（CUDA 版本按显卡安装）
- 其余依赖见 `requirements.txt`

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

### 训练

```bash
python main.py
```

默认输出到 `outputs/renju/`。常用环境变量：

| 变量 | 说明 |
|---|---|
| `GOMOKU_OUTPUT_DIR` | 输出目录 |
| `GOMOKU_NUM_WORKERS` | 自对弈 worker 数（默认 32） |
| `GOMOKU_RULESET` | `renju` 或 `freestyle` |
| `GOMOKU_SELFPLAY_OPENINGS` | `rif26` 启用 26 种 RIF 开局 |
| `GOMOKU_SELFPLAY_FREE_RATIO` | 自由开局比例（默认 0.2） |
| `GOMOKU_TRAIN_SECONDS` | 训练时间上限（秒，0 表示不限） |
| `GOMOKU_SAVE_INTERVAL` | 检查点间隔（默认 1000 步） |
| `GOMOKU_PROFILE=smoke` | 小规模端到端自检配置 |

PowerShell 示例：

```powershell
$env:GOMOKU_RULESET='renju'
$env:GOMOKU_SELFPLAY_OPENINGS='rif26'
$env:GOMOKU_OUTPUT_DIR='outputs/renju'
python main.py
```

训练过程中会在输出目录写入 `training.log`、`run_config.json`、`trainer_status.json`、`model_weights/`、`training_state.db`。停止训练：在输出目录放一个 `STOP` 文件，主进程会优雅保存并退出。

### 测试

```bash
python -m unittest discover -s tests -v
node tests/test_web_mcts.js
```

测试覆盖禁手边界、PUCT 符号回溯、价值目标、数据切片、SQLite 状态恢复、RIF 开局合法性和 ONNX 推理一致性；`node tests/test_web_mcts.js` 专门检查浏览器 MCTS 在只有一个成五点时必须落子成五。

## 评估与复现

### 固定开局 + 双方交换执子

`tools/evaluate_progress.py` 使用 4 个固定开局，每个开局双方各执一次，共 8 局；评估时关闭根节点噪声和温度采样。

```bash
# 8 局快速进度评估
python tools/evaluate_progress.py \
  --checkpoint weights/alphazero_20500.pt \
  --sims 128 --pairs 4

# 对随机初始模型
python tools/evaluate_progress.py \
  --checkpoint weights/alphazero_20500.pt \
  --sims 128 --pairs 4 --opponent initial
```

### 100 局验收

验收协议使用冻结的 50 个开局（`tests/fixtures/renju_confirmation_openings_v1.json`），每个开局双方各执一次，共 100 局，每步 256 次模拟：

```bash
python tools/evaluate_progress.py \
  --checkpoint weights/alphazero_20500.pt \
  --sims 256 --suite confirmation

python tools/evaluate_progress.py \
  --checkpoint weights/alphazero_20500.pt \
  --sims 256 --suite confirmation --opponent initial
```

原始结果（含每局完整着法）在 `training/evaluations/`。验收通过条件写在 `tools/evaluate_progress.py` 中：得分率 ≥ 70%、黑白两侧都不低于 50%、配对 bootstrap 区间下界 > 50%。

### 两个模型互下

```bash
python tools/arena.py \
  --a weights/alphazero_20500.pt \
  --b weights/alphazero_27702.pt \
  --games 10 --sims 128
```

## 浏览器对战页原理

浏览器端没有 Python，也没有服务端搜索。页面加载三个纯 JavaScript 文件：

1. `docs/js/renju.js`：把 `renju_rules.py` 的禁手规则逐条移植到 JavaScript，输入是 `Int8Array(225)` 棋盘；
2. `docs/js/mcts.js`：异步 PUCT。每次模拟调用一次 ONNX 前向，得到 policy/value 后展开叶节点；
3. `docs/js/app.js`：Canvas 棋盘、模型加载、WebGPU/WASM 回退、落子与禁手判负。

推理使用 [ONNX Runtime Web](https://onnxruntime.ai/docs/tutorials/web/) 1.20.1。运行时和 wasm 文件已经放在 `docs/vendor/`，页面不依赖任何外部 CDN，断网也能玩（模型和运行时都从同一静态站点加载）：

- 优先 `executionProviders: ['webgpu']`；
- 失败时回退 `['wasm']`，并设置 `ort.env.wasm.numThreads = 1`（GitHub Pages 没有 COOP/COEP 响应头，不能启用多线程 WASM）；
- 模型输入 `obs: float32[1,3,15,15]`，输出 `policy[1,225]`、`value[1,1]`、`moves_left[1,1]`。

浏览器版规则与 Python 版做过逐点对比：在 1,609 个随机局面点、8,000 个密集局面点和 15,000 个定向三三局面上，禁手原因完全一致；候选着法集合与终局判定也在 264 个对局快照上一致。

浏览器 MCTS 的值始终以“当前行棋方”的视角存储，回溯跨过一条边时变号。因此自己形成五连的着法在叶节点是终局 `+1`，父节点会把它记为对自己有利的 `+1`；在这套符号约定下，128 次模拟已经足够发现一步成五，不需要 256 次。

## 已知限制

- 只支持 15×15 五子棋和 RIF 禁手规则，不支持连珠的其他比赛变体（如指定开局、五手两打）。
- 浏览器端为了保持页面简单，MCTS 在主线程运行；256 次模拟时页面会短暂无响应，属于预期行为。
- CPU/WASM 推理明显慢于 WebGPU；移动端建议 64 次模拟。
- 第 27,702 步模型没有做 100 局验收，不能仅凭训练步数断言它一定强于第 20,500 步；推荐使用验收版。
- 训练使用单机 32 worker，复现完整 29 小时运行需要相近的 GPU 和 CPU 资源；小规模复现可用 `GOMOKU_PROFILE=smoke`。

## 致谢

规则实现遵循 RIF 对禁手的定义；搜索与训练遵循 AlphaZero 的 PUCT/自我对弈框架。感谢所有开源工具：PyTorch、ONNX、ONNX Runtime Web。
