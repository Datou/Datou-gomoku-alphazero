# Datou的五子棋-AlphaZero

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

这是一个基于AlphaZero算法实现的、能够自我学习和进化的五子棋AI程序。它不依赖任何人类棋谱数据，完全通过自我对弈（Self-Play）从零开始学习，并逐步成长为强大的五子棋引擎。

本人并不会编程，本程序代码全部由Gemini Pro 2.5生成，如遇bug请询问Gemini，它会耐心解答并提供程序补丁。


## 🌟 核心特性

-   🤖 **自我对弈 (Self-Play)**: AI通过与自身（及历史上的强大版本）进行大量对弈来生成高质量的训练数据。对弈模型池由最新的冠军模型和“名人堂”中的元老模型共同组成，以保证数据的多样性和探索性。

-   🧠 **训练与进化 (Training & Evolution)**:
    -   **经验回放**: 使用一个巨大的Replay Buffer（容量百万级）存储历史经验，作为模型的长期记忆。
    -   **挑战者机制**: 从经验池中采样数据，训练一个新的“挑战者”模型。
    -   **评估竞技场**: 让“挑战者”与现任“最强模型”进行多轮对决。只有胜率超过预设阈值（如55%）的挑战者才能晋级。

-   🚀 **自动超参数优化 (Automated Hyperparameter Tuning)**: 当默认参数训练出的模型挑战失败时，系统会自动启动 [Optuna](https://optuna.org/) 进行超参数搜索，尝试多种学习率、批大小等组合，以找到更优的训练方案，极大地提升了自动化水平和模型性能。
<img width="2350" height="1225" alt="image" src="https://github.com/user-attachments/assets/ee899856-1864-4bd8-8efb-5307bbca6276" />


-   🎮 **交互式Web界面 (Interactive Web UI)**: 内置一个基于Flask的轻量级Web服务器，你可以：
    -   **人机对战**: 随时与当前最强的AI模型进行对弈。
    -   **棋局回放**: 在线观看和复盘历史上的精彩对局。
<img width="1974" height="1422" alt="image" src="https://github.com/user-attachments/assets/cbc7c3d6-0c6b-4d06-969d-7c56f6d494a9" />


-   📊 **可视化分析 (Visualization)**: 能够生成并保存模型在不同阶段的策略热力图（Policy Heatmap），直观地展示AI对开局走法的理解。初始的随机参数AI模型大概率会贴边落子，这对于CNN网络和蒙特卡洛搜索算法而言是正常现象，随着新模型战胜旧模型不断迭代，AI会慢慢走出边角。
<img width="1822" height="1570" alt="image" src="https://github.com/user-attachments/assets/f929a6b9-16a2-4eaa-9e5e-7ec5f7c1757f" />


-   📈 **命令行调度器 (Command-Line Scheduler)**: 程序启动时会提供一个交互式菜单，智能地引导用户选择下一个动作（如“开始自对弈”或“继续训练”），使复杂的训练流程管理变得简单清晰。
<img width="2350" height="1225" alt="image" src="https://github.com/user-attachments/assets/815bbc5b-082d-4513-b19a-0dc056d50398" />


## 🛠️ 技术栈

-   **核心框架**: [PyTorch](https://pytorch.org/)
-   **数值计算**: [NumPy](https://numpy.org/)
-   **性能加速**: [Numba](https://numba.pydata.org/) (用于加速游戏核心逻辑)
-   **超参数优化**: [Optuna](https://optuna.org/)
-   **Web框架**: [Flask](https://flask.palletsprojects.com/)
-   **数据可视化**: [Matplotlib](https://matplotlib.org/), [Seaborn](https://seaborn.pydata.org/)
-   **命令行工具**: [tqdm](https://github.com/tqdm/tqdm), [colorama](https://github.com/tartley/colorama)

## 🚀 快速开始

### 1. 克隆与安装

克隆本仓库并安装所需的依赖库：

```bash
git clone https://github.com/Datou/Datou-gomoku-alphazero.git
cd Datou-gomoku-alphazero
install.cmd
```

**GPU支持**:
如果你的机器上有NVIDIA GPU并已安装CUDA，强烈建议安装GPU版本的PyTorch以获得数十倍的训练加速。请访问[PyTorch官网](https://pytorch.org/get-started/locally/)获取与你CUDA版本匹配的安装命令，例如：
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

### 2. 运行程序

直接运行主程序即可启动：
```bash
train.cmd
```

程序启动后，你将在命令行看到一个交互式菜单，它会根据当前的状态（例如，是否已有足够的自对弈数据）建议你执行下一步操作。

同时，Web界面会自动启动，在浏览器中打开 `http://127.0.0.1:5001` 即可开始与AI对战。

### 命令行参数

-   `--no-gui`: 在无头模式下运行，不启动Web服务器。
-   `--clean`: 清理所有历史数据（包括模型存档、经验池等），从零开始一次全新的训练。
-   `--debug-selfplay`: 为自对弈工作进程启用详细的日志输出。

## 📁 目录结构

```
.
├── checkpoints/          # 存放所有模型存档、日志和优化器状态
│   ├── hall_of_fame/     # 历代最强模型（名人堂）
│   ├── logs/             # TensorBoard 日志
│   ├── replays/          # 保存的对局回放 (JSON格式)
│   ├── visuals/          # 保存的策略热力图 (PNG格式)
│   └── ...
├── main.py               # 主程序入口，包含智能调度逻辑
├── network.py            # 神经网络结构定义
├── requirements.txt      # 项目依赖库列表
└── templates/
    └── index.html        # Web界面的HTML模板
```

## 🙏 致谢

本项目的实现深受 DeepMind 的 AlphaGo 及 AlphaZero 系列论文启发。

## 📜 许可证

本项目采用 [MIT License](LICENSE) 开源。
