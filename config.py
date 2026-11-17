# config.py
import os
import torch

class Config:
    def __init__(self):
        self.OUTPUT_DIR = os.path.abspath(os.environ.get(
            'GOMOKU_OUTPUT_DIR', os.path.join(os.path.dirname(__file__), 'outputs', 'renju')))
        self.TRAINING_SCHEMA_VERSION = 2
        self.LOG_TACTICAL_METRICS = False
        # ================================================================
        #                      系统与环境配置
        # ================================================================
        self.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # [REVISED] 统一设置工作进程数量。
        # 这些进程将根据Trainer的指令，在自对弈和复盘模式间动态切换。
        self.NUM_WORKERS = int(os.environ.get('GOMOKU_NUM_WORKERS', '32'))
        self.INFERENCE_PRECISION = 'fp16'
        self.INFERENCE_CHANNELS_LAST = True
        self.INFERENCE_CUDA_GRAPHS = True
        self.TRAIN_AMP = True
        self.SHARED_INFERENCE = True
        
        # ================================================================
        #                      游戏与MCTS配置
        # ================================================================
        self.BOARD_SIZE = 15
        self.N_IN_ROW = 5
        self.ACTION_SPACE_SIZE = self.BOARD_SIZE * self.BOARD_SIZE

        # clean_v2 keeps its original freestyle rules. Renju uses a separate run.
        # In ordinary alternating play the canonical stone counts identify color:
        # own == opponent means Black; opponent == own + 1 means White.
        # Thus existing three-plane weights can be transferred without reshaping.
        self.RULESET = os.environ.get('GOMOKU_RULESET', 'freestyle')
        if self.RULESET not in ('freestyle', 'renju'):
            raise ValueError('GOMOKU_RULESET must be freestyle or renju.')
        self.ENABLE_RENJU = self.RULESET == 'renju'
        self.RULESET_ID = 'renju_forbidden_v1' if self.ENABLE_RENJU else 'freestyle_v1'

        # Self-play opening prefixes. Empty means every game starts on an empty
        # board. 'rif26' starts each game from one of the 26 RIF three-stone
        # openings (randomly rotated/reflected), so the network is trained from
        # standard balanced positions instead of always seeing the same empty
        # first-player line. Only applies to self-play, never to evaluation.
        self.SELFPLAY_OPENINGS = os.environ.get('GOMOKU_SELFPLAY_OPENINGS', '')
        if self.SELFPLAY_OPENINGS not in ('', 'rif26'):
            raise ValueError("GOMOKU_SELFPLAY_OPENINGS must be '' or 'rif26'.")
        if self.SELFPLAY_OPENINGS and not self.ENABLE_RENJU:
            raise ValueError('Balanced RIF openings are only valid for Renju.')
        # Fraction of self-play games that skip the standard two-stone prefix and
        # explore a free opening (only Black's mandatory centre stone is fixed).
        # Keeps the model able to handle non-book openings instead of only
        # memorising the 26 RIF prefixes.
        self.SELFPLAY_FREE_OPENING_RATIO = float(
            os.environ.get('GOMOKU_SELFPLAY_FREE_RATIO', '0.2'))
        if not 0.0 <= self.SELFPLAY_FREE_OPENING_RATIO <= 1.0:
            raise ValueError('GOMOKU_SELFPLAY_FREE_RATIO must be within [0, 1].')

        # User forbids tactical action overrides; ordinary rule legality remains.
        self.ENABLE_FORCED_MOVE_PRUNING = False

        self.NUM_SIMULATIONS = 256

        # --- AlphaZero PUCT 搜索参数 ---
        # c_puct 控制"探索"相对于"Q 值"的权重；30 是旧代码的值，会让搜索几乎
        # 退化成按先验平摊访问次数（Q 值完全不起作用）。
        self.C_PUCT = 2.0
        # 根节点 Dirichlet 噪声：只加在先验最高的 K 个着法上，避免在 200+ 个
        # 合法点上被稀释。
        self.ROOT_DIRICHLET_ALPHA = 0.3
        self.ROOT_DIRICHLET_EPSILON = 0.25
        self.ROOT_NOISE_TOP_K = 32
        # 开局前 N 手按访问分布采样，之后取 argmax。20 手在 256 次模拟下会把
        # 防守着法采丢（实测白方漏挡单点成五 30%），故缩短到 8 手；开局多样性
        # 由根节点 Dirichlet 噪声和 RIF 开局库保证。
        self.PLAY_TEMPERATURE_MOVES = 8
        # 候选着法剪枝：只搜索"已有棋子附近 radius 格"内的空点。五子棋里
        # 远离棋子的着法没有意义，这一步能把 225 个动作压到 30~60 个。
        self.ACTION_RADIUS = 2
        
        # ================================================================
        #                      网络结构配置 (为五子棋优化)
        # ================================================================
        # VALUE: 赢, 输, 平局. 没了.
        self.VALUE_SUPPORT_MIN = -1
        self.VALUE_SUPPORT_MAX = 1
        self.VALUE_SUPPORT_BINS = 3 # 分别对应 -1, 0, 1

        # REWARD support is kept for checkpoint-format compatibility; AlphaZero
        # does not train a reward head.
        self.REWARD_SUPPORT_MIN = -1
        self.REWARD_SUPPORT_MAX = 1
        self.REWARD_SUPPORT_BINS = 3 # 奖励也只有-1, 0, 1
        
        self.NUM_RES_BLOCKS = 8 # 6
        self.NUM_FILTERS = 128 # 128
        self.HEAD_HIDDEN_DIM = 64
        
        # ================================================================
        #                      训练过程配置 (为五子棋优化)
        # ================================================================
        self.PHYSICAL_BATCH_SIZE = 128
        # Each newly generated position funds at most eight training-sample uses.
        self.TRAIN_SAMPLES_PER_POSITION = 8.0
        self.WARMUP_UPDATES = 100
        # 训练步频上限：自对弈每秒只产生约 7 个样本，而训练器默认每秒能吞 360*11 个，
        # 相当于对同一批数据反复过拟合（实测分布外 value 准确率只有 50%）。
        self.MAX_TRAIN_STEPS_PER_SEC = 6.0
        self.GRADIENT_ACCUMULATION_STEPS = 1
        
        self.TRAIN_BUFFER_SIZE = 100000
        self.REPLAY_BUFFER_PREFILL = 4096
        self.MIN_BUFFER_LEAD = 10000

        self.ENABLE_BACKPRESSURE = False
        # 旧实现用 (buffer_size - train_step_count) 做背压：前者是"样本数"，
        # 后者是"优化步数"，量纲不同，结果是一旦 buffer 超过阈值就长期暂停
        # 自对弈，数据迅速变旧。需要限流时应该按"本轮模型产生的样本数"来算。
        
        self.OPTIMIZER_TYPE = 'Adam'
        self.LEARNING_RATE = 2e-4
        # Cosine decay from LEARNING_RATE to LR_MIN over LR_DECAY_UPDATES optimizer
        # steps, then the learning rate stays at LR_MIN (it used to climb back up).
        self.LR_DECAY_UPDATES = 200000
        self.LR_MIN = 1e-7
        self.WEIGHT_DECAY = 1e-4
        self.BARLOW_LAMBDA = 5e-3
        
        self.TARGET_MODEL_TAU = 0.995
        self.NUM_UNROLL_STEPS = 0
        # value 目标 = 自对弈的最终胜负 z（AlphaZero 做法），moves_left 为
        # 辅助任务，目标做 log 归一化后权重要小，否则它（数值可达上百）
        # 会完全压过 policy/value 的梯度。
        self.LOSS_WEIGHTS = {'policy': 1.0, 'value': 1.0, 'moves_left': 0.05}

        self.GRAD_CLIP_NORM = 5.0
        
        # self.LOSS_WEIGHTS overrides above
        
        # ================================================================
        #               [REVISED] Re-analysis (浪涌式复盘) 配置
        # ================================================================
        # 是否启用动态的“浪涌式复盘”模式
        self.ENABLE_REANALYSIS = False
        
        # 当一个游戏的分析版本落后于当前训练器超过 N 步时，
        # Trainer将触发全局模式切换，命令所有Worker进入复盘模式。
        self.REANALYSIS_AGE_THRESHOLD = 900
        
        # ================================================================
        #                其他配置
        # ================================================================
        # [新增] 是否启用优先经验回放 (PER) 的总开关
        self.ENABLE_PER = False

        # PER (优先经验回放) 参数
        self.PER_ALPHA = 0.6; self.PER_BETA = 0.4; self.PER_BETA_INCREMENT = 0.00001; self.PER_EPSILON = 1e-6
        
        self.PREPARED_BATCH_QUEUE_SIZE = 10
        self.INFERENCE_BATCH_SIZE = 32

        self.SAVE_INTERVAL = int(os.environ.get('GOMOKU_SAVE_INTERVAL', '1000'))
        self.MODEL_UPDATE_INTERVAL = 50
        
        self.CURRENT_CONFIG = "Gomoku_15x15_" + self.RULESET_ID

        # Used only by the isolated, bounded end-to-end verification run.
        if os.environ.get('GOMOKU_PROFILE') == 'smoke':
            self.NUM_WORKERS = 2
            self.NUM_SIMULATIONS = 24
            self.PHYSICAL_BATCH_SIZE = 16
            self.REPLAY_BUFFER_PREFILL = 64
            self.MODEL_UPDATE_INTERVAL = 5
            self.SAVE_INTERVAL = 10
            self.WARMUP_UPDATES = 5

config = Config()
