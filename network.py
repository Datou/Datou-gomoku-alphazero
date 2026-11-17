# network.py (FINAL, CLEANED VERSION)

import torch
import torch.nn as nn
import torch.nn.functional as F
from config import config # Import config directly

# Helper functions are fine, keep them.
def support_to_scalar(logits, support_min, support_max, support_bins):
    if logits.device.type == 'mps': logits = logits.to('cpu')
    support = torch.linspace(support_min, support_max, support_bins, device=logits.device)
    probabilities = F.softmax(logits, dim=1)
    return torch.sum(support * probabilities, dim=1, keepdim=True)

def scalar_to_support(scalar, support_min, support_max, support_bins):
    scalar = scalar.clamp(support_min, support_max)
    scaling = (support_bins - 1) / (support_max - support_min)
    float_idx = (scalar - support_min) * scaling
    low_idx, high_idx = torch.floor(float_idx).long(), torch.ceil(float_idx).long()
    high_weight = float_idx - low_idx.float()
    low_weight = 1 - high_weight
    target_support = torch.zeros(scalar.size(0), support_bins, device=scalar.device)
    target_support.scatter_add_(1, low_idx.unsqueeze(1), low_weight.unsqueeze(1))
    target_support.scatter_add_(1, high_idx.unsqueeze(1), high_weight.unsqueeze(1))
    return target_support

def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)

class SEResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None, reduction=16):
        super().__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels, eps=1e-4)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels, eps=1e-4)
        self.downsample = downsample
        
        # Squeeze-and-Excitation (SE) Block
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(out_channels, out_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(out_channels // reduction, out_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
            
        # Apply SE
        b, c, _, _ = out.size()
        y = self.avg_pool(out).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        out = out * y.expand_as(out)
        
        out += identity
        return self.relu(out)

class RepresentationNetwork(nn.Module):
    def __init__(self, observation_shape, num_blocks, num_channels):
        super().__init__()
        self.conv = conv3x3(observation_shape[0], num_channels)
        self.bn = nn.BatchNorm2d(num_channels, eps=1e-4)
        self.resblocks = nn.Sequential(*[SEResBlock(num_channels, num_channels) for _ in range(num_blocks)])
    def forward(self, x):
        return self.resblocks(F.relu(self.bn(self.conv(x))))

class PredictionNetwork(nn.Module):
    def __init__(self, num_channels, board_size, policy_output_size, value_support_bins, head_hidden_dim):
        super().__init__()
        
        # Policy Head
        self.policy_conv = nn.Conv2d(num_channels, 2, kernel_size=1)
        self.policy_bn = nn.BatchNorm2d(2, eps=1e-4)
        self.policy_fc = nn.Linear(2 * board_size * board_size, policy_output_size)
        self.policy_dropout = nn.Dropout(0.1)
        
        # Value Head (Classic)
        self.value_conv = nn.Conv2d(num_channels, 1, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(1, eps=1e-4)
        
        # Global Pooling Extension for Value Head
        # Concatenate: [Global Avg Pool (num_channels)] + [Flattened Conv1x1 (board_size^2)]
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        self.value_fc1 = nn.Linear(board_size * board_size + num_channels, head_hidden_dim) 
        self.value_fc2 = nn.Linear(head_hidden_dim, value_support_bins)
        self.value_dropout = nn.Dropout(0.1)
        
        # Moves Left Head (Auxiliary)
        # Predicts (0-1) normalized game length or bucketed length. Let's stick to scalar regression for simplicity now.
        # It shares the global pooling features.
        self.moves_left_fc1 = nn.Linear(board_size * board_size + num_channels, head_hidden_dim)
        self.moves_left_fc2 = nn.Linear(head_hidden_dim, 1) # Outputs a scalar estimate

    def forward(self, x):
        p = F.relu(self.policy_bn(self.policy_conv(x))).reshape(x.size(0), -1)
        policy_logits = self.policy_fc(self.policy_dropout(p))
        
        # Value Head with Global Context
        v_local = F.relu(self.value_bn(self.value_conv(x))).reshape(x.size(0), -1)
        v_global = self.global_pool(x).view(x.size(0), -1)
        
        v_combined = torch.cat([v_local, v_global], dim=1)
        
        v = F.relu(self.value_fc1(v_combined))
        value_logits = self.value_fc2(self.value_dropout(v))
        
        # Moves Left Prediction
        ml = F.relu(self.moves_left_fc1(v_combined))
        moves_left_pred = self.moves_left_fc2(ml)
        
        return policy_logits, value_logits, moves_left_pred

class GomokuNetEZ(nn.Module):
    def __init__(self, config_obj):
        super().__init__()
        # Use the passed config object
        self.board_size = config_obj.BOARD_SIZE
        self.action_space_size = config_obj.ACTION_SPACE_SIZE
        num_res_blocks, num_filters, head_hidden_dim = config_obj.NUM_RES_BLOCKS, config_obj.NUM_FILTERS, config_obj.HEAD_HIDDEN_DIM
        self.value_support_min, self.value_support_max, self.value_support_bins = config_obj.VALUE_SUPPORT_MIN, config_obj.VALUE_SUPPORT_MAX, config_obj.VALUE_SUPPORT_BINS
        observation_shape = (3, self.board_size, self.board_size)
        
        self.representation_net = RepresentationNetwork(observation_shape, num_res_blocks, num_filters)
        self.prediction_net = PredictionNetwork(num_filters, self.board_size, self.action_space_size, self.value_support_bins, head_hidden_dim)
        
        for m in self.modules():
            if isinstance(m, SEResBlock): nn.init.constant_(m.bn2.weight, 0)

    def representation(self, state_input): return self.representation_net(state_input)
    def prediction(self, hidden_state): return self.prediction_net(hidden_state)
    @torch.no_grad()
    def initial_inference(self, obs):
        # NOTE: the model is expected to already be in eval() mode (the
        # inference server / web UI do this once). Calling self.eval() on every
        # single inference walked all 8 residual blocks and cost ~15% of the
        # search time.
        hidden_state = self.representation(obs)
        policy_logits, value_logits, moves_left = self.prediction(hidden_state)
        value_scalar = support_to_scalar(value_logits, self.value_support_min, self.value_support_max, self.value_support_bins)
        return policy_logits, value_scalar, hidden_state
