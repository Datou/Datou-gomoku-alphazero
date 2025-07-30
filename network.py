# =====================================================================
#
#                    network.py (V2 - Configurable)
#
# - 将网络尺寸 (num_res_blocks, num_filters) 暴露为初始化参数，
#   允许在外部轻松创建不同大小的模型。
#
# =====================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F

def kaiming_init(m):
    if isinstance(m, (nn.Conv2d)):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None: nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1); nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None: nn.init.constant_(m.bias, 0)

class ResidualBlock(nn.Module):
    def __init__(self, num_filters):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(num_filters, num_filters, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_filters)
        self.conv2 = nn.Conv2d(num_filters, num_filters, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_filters)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return F.relu(out)

class GomokuNet(nn.Module):
    # --- 核心修改：将网络尺寸变为可配置参数 ---
    def __init__(self, board_size=15, num_res_blocks=10, num_filters=256, dropout_p=0.3):
        super(GomokuNet, self).__init__()
        self.board_size = board_size
        self.dropout_p = dropout_p
        
        # 使用传入的参数来构建网络
        self.conv_in = nn.Conv2d(3, num_filters, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_in = nn.BatchNorm2d(num_filters)
        self.res_blocks = nn.ModuleList([ResidualBlock(num_filters) for _ in range(num_res_blocks)])
        
        # Policy Head
        self.policy_conv = nn.Conv2d(num_filters, 1, kernel_size=1, stride=1)
        
        # Value Head
        self.value_conv = nn.Conv2d(num_filters, 1, kernel_size=1, stride=1, bias=False)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(board_size * board_size, 256)
        self.value_dropout = nn.Dropout(p=self.dropout_p)
        self.value_fc2 = nn.Linear(256, 1)

        # Auxiliary Outcome Head
        self.outcome_conv = nn.Conv2d(num_filters, 2, kernel_size=1, stride=1, bias=False)
        self.outcome_bn = nn.BatchNorm2d(2)
        self.outcome_fc1 = nn.Linear(2 * board_size * board_size, 64)
        self.outcome_dropout = nn.Dropout(p=self.dropout_p)
        self.outcome_fc2 = nn.Linear(64, 1)

        self.apply(kaiming_init)

    def forward(self, x):
        x = F.relu(self.bn_in(self.conv_in(x)))
        for res_block in self.res_blocks:
            x = res_block(x)
        
        policy = self.policy_conv(x)
        policy = policy.view(-1, self.board_size * self.board_size)
        policy = F.log_softmax(policy, dim=1)
        
        value = F.relu(self.value_bn(self.value_conv(x)))
        value = value.view(-1, self.board_size * self.board_size)
        value = F.relu(self.value_fc1(value))
        value = self.value_dropout(value) 
        value = torch.tanh(self.value_fc2(value))
        
        outcome = F.relu(self.outcome_bn(self.outcome_conv(x)))
        outcome = outcome.view(-1, 2 * self.board_size * self.board_size)
        outcome = F.relu(self.outcome_fc1(outcome))
        outcome = self.outcome_dropout(outcome)
        outcome = torch.tanh(self.outcome_fc2(outcome))
        
        return policy, value, outcome