# loss.py
#
# AlphaZero loss: policy cross-entropy + value cross-entropy over the support
# + an auxiliary moves-left regression. The value target is the final game
# result z from the point of view of the player to move; it is never
# bootstrapped from the model's own value estimate.

import math

import numpy as np
import torch
import torch.nn.functional as F

from config import config
from network import scalar_to_support, support_to_scalar


def augment_actions(actions, k, flip, h, w):
    """Map flat action indices through rot90(k) and an optional horizontal flip."""
    rows, cols = actions // w, actions % w
    if k == 1:
        rows, cols = w - 1 - cols, rows
    elif k == 2:
        rows, cols = h - 1 - rows, w - 1 - cols
    elif k == 3:
        rows, cols = cols, h - 1 - rows
    if flip:
        cols = w - 1 - cols
    return rows * w + cols


def calculate_loss(model, target_model, batch, is_weights):
    model.train()
    target_model.eval()
    obs_b, act_b, rew_b, pi_b, mcts_val_b, moves_left_b = batch
    mcts_val_b = mcts_val_b.float()
    moves_left_b = moves_left_b.float()
    loss_weights = config.LOSS_WEIGHTS

    # Shared augmentation for the observation, policy and action planes.
    k = np.random.randint(4)
    flip = np.random.choice([True, False])
    obs_b_aug = torch.rot90(obs_b, k, dims=[3, 4])
    if flip:
        obs_b_aug = torch.flip(obs_b_aug, dims=[4])
    b, u_plus_1, c, h, w = obs_b_aug.shape
    pi_b_aug = pi_b.view(b, u_plus_1, h, w)
    pi_b_aug = torch.rot90(pi_b_aug, k, dims=[2, 3])
    if flip:
        pi_b_aug = torch.flip(pi_b_aug, dims=[3])
    pi_b_aug = pi_b_aug.flatten(start_dim=2)

    value_targets = mcts_val_b[:, :1].clamp(
        config.VALUE_SUPPORT_MIN, config.VALUE_SUPPORT_MAX)

    with torch.autocast(device_type=config.DEVICE.type, dtype=torch.float16,
                        enabled=config.DEVICE.type == 'cuda' and config.TRAIN_AMP):
        hidden_state = model.representation(obs_b_aug[:, 0])
        policy_logits, value_logits, moves_left_pred = model.prediction(hidden_state)

    policy_loss = F.cross_entropy(policy_logits.float(), pi_b_aug[:, 0], reduction='none')
    value_loss = F.cross_entropy(
        value_logits.float(),
        scalar_to_support(value_targets[:, 0], config.VALUE_SUPPORT_MIN,
                          config.VALUE_SUPPORT_MAX, config.VALUE_SUPPORT_BINS),
        reduction='none')

    moves_left_target = torch.log1p(moves_left_b[:, 0].clamp(min=0)) / math.log1p(
        config.BOARD_SIZE * config.BOARD_SIZE)
    moves_left_loss = F.mse_loss(
        moves_left_pred.float().squeeze(-1), moves_left_target, reduction='none')

    pred_values_scalar = support_to_scalar(
        value_logits.float(), config.VALUE_SUPPORT_MIN,
        config.VALUE_SUPPORT_MAX, config.VALUE_SUPPORT_BINS)
    td_errors = torch.abs(pred_values_scalar.detach().squeeze(-1) - value_targets[:, 0])

    final_policy_loss = (policy_loss * is_weights).mean()
    final_value_loss = (value_loss * is_weights).mean()
    final_moves_left_loss = (moves_left_loss * is_weights).mean()

    weighted_loss = (
        loss_weights['policy'] * final_policy_loss
        + loss_weights['value'] * final_value_loss
        + loss_weights['moves_left'] * final_moves_left_loss
    )

    log_values = (
        weighted_loss.item(),
        final_policy_loss.item(),
        final_value_loss.item(),
        0.0,  # reward loss (not used by AlphaZero)
        0.0,  # consistency loss (not used by AlphaZero)
        td_errors.cpu().numpy(),
    )
    return weighted_loss, log_values
