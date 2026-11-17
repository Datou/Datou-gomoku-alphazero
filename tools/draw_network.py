"""Draw the AlphaZero Renju network diagram used in README.md."""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


def box(ax, x, y, w, h, text, face, edge, fontsize=11, weight='normal'):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle='round,pad=0.035,rounding_size=0.16',
        linewidth=1.6, edgecolor=edge, facecolor=face, zorder=2,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha='center', va='center',
            fontsize=fontsize, color='#17202a', linespacing=1.45,
            fontweight=weight, zorder=3)


def arrow(ax, start, end, color='#475569', rad=0.0):
    ax.add_patch(FancyArrowPatch(
        start, end, arrowstyle='-|>', mutation_scale=15,
        linewidth=1.8, color=color, zorder=1,
        connectionstyle=f'arc3,rad={rad}',
    ))


fig, ax = plt.subplots(figsize=(16, 9))
ax.set_xlim(0, 16)
ax.set_ylim(0, 9)
ax.axis('off')

ax.text(8, 8.72, 'AlphaZero Renju Network  (3 × 15 × 15 input)',
        ha='center', va='center', fontsize=19, fontweight='bold', color='#0f172a')

box(ax, 0.25, 3.10, 2.05, 2.75,
    'Input\n3 × 15 × 15\n\n• current player\n• opponent\n• last move',
    '#e8f1ff', '#3b82f6', fontsize=11.3, weight='bold')

box(ax, 2.55, 3.50, 1.80, 2.05,
    'Stem\n3×3 Conv\n128 filters\nBN + ReLU',
    '#eef2f7', '#64748b', fontsize=11.5)

box(ax, 4.65, 2.95, 2.15, 3.10,
    'Shared trunk\n\n8 × SE-ResBlock\n128 channels\n\nreceptive field\nfor local threats',
    '#e8f8ef', '#10b981', fontsize=12, weight='bold')

box(ax, 7.25, 6.45, 3.65, 1.95,
    'Policy head\n1×1 Conv (2) → BN → ReLU\nFlatten 450 → Dropout 0.1\nFC 225\n→ 225 policy logits',
    '#fff1e6', '#f97316', fontsize=11.5)

box(ax, 7.25, 3.40, 3.65, 2.35,
    'Value head\n1×1 Conv (1) → BN → ReLU\nFlatten 225\n+ global avg pool 128\nconcat 353\nFC 64 → ReLU → FC 3\nsoftmax → E[z]  (bins -1, 0, +1)',
    '#f3e8ff', '#8b5cf6', fontsize=11.2)

box(ax, 7.25, 0.55, 3.65, 2.05,
    'Moves-left head (aux)\nconcat 353 → FC 64 → ReLU\n→ FC 1\nlog1p target · loss weight 0.05',
    '#e6f7f8', '#0891b2', fontsize=11.2)

box(ax, 11.30, 2.35, 4.25, 4.30,
    'Inside one SE-ResBlock  (×8)\n\n'
    'x ─ 3×3 Conv 128 → BN → ReLU\n'
    '    3×3 Conv 128 → BN\n'
    '    SE: GAP → FC 8 → ReLU\n'
    '         FC 128 → Sigmoid\n'
    '    scale channels\n\n'
    'out = ReLU(x + scale(conv(x)))',
    '#fff7e6', '#d97706', fontsize=10.8)

arrow(ax, (2.32, 4.48), (2.52, 4.48))
arrow(ax, (4.37, 4.48), (4.62, 4.48))
arrow(ax, (6.84, 4.80), (7.20, 7.35), rad=-0.16)
arrow(ax, (6.84, 4.50), (7.20, 4.55), rad=0.0)
arrow(ax, (6.84, 4.18), (7.20, 1.65), rad=0.16)

ax.text(8.0, 0.12,
        'Training: policy + value + moves-left.  Playing: policy + value only.  '
        'All 3×3 convolutions use padding=1.',
        ha='center', va='center', fontsize=9.5, color='#64748b')

output = Path(__file__).resolve().parents[1] / 'docs' / 'network_architecture.png'
output.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(output, dpi=200, bbox_inches='tight', facecolor='white')
print(output)
