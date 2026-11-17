"""Export an AlphaZero checkpoint to ONNX for the static browser player."""
import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from network import GomokuNetEZ, support_to_scalar


class ExportNet(torch.nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, obs):
        hidden = self.net.representation(obs)
        policy_logits, value_logits, moves_left = self.net.prediction(hidden)
        value = support_to_scalar(value_logits, self.net.value_support_min,
                                  self.net.value_support_max, self.net.value_support_bins)
        return policy_logits, value, moves_left


def export(checkpoint: Path, output: Path):
    params = json.loads(checkpoint.with_suffix('.json').read_text(encoding='utf-8'))
    net = GomokuNetEZ(SimpleNamespace(**params))
    net.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True))
    net.eval()
    wrapper = ExportNet(net).eval()
    dummy = torch.zeros(1, 3, params['BOARD_SIZE'], params['BOARD_SIZE'], dtype=torch.float32)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        wrapper, dummy, str(output), input_names=['obs'],
        output_names=['policy', 'value', 'moves_left'],
        dynamic_axes={'obs': {0: 'batch'}, 'policy': {0: 'batch'},
                      'value': {0: 'batch'}, 'moves_left': {0: 'batch'}},
        opset_version=17, do_constant_folding=True, dynamo=False)
    import onnx
    model = onnx.load(str(output))
    onnx.checker.check_model(model)
    print('exported', output, round(output.stat().st_size / 1e6, 2), 'MB')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    export(args.checkpoint.resolve(), args.output.resolve())
