"""Checkpoint architecture, rules and search settings travel together."""
import json
from pathlib import Path


FIELDS = ('NUM_RES_BLOCKS', 'NUM_FILTERS', 'HEAD_HIDDEN_DIM',
          'BOARD_SIZE', 'ACTION_SPACE_SIZE', 'VALUE_SUPPORT_MIN', 'VALUE_SUPPORT_MAX',
          'VALUE_SUPPORT_BINS', 'REWARD_SUPPORT_MIN', 'REWARD_SUPPORT_MAX',
          'REWARD_SUPPORT_BINS', 'ENABLE_RENJU', 'RULESET_ID', 'RULESET',
          'ENABLE_FORCED_MOVE_PRUNING', 'C_PUCT', 'ACTION_RADIUS')


def checkpoint_parameters(config):
    return {key: getattr(config, key) for key in FIELDS}


def read_checkpoint_parameters(checkpoint):
    checkpoint = Path(checkpoint).resolve()
    params = json.loads(checkpoint.with_suffix('.json').read_text(encoding='utf-8-sig'))
    output = checkpoint.parent.parent
    expected = None
    run = output / 'run_config.json'
    if not run.exists():
        run = output / 'training' / 'run_config.json'
    if run.exists():
        expected = json.loads(run.read_text(encoding='utf-8-sig'))['config']['RULESET_ID']
    warm = output / 'warmstart.json'
    if not warm.exists():
        warm = output / 'training' / 'warmstart.json'
    if warm.exists():
        target = json.loads(warm.read_text(encoding='utf-8-sig'))['target_ruleset_id']
        if expected is not None and expected != target:
            raise ValueError('Experiment rule manifests disagree.')
        expected = target
    actual = params.get('RULESET_ID')
    if expected == 'renju_forbidden_v1' and (actual != expected or params.get('ENABLE_RENJU') is not True):
        raise ValueError('Renju checkpoint lacks matching explicit rule metadata; repair metadata before evaluation.')
    if actual and actual != ('renju_forbidden_v1' if params.get('ENABLE_RENJU') else 'freestyle_v1'):
        raise ValueError('Checkpoint rules and ENABLE_RENJU disagree.')
    if expected and actual and expected != actual:
        raise ValueError('Checkpoint rules differ from its experiment.')
    if params.get('ENABLE_FORCED_MOVE_PRUNING', False):
        raise ValueError('Evaluation requires ordinary PUCT without tactical overrides.')
    return params
