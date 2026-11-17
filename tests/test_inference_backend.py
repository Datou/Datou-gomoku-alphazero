import copy
import unittest

import torch

from config import config
from network import GomokuNetEZ
from inference_backend import InferenceBackend


class TestInferenceBackend(unittest.TestCase):
    def setUp(self):
        self.settings = copy.copy(config)
        self.settings.NUM_RES_BLOCKS = 2
        self.settings.NUM_FILTERS = 32

    def test_cpu_fallback_matches_eager_and_checks_batch_size(self):
        model = GomokuNetEZ(self.settings).cpu().eval()
        obs = torch.zeros(3, 3, 15, 15)
        expected = model.initial_inference(obs)
        backend = InferenceBackend(model, 'fp16', True, True, max_batch_size=4)
        actual = backend.predict(obs)
        for before, after in zip(expected, actual):
            torch.testing.assert_close(before, after)
        self.assertFalse(backend.cuda_graphs)
        with self.assertRaises(ValueError):
            backend.predict(torch.zeros(5, 3, 15, 15))

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA is required')
    def test_graph_padding_precision_and_live_weight_updates(self):
        torch.manual_seed(42)
        model = GomokuNetEZ(self.settings).cuda().eval()
        raw = torch.rand(5, 15, 15, device='cuda')
        obs = torch.stack([(raw < 0.2).float(), (raw > 0.8).float(), torch.zeros_like(raw)], dim=1)
        with torch.no_grad():
            expected_p, expected_v, _ = model.initial_inference(obs)
        backend = InferenceBackend(model, 'fp16', True, True, max_batch_size=8)
        p, v = backend.predict(obs)
        torch.testing.assert_close(p, expected_p, atol=0.01, rtol=0.01)
        torch.testing.assert_close(v, expected_v, atol=0.005, rtol=0.01)
        before = p.clone()
        # Updating weights in-place must update captured inference too: stale
        # autocast weight caches would silently freeze the self-play model.
        weights = {name:tensor.clone() for name,tensor in model.state_dict().items()}
        weights['prediction_net.policy_fc.bias'] += 0.25
        model.load_state_dict(weights)
        after, _ = backend.predict(obs)
        torch.testing.assert_close(after - before, torch.full_like(before, 0.25), atol=0.01, rtol=0)
        small, _ = backend.predict(obs[:3])
        torch.testing.assert_close(small, after[:3], atol=0.01, rtol=0.01)
        host_obs = obs.cpu()
        host_p, host_v = backend.predict_cpu(host_obs)
        torch.testing.assert_close(host_p, after.cpu(), atol=0.01, rtol=0.01)
        # The captured host-to-device copy must consume the NEW observation,
        # and the captured device-to-host copy must deliver the NEW output.
        changed_obs = 1 - host_obs
        with torch.no_grad():
            changed_expected, _, _ = model.initial_inference(changed_obs.cuda())
        changed_p, _ = backend.predict_cpu(changed_obs)
        torch.testing.assert_close(changed_p, changed_expected.cpu(), atol=0.01, rtol=0.01)
        before_update = changed_p.clone()
        weights['prediction_net.policy_fc.bias'] += 0.125
        model.load_state_dict(weights)
        updated_p, _ = backend.predict_cpu(changed_obs)
        torch.testing.assert_close(updated_p - before_update,
            torch.full_like(before_update, 0.125), atol=0.01, rtol=0)

