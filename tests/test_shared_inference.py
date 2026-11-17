import multiprocessing as mp
from queue import Empty, Queue
import unittest

import numpy as np

from shared_inference import SharedInferenceTransport


def shared_server(transport, requests):
    worker, kind, sequence = requests.get(timeout=10)
    assert kind == 'shared_initial'
    value = float(transport.observations[worker].sum())
    transport.publish(worker, sequence, np.arange(225, dtype=np.float32), value)


class TestSharedInference(unittest.TestCase):
    def test_spawn_roundtrip_uses_shared_arrays(self):
        context = mp.get_context('spawn')
        transport = SharedInferenceTransport(context, 2, 15)
        requests = context.Queue()
        client = transport.client(1, requests)
        process = context.Process(target=shared_server, args=(transport, requests))
        process.start()
        try:
            obs = np.zeros((3,15,15),dtype=np.float32)
            obs[0,7,7] = 1
            client.put((1,'initial',obs))
            policy,value,hidden = client.get(timeout=10)
            np.testing.assert_array_equal(policy,np.arange(225,dtype=np.float32))
            self.assertEqual(value,1)
            self.assertIsNone(hidden)
            with self.assertRaises(Empty):
                client.get_nowait()
            process.join(timeout=5)
            self.assertEqual(process.exitcode,0)
        finally:
            if process.is_alive():
                process.terminate()
                process.join()
            requests.close()

    def test_late_reply_does_not_satisfy_new_request(self):
        transport = SharedInferenceTransport(mp.get_context('spawn'),1,15)
        requests = Queue()
        client = transport.client(0,requests)
        obs = np.zeros((3,15,15),dtype=np.float32)
        client.put((0,'initial',obs))
        client.put((0,'initial',obs))
        transport.publish(0,1,np.zeros(225),-1)
        with self.assertRaises(Empty):
            client.get_nowait()
        transport.publish(0,2,np.ones(225)/225,0.5)
        self.assertEqual(client.get_nowait()[1],0.5)
