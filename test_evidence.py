import argparse
import unittest
import numpy as np
from evidence_lab import quality,run
class EvidenceTests(unittest.TestCase):
    def test_wrong_ranking_fails(self):
        a=np.array([[1.,0.,-1.]])
        self.assertEqual(quality(a,-a,1)['recall_at_k'],0)
        self.assertFalse(quality(a,-a,1)['scores_close'])
    def test_remainder_and_reproducibility(self):
        a=argparse.Namespace(docs=31,queries=7,dim=9,batch=3,repeats=2,warmup=1,k=3,seed=42,backend='cpu',output='unused')
        x,y=run(a),run(a)
        self.assertEqual(x['dataset_sha256'],y['dataset_sha256'])
        self.assertTrue(all(r['accepted'] for r in x['results']))
        self.assertTrue(all(len(r['samples_ms'])==2 for r in x['results']))
        self.assertFalse(x['environment']['gpu_verified'])
        a.k=32
        with self.assertRaises(ValueError):run(a)

class ScaleTests(unittest.TestCase):
    def test_chunked_generation_matches_original(self):
        from evidence_lab import normalized_vectors
        rng = np.random.default_rng(42)
        expected = rng.normal(size=(4101, 9)).astype('float32')
        expected /= np.linalg.norm(expected, axis=1, keepdims=True)
        actual = normalized_vectors(np.random.default_rng(42), 4101, 9)
        np.testing.assert_array_equal(expected, actual)

    def test_rowwise_quality_and_ties(self):
        x = np.array([[1., 1., 0.], [0., 2., 1.]])
        self.assertEqual(quality(x, x.copy(), 2)['recall_at_k'], 1.)
        self.assertFalse(quality(x, x + 1, 2)['scores_close'])

    def test_million_vector_plan_and_preallocation_rejection(self):
        from evidence_lab import memory_plan
        from unittest.mock import patch
        a = argparse.Namespace(docs=1000000, queries=64, dim=384, batch=32,
            repeats=1, warmup=0, k=10, seed=42, backend='cpu', memory_budget_mib=1)
        plan = memory_plan(a)
        self.assertGreater(plan['corpus_mib'], 1400)
        self.assertLess(plan['estimated_host_mib'], 4096)
        with patch('evidence_lab.np.random.default_rng', side_effect=AssertionError('allocated')):
            with self.assertRaisesRegex(ValueError, 'Memory estimate'):
                run(a)

if __name__=='__main__':unittest.main()
