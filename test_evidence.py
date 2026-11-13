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
if __name__=='__main__':unittest.main()
