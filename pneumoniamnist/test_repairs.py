"""Small regression checks; no dataset download or training run."""
import tempfile
import unittest
import random
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from runtime_utils import rng_state, restore_rng
from train_replication import ResNet18
from run_residual_ablation import PlainNet18
from run_weighted_experiment import evaluate
from run_focal_experiment import FocalLoss

class RepairTests(unittest.TestCase):
    def test_weighted_epoch_loss(self):
        logits=torch.tensor([[2.,0.],[1.,3.],[0.,1.],[3.,0.],[0.,4.]])
        y=torch.tensor([[0],[1],[0],[0],[1]])
        criterion=torch.nn.CrossEntropyLoss(weight=torch.tensor([2.878,1.]))
        expected=criterion(logits,y[:,0]).item()
        for batch in [1,2,3,5]:
            actual=evaluate(torch.nn.Identity(),DataLoader(TensorDataset(logits,y),batch_size=batch),criterion,'cpu')[0]
            self.assertAlmostEqual(expected,actual,places=6)

    def test_rng_serialization(self):
        state=rng_state()
        expected=(torch.rand(3),np.random.rand(3),random.random())
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'rng.pt';torch.save(state,path)
            restore_rng(torch.load(path,weights_only=True))
        self.assertTrue(torch.equal(expected[0],torch.rand(3)))
        self.assertTrue(np.array_equal(expected[1],np.random.rand(3)))
        self.assertEqual(expected[2],random.random())

    def test_architectures(self):
        for cls,count in [(ResNet18,11168706),(PlainNet18,10994882)]:
            model=cls().eval()
            self.assertEqual(sum(p.numel() for p in model.parameters()),count)
            logits=model(torch.zeros(1,1,28,28))
            self.assertEqual(tuple(logits.shape),(1,2))
            logits.sum().backward()
            self.assertTrue(torch.isfinite(model.conv1.weight.grad).all())

    def test_focal_reduces_to_ce(self):
        x=torch.tensor([[2.,1.],[1.,3.]])
        y=torch.tensor([0,1])
        self.assertTrue(torch.allclose(FocalLoss(gamma=0)(x,y),torch.nn.functional.cross_entropy(x,y)))

if __name__=='__main__':
    torch.set_num_threads(4)
    unittest.main()
