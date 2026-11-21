import csv
import hashlib
import json
from pathlib import Path
import sys
import tarfile
import tempfile
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from run_rectangular_server import training_complete, pack_results


class TestServerExport(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)/'experiment'
        self.root.mkdir()

    def test_return_archive_has_verified_weights_without_optimizer_and_keeps_original(self):
        (self.root/'summary.json').write_text('{}')
        training = self.root/'training'
        training.mkdir()
        checkpoint = training/'best.pth'
        torch.save(dict(model={'weight':torch.ones(2)}, optimizer={'large':torch.ones(100)}, epoch=29),checkpoint)
        original = checkpoint.read_bytes()
        (self.root/'smoke_training').mkdir()
        (self.root/'smoke_training/unwanted.pth').write_bytes(b'smoke')
        pack_results(self.root)
        self.assertEqual(checkpoint.read_bytes(),original)
        with tarfile.open(self.root.with_suffix('.tar.gz')) as tar:
            self.assertFalse(any('smoke_training' in name for name in tar.getnames()))
            sums = tar.extractfile('experiment/SHA256SUMS').read().decode()
            for line in sums.splitlines():
                digest,name = line.split('  ',1)
                self.assertEqual(hashlib.sha256(tar.extractfile('experiment/'+name).read()).hexdigest(),digest)
            import io
            data = torch.load(io.BytesIO(tar.extractfile('experiment/training/best.pth').read()),weights_only=False)
            self.assertNotIn('optimizer',data)
            torch.testing.assert_close(data['model']['weight'],torch.ones(2))
            self.assertEqual(data['original_checkpoint_sha256'],hashlib.sha256(original).hexdigest())

    def test_incomplete_or_nonfinite_training_is_rejected(self):
        config = dict(epochs=2,train_samples=10,eval_samples=4,input_contract_version=2,
            crop_mode='rect',crop_padding=1.,crop_border=0,backbone='resnet18',fpn_dim=128,
            loss='spatial_kl',decoder='local',no_pretrained=False,init_checkpoint=None,
            device='cuda',amp_dtype='bfloat16')
        (self.root/'config.json').write_text(json.dumps(config))
        (self.root/'metrics.csv').write_text('epoch,train_loss\n1,1.0\n')
        with self.assertRaisesRegex(ValueError,'epoch count'):
            training_complete(self.root,2,10,4)
        (self.root/'metrics.csv').write_text('epoch,train_loss\n1,1.0\n2,nan\n')
        with self.assertRaisesRegex(ValueError,'Non-finite'):
            training_complete(self.root,2,10,4)

    def test_wrong_protocol_cannot_be_exported_as_rectangular_training(self):
        (self.root/'config.json').write_text(json.dumps(dict(epochs=30,train_samples=40000,
            eval_samples=3048,input_contract_version=1)))
        with self.assertRaisesRegex(ValueError,'contract differs'):
            training_complete(self.root,30,40000,3048)

    def test_photometric_recipe_cannot_silently_run_without_augmentation(self):
        config=dict(epochs=30,train_samples=40000,eval_samples=3048,input_contract_version=2,
            crop_mode='rect',crop_padding=1.,crop_border=0,backbone='resnet18',fpn_dim=128,
            loss='spatial_kl',decoder='local',no_pretrained=False,init_checkpoint=None,
            device='cuda',amp_dtype='bfloat16',photometric_augmentation=False)
        (self.root/'config.json').write_text(json.dumps(config))
        with self.assertRaisesRegex(ValueError,'photometric_augmentation'):
            training_complete(self.root,30,40000,3048,dict(photometric_augmentation=True,
                photometric_profile='balanced'))


if __name__ == '__main__':
    unittest.main()
