import uuid
import argparse
import pathlib

import wandb
import torch
import torchvision
import pytorch_lightning as pl

from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from .models import SegmentationModel
from .dataset import get_dataset
from . import common

import numpy as np
from PIL import Image, ImageDraw


@torch.no_grad()
def visualize(batch, out, loss):
    images = list()

    for i in range(out.shape[0]):
        _loss = loss[i]
        _out = out[i]
        rgb, topdown, points, meta = [x[i] for x in batch]

        _rgb = np.uint8(rgb.detach().cpu().numpy().transpose(1, 2, 0) * 255)
        _topdown = Image.fromarray(common.COLOR[topdown.argmax(0).detach().cpu().numpy()])
        _draw = ImageDraw.Draw(_topdown)
        _draw.text((5, 10), 'Loss: %.3f' % _loss)
        _draw.text((5, 30), 'Meta: %s' % meta)

        for x, y in points:
            x = (x + 1) / 2 * 256
            y = (y + 1) / 2 * 256

            _draw.ellipse((x-2, y-2, x+2, y+2), (0, 0, 255))

        for x, y in _out:
            x = (x + 1) / 2 * 256
            y = (y + 1) / 2 * 256

            _draw.ellipse((x-2, y-2, x+2, y+2), (255, 0, 0))

        _topdown.thumbnail((128, 128))

        image = np.array(_topdown).transpose(2, 0, 1)
        images.append((_loss, torch.ByteTensor(image)))

    images.sort(key=lambda x: x[0], reverse=True)

    result = torchvision.utils.make_grid([x[1] for x in images], nrow=4)
    result = wandb.Image(result.numpy().transpose(1, 2, 0))

    return result


class MapModel(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()

        self.hparams = hparams
        self.net = SegmentationModel(9, 4)

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_nb):
        img, topdown, points, meta = batch
        out = self.forward(topdown)

        loss = torch.nn.functional.l1_loss(out, points, reduction='none').mean((1, 2))
        loss_mean = loss.mean()

        metrics = {'train_loss': loss_mean.item()}

        if batch_nb % 250 == 0:
            metrics['train_image'] = visualize(batch, out, loss)

        self.logger.log_metrics(metrics, self.global_step)

        return {'loss': loss_mean}

    def validation_step(self, batch, batch_nb):
        img, topdown, points, meta = batch
        out = self.forward(topdown)

        loss = torch.nn.functional.l1_loss(out, points, reduction='none').mean((1, 2))
        loss_mean = loss.mean()

        if batch_nb == 0:
            self.logger.log_metrics({
                'val_image': visualize(batch, out, loss)
                }, self.global_step)

        return {'val_loss': loss_mean.item()}

    def validation_epoch_end(self, outputs):
        results = {'val_loss': list()}

        for output in outputs:
            for key in results:
                results[key].append(output[key])

        summary = {key: np.mean(val) for key, val in results.items()}
        self.logger.log_metrics(summary, self.global_step)

        return summary

    def configure_optimizers(self):
        return torch.optim.Adam(
                self.net.parameters(),
                lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)

    def train_dataloader(self):
        return get_dataset(self.hparams.dataset_dir, True, self.hparams.batch_size)

    def val_dataloader(self):
        return get_dataset(self.hparams.dataset_dir, False, self.hparams.batch_size)


def main(hparams):
    model = MapModel(hparams)
    logger = WandbLogger(id=hparams.id, save_dir=str(hparams.save_dir), project='topdown')
    checkpoint_callback = ModelCheckpoint(hparams.save_dir, save_top_k=2)

    try:
        resume_from_checkpoint = sorted(hparams.save_dir.glob('*.ckpt'))[-1]
    except:
        resume_from_checkpoint = None

    trainer = pl.Trainer(
            gpus=1, max_epochs=10,
            resume_from_checkpoint=resume_from_checkpoint,
            logger=logger, checkpoint_callback=checkpoint_callback)

    trainer.fit(model)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_epoch', type=int, default=50)
    parser.add_argument('--save_dir', type=pathlib.Path, default='checkpoints')
    parser.add_argument('--id', type=str, default=uuid.uuid4().hex)

    # Data args.
    parser.add_argument('--dataset_dir', type=pathlib.Path, required=True)
    parser.add_argument('--batch_size', type=int, default=16)

    # Optimizer args.
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=5e-6)

    parsed = parser.parse_args()
    parsed.save_dir = parsed.save_dir / parsed.id
    parsed.save_dir.mkdir(parents=True, exist_ok=True)

    main(parsed)
