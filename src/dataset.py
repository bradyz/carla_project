from pathlib import Path

import numpy as np
import torch
import imgaug.augmenters as iaa
import pandas as pd

from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

from .dataset_wrapper import Wrap
from . import common


# Reproducibility.
np.random.seed(0)
torch.manual_seed(0)

# Data has frame skip of 5.
GAP = 2
STEPS = 4
N_CLASSES = len(common.COLOR)


def get_dataset(dataset_dir, is_train=True, batch_size=128, num_workers=4, **kwargs):
    data = list()
    transform = transforms.Compose([
        get_augmenter() if is_train else lambda x: x,
        transforms.ToTensor()
        ])

    for _dataset_dir in sorted(Path(dataset_dir).glob('*')):
        add = False
        add |= (is_train and int(_dataset_dir.stem) % 10 < 8)
        add |= (not is_train and int(_dataset_dir.stem) % 10 >= 8)

        if add:
            data.append(CarlaDataset(_dataset_dir, transform, **kwargs))

    print('%d frames.' % sum(map(len, data)))

    data = torch.utils.data.ConcatDataset(data)
    data = Wrap(data, batch_size, 1000 if is_train else 100, num_workers)

    return data


def get_augmenter():
    seq = iaa.Sequential([
        iaa.Sometimes(0.05, iaa.GaussianBlur((0.0, 1.3))),
        iaa.Sometimes(0.05, iaa.AdditiveGaussianNoise(scale=(0.0, 0.05 * 255))),
        iaa.Sometimes(0.05, iaa.Dropout((0.0, 0.1))),
        iaa.Sometimes(0.10, iaa.Add((-0.05 * 255, 0.05 * 255), True)),
        iaa.Sometimes(0.20, iaa.Add((0.25, 2.5), True)),
        iaa.Sometimes(0.05, iaa.contrast.LinearContrast((0.5, 1.5))),
        iaa.Sometimes(0.05, iaa.MultiplySaturation((0.0, 1.0))),
        ])

    return seq.augment_image


class CarlaDataset(Dataset):
    def __init__(self, dataset_dir, transform=transforms.ToTensor()):
        dataset_dir = Path(dataset_dir)

        self.transform = transform
        self.dataset_dir = dataset_dir
        self.frames = list()
        self.measurements = pd.read_csv(dataset_dir / 'measurements.csv')

        for image_path in sorted((dataset_dir / 'rgb').glob('*.png')):
            frame = str(image_path.stem)

            assert (dataset_dir / 'rgb_left' / ('%s.png' % frame)).exists()
            assert (dataset_dir / 'rgb_right' / ('%s.png' % frame)).exists()
            assert (dataset_dir / 'map' / ('%s.png' % frame)).exists()

            self.frames.append(frame)

    def __len__(self):
        return len(self.frames) - GAP * STEPS

    def __getitem__(self, i):
        path = self.dataset_dir
        frame = self.frames[i]
        meta = '%s %s' % (path.stem, frame)

        rgb = Image.open(path / 'rgb' / ('%s.png' % frame))
        rgb = transforms.functional.to_tensor(rgb)

        topdown = Image.open(path / 'map' / ('%s.png' % frame))
        topdown = topdown.crop((128, 0, 128 + 256, 256))
        topdown = np.array(topdown)
        topdown = common.CONVERTER[topdown]
        topdown = torch.LongTensor(topdown)
        topdown = torch.nn.functional.one_hot(topdown, N_CLASSES).permute(2, 0, 1).float()

        u = np.array(self.measurements.iloc[i][['x', 'y']])
        theta = np.radians(90 + self.measurements.iloc[i]['theta'])
        R = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta),  np.cos(theta)],
            ])

        points = list()

        for skip in range(1, STEPS+1):
            j = i + GAP * skip
            v = np.array(self.measurements.iloc[j][['x', 'y']])

            target = R.T.dot(v - u)
            target *= 5.5
            target += [128, 256]

            points.append(target)

        points = torch.FloatTensor(points)
        points = torch.clamp(points, 0, 256)
        points = (points / 256) * 2 - 1

        return rgb, topdown, points, meta


if __name__ == '__main__':
    import sys
    import cv2
    from PIL import ImageDraw

    data = CarlaDataset(sys.argv[1])

    for i in range(len(data)):
        rgb, topdown, points, meta = data[i]

        _rgb = np.uint8(rgb.detach().cpu().numpy().transpose(1, 2, 0) * 255)
        _topdown = Image.fromarray(common.COLOR[topdown.argmax(0).detach().cpu().numpy()])
        _draw = ImageDraw.Draw(_topdown)

        for x, y in points:
            x = (x + 1) / 2 * 256
            y = (y + 1) / 2 * 256

            _draw.ellipse((x-2, y-2, x+2, y+2), (255, 0, 0))

        cv2.imshow('map', cv2.cvtColor(_rgb, cv2.COLOR_BGR2RGB))
        cv2.imshow('rgb', cv2.cvtColor(np.array(_topdown), cv2.COLOR_BGR2RGB))
        cv2.waitKey(10)
