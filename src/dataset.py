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


# https://github.com/guopei/PoseEstimation-FCN-Pytorch/blob/master/heatmap.py
def gaussian(img, pt, sigma=8):
    pt = [
            np.clip(pt[0], sigma // 2, img.shape[1]-sigma // 2),
            np.clip(pt[1], sigma // 2, img.shape[0]-sigma // 2)
            ]

    # Check that any part of the gaussian is in-bounds
    ul = [int(pt[0] - 3 * sigma), int(pt[1] - 3 * sigma)]
    br = [int(pt[0] + 3 * sigma + 1), int(pt[1] + 3 * sigma + 1)]

    # If not, just return the image as is
    # if (ul[0] > img.shape[1] or ul[1] >= img.shape[0] or
            # br[0] < 0 or br[1] < 0):
        # return img

    # Generate gaussian
    size = 6 * sigma + 1
    x = np.arange(0, size, 1, float)
    y = x[:, np.newaxis]
    x0 = y0 = size // 2

    # The gaussian is not normalized, we want the center value to equal 1
    g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))

    # Usable gaussian range
    g_x = max(0, -ul[0]), min(br[0], img.shape[1]) - ul[0]
    g_y = max(0, -ul[1]), min(br[1], img.shape[0]) - ul[1]

    # Image range
    img_x = max(0, ul[0]), min(br[0], img.shape[1])
    img_y = max(0, ul[1]), min(br[1], img.shape[0])

    img[img_y[0]:img_y[1], img_x[0]:img_x[1]] = g[g_y[0]:g_y[1], g_x[0]:g_x[1]]

    return img


def preprocess_semantic(semantic_np):
    topdown = common.CONVERTER[semantic_np]
    topdown = torch.LongTensor(topdown)
    topdown = torch.nn.functional.one_hot(topdown, N_CLASSES).permute(2, 0, 1).float()

    return topdown


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
        topdown = preprocess_semantic(topdown)

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

        heatmap = np.zeros((256, 256), dtype=np.float32)
        distance = np.linalg.norm(points[-1].round().numpy() - [128, 256])

        if distance < 10:
            gaussian(heatmap, [np.random.randint(64, 256-64), np.random.randint(32, 256-32)])
        else:
            try:
                gaussian(heatmap, points[-1])
            except:
                print(points[-1])

                gaussian(heatmap, [np.random.randint(64, 256-64), np.random.randint(32, 256-32)])

        heatmap = torch.FloatTensor(heatmap).unsqueeze(0)
        points = (points / 256) * 2 - 1

        return rgb, topdown, points, heatmap, meta


def heatmap_from_point(x, y, size=256):
    heatmap = np.zeros((256, 256), dtype=np.float32)
    gaussian(heatmap, (x, y))

    return heatmap


if __name__ == '__main__':
    import sys
    import cv2
    from PIL import ImageDraw

    data = CarlaDataset(sys.argv[1])

    for i in range(len(data)):
        rgb, topdown, points, heatmap, meta = data[i]

        _heatmap = np.uint8(heatmap.detach().cpu().squeeze().numpy() * 255)
        _rgb = np.uint8(rgb.detach().cpu().numpy().transpose(1, 2, 0) * 255)
        _topdown = Image.fromarray(common.COLOR[topdown.argmax(0).detach().cpu().numpy()])
        _draw = ImageDraw.Draw(_topdown)

        for x, y in points:
            x = (x + 1) / 2 * 256
            y = (y + 1) / 2 * 256

            _draw.ellipse((x-2, y-2, x+2, y+2), (255, 0, 0))

        cv2.imshow('heat', _heatmap)
        cv2.imshow('map', cv2.cvtColor(_rgb, cv2.COLOR_BGR2RGB))
        cv2.imshow('rgb', cv2.cvtColor(np.array(_topdown), cv2.COLOR_BGR2RGB))
        cv2.waitKey(10)
