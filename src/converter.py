import torch
import numpy as np


PIXELS_PER_WORLD = 5.5
HEIGHT = 144
WIDTH = 256
FOV = 90
MAP_SIZE = 256
CAM_HEIGHT = 1.3 * 2


class Converter(torch.nn.Module):
    def __init__(
            self, w=WIDTH, h=HEIGHT, fov=FOV,
            map_size=MAP_SIZE, pixels_per_world=PIXELS_PER_WORLD,
            hack=4, cam_height=CAM_HEIGHT):
        super().__init__()

        F = w / (2 * np.tan(fov * np.pi / 360))
        A = np.array([
            [F, 0, w/2],
            [0, F, h/2],
            [0, 0,   1]
        ])

        self.map_size = map_size
        self.pixels_per_world = pixels_per_world
        self.w = w
        self.h = h
        self.f = F
        self.hack = hack
        self.cam_height = cam_height

        self.register_buffer('A', torch.FloatTensor(A))
        self.register_buffer('A_inv', torch.FloatTensor(np.linalg.inv(A)))
        self.register_buffer('pos_map', torch.FloatTensor([map_size // 2, map_size]))

    def forward(self, map_coords):
        return self.map_to_cam(map_coords)

    def map_to_cam(self, map_coords):
        world_coords = self.map_to_world(map_coords)
        cam_coords = self.world_to_cam(world_coords)

        return cam_coords

    def map_to_world(self, pixel_coords):
        relative_pixel = pixel_coords - self.pos_map
        relative_pixel[..., 1] *= -1

        return relative_pixel / self.pixels_per_world

    def cam_to_map(self, points):
        world_coords = self.cam_to_world(points)
        map_coords = self.world_to_map(world_coords)

        return map_coords

    def cam_to_world(self, points):
        xt = (points[..., 0] - self.A[0, 2]) / self.A[0, 0]
        yt = (points[..., 1] - self.A[1, 2]) / self.A[1, 1]

        world_z = self.cam_height / (yt + 1e-8)
        world_x = world_z * xt

        world_output = torch.stack([world_x, world_z], -1)
        world_output[..., 1] -= self.hack
        world_output = world_output.squeeze()

        return world_output

    def world_to_cam(self, world_coords):
        world_x = world_coords[..., 0].reshape(-1)
        world_y = world_coords[..., 1].reshape(-1) + self.hack
        world_z = torch.FloatTensor(world_x.shape[0] * [self.cam_height])
        world_z = world_z.type_as(world_coords)

        xyz = torch.stack([world_x, world_z, world_y], -1)

        result = xyz.matmul(self.A.T)
        result = result[:, :2] / result[:, -1].unsqueeze(1)

        result[:, 0] = torch.clamp(result[:, 0], 0, self.w)
        result[:, 1] = torch.clamp(result[:, 1], 0, self.h)

        return result.reshape(*world_coords.shape)

    def world_to_map(self, world):
        pixel = world * self.pixels_per_world
        pixel[..., 1] *= -1

        pixel_coords = pixel + self.pos_map

        return pixel_coords
