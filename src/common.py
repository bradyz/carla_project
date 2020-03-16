import numpy as np


def colorize_segmentation(segmentation):
    colors = np.uint8([
            (  0,   0,   0),    # unlabeled
            ( 70,  70,  70),    # building
            (190, 153, 153),    # fence
            (250, 170, 160),    # other
            (220,  20,  60),    # ped
            (153, 153, 153),    # pole
            (157, 234,  50),    # road line
            (128,  64, 128),    # road
            (244,  35, 232),    # sidewalk
            (107, 142,  35),    # vegetation
            (  0,   0, 142),    # car
            (102, 102, 156),    # wall
            (220, 220,   0),    # traffic sign
            (255,   0,   0),
            (255, 255,   0),
            (  0, 255,   0),
            ])

    return colors[segmentation]
