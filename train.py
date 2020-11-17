import argparse
import random
import os
import getpass
import datetime
import shutil
import numpy as np
import torch
from torchvision.models import resnet34

from dataset import SegDataset, AlignCollate

train_dataset = SegDataset()
assert train_dataset

train_align_collate = AlignCollate(
    'training',
    ts.N_CLASSES,
    ts.MAX_N_OBJECTS,
    ts.MEAN,
    ts.STD,
    ts.IMAGE_HEIGHT,
    ts.IMAGE_WIDTH,
    random_hor_flipping=ts.HORIZONTAL_FLIPPING,
    random_ver_flipping=ts.VERTICAL_FLIPPING,
    random_transposing=ts.TRANSPOSING,
    random_90x_rotation=ts.ROTATION_90X,
    random_rotation=ts.ROTATION,
    random_color_jittering=ts.COLOR_JITTERING,
    random_grayscaling=ts.GRAYSCALING,
    random_channel_swapping=ts.CHANNEL_SWAPPING,
    random_gamma=ts.GAMMA_ADJUSTMENT,
    random_resolution=ts.RESOLUTION_DEGRADING)
