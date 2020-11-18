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
from setting import Settings

settings = Settings()
train_dataset = SegDataset()
assert train_dataset

train_align_collate = AlignCollate(
    'training',
    settings.N_CLASSES,
    settings.MAX_N_OBJECTS,
    settings.MEAN,
    settings.STD,
    settings.IMAGE_HEIGHT,
    settings.IMAGE_WIDTH)

train_loader = torch.utils.data.DataLoader(train_dataset,
                                           batch_size = 8,
                                           shuffle = True,
                                           collate_fn = train_align_collate)
