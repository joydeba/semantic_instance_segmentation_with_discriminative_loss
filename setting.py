import os
import numpy as np

class Settings(object):
    def __init__(self):
        #data setting
        self.MAX_N_OBJECTS = 20
        self.N_CLASSES = 1 + 1

        #model setting
        self.IMAGE_HEIGHT = 512
        self.IMAGE_WIDTH = 512

        self.DELTA_VAR = 0.5
        self.DELTA_DIST = 1.5
        self.NORM = 2

        self.RGB_MEAN = [0.5216978443211246, 0.3897754262671106, 0.20621611439141693]
        self.RGB_STD = [0.21239829181879105, 0.15175542704118347, 0.11302210720350032]

        #training setting
        self.WEIGHT_DECAY = 0.001