
from __future__ import print_function, division

import shutil

import os
import sys
import numpy as np
import glob
import random
from PIL import Image
from tqdm import tqdm
import argparse
from bisect import bisect_right, bisect_left
import pandas as pd
from skimage import io, transform
import time
from datetime import datetime

import matplotlib.pyplot as plt

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")


def return_idx(timestamps, data):
    "Return the idx of the timestamp of the camera"
    
    idx_sensor = []
    for temp in timestamps:
        idx=int(data[data['timestamp']==temp].index[0])
        idx_sensor.append(idx)
    return idx_sensor


def return_idx_single(temp, data):
    "Return the idx of the timestamp of the camera"
    
    idx_sensor = []
    idx = data[data['timestamp']==temp].index[0]

    return idx


def quaternion_angular_error(q1, q2):
    """
    angular error between two quaternions
    :param q1: (4, )
    :param q2: (4, )
    :return:
    """
    d = abs(np.dot(q1, q2))
    abs_q1 = np.linalg.norm(q1)
    abs_q2 = np.linalg.norm(q2)
    d = d / (abs_q1 * abs_q2)
    d = min(1.0, max(-1.0, d))
    theta = 2 * np.arccos(d) * 180 / np.pi
    return theta