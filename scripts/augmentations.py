#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script contains some image augmentations by calling 'opencv' and 
'keras.preprocessing.image'. Currently, 4 kinds of augmentations: 
'Flip', 'Rotate', 'Shift', 'Zoom' are available.

@author: cttsai (Chia-Ta Tsai), @Oct 2017
"""

from random import choice
from random import random
import cv2
import numpy as np
import keras.preprocessing.image as prep

#data augmentations
###############################################################################
def HorizontalFlip(image, prob):

    if random() < prob:
        image = cv2.flip(image, 1)

    return image


def VerticalFlip(image, prob):

    if random() < prob:
        image = cv2.flip(image, 0)

    return image


def Rotate90(image, prob):

    if random() < prob:
        image = np.rot90(image, k=choice([0, 1, 2, 3]), axes=(0, 1))

    return image


def Rotate(image, prob, rotate_rg=10):

    if random() < prob:
        image = prep.random_rotation(
            image, rg=rotate_rg, row_axis=0, col_axis=1, channel_axis=2)

    return image


def Shift(image, prob, width_rg=0.1, height_rg=0.1):

    if random() < prob:
        image = prep.random_shift(
            image,
            wrg=width_rg,
            hrg=height_rg,
            row_axis=0,
            col_axis=1,
            channel_axis=2)

    return image


def Zoom(image, prob, zoom_rg=(0.1, 0.1)):

    if random() < prob:
        image = prep.random_zoom(
            image, zoom_range=zoom_rg, row_axis=0, col_axis=1, channel_axis=2)

    return image


def Noise(image, prob, noise_rg=0.02):
    """ Noise range must defines upper limit of noise amplitude"""
    if random() < prob:
        noise_amp = (image.max() - image.min()) * random() * noise_rg
        image = image + noise_amp*np.random.normal(size=image.shape)
    return image



def augment(x, **config):
    """ Mutates depending on config specifications """
    hflip_prob = config.get('hflip_prob')
    vflip_prob = config.get('vflip_prob')
    rot90_prob = config.get('rot90_prob')
    rot_prob   = config.get('rot_prob')
    rotate_rg  = config.get('rotate_rg')
    shift_prob = config.get('shift_prob')
    shift_width_rg = config.get('shift_width_rg')
    shift_height_rg = config.get('shift_height_rg')
    zoom_prob  = config.get('zoom_prob')
    zoom_rg    = config.get('zoom_rg')
    noise_prob = config.get('noise_prob')
    noise_rg   = config.get('noise_rg')

    if hflip_prob > 0:
        x = HorizontalFlip(x, hflip_prob)
    if vflip_prob > 0:
        x = VerticalFlip(x, vflip_prob)
    if rot90_prob > 0:
        x = Rotate90(x, rot90_prob)
    if rot_prob > 0:
        x = Rotate(x, rot_prob, rotate_rg)
    if shift_prob > 0:
        x = Shift(x, shift_prob, shift_width_rg, shift_height_rg)
    if zoom_prob > 0:
        x = Zoom(x, zoom_prob, zoom_rg)
    if noise_prob > 0:
        x = Noise(x, noise_prob, noise_rg)
        
    return x

