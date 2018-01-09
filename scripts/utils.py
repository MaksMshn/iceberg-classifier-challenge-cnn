#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This scripts contains some data processing functions

@author: (Chia-Ta Tsai), @Oct 2017
"""

import pandas as pd
import numpy as np


##### Preprcessings

def band2(df, **config):
    """Just put two channels together"""
    band1 = np.array([
        np.array(band).astype(np.float32).reshape(75, 75)
        for band in df["band_1"]
    ])
    band2 = np.array([
        np.array(band).astype(np.float32).reshape(75, 75)
        for band in df["band_2"]
    ])
    return np.stack((band1, band2), axis=-1)


def band3(df, **config):
    """operations on the 3rd band should be defined as a func in config"""

    band3_op = config.get('band3_op', 'lambda x1, x2: (x1+x2)/2')

    band1 = np.array([
        np.array(band).astype(np.float32).reshape(75, 75)
        for band in df["band_1"]
    ])
    band2 = np.array([
        np.array(band).astype(np.float32).reshape(75, 75)
        for band in df["band_2"]
    ])
    band3 = eval(band3_op)(band1, band2)
    return np.stack((band1, band2, band3), axis=-1)


def create_dataset(
        file,
        labeled,
        loc='../input',
        **config):
    """
        labeled: boolean Train or Test
    """

    inc_angle_fill = config.get('inc_angle_fill', -1)
    preproc_func = eval(config.get('preproc_strat', 'band3'))

    if config.get('soft_targets'):
        soft_val = config.get('soft_val')
    else:
        soft_val = None

    df = pd.read_json('{}/{}'.format(loc, file))
    df['inc_angle'] = df['inc_angle'].replace('na', inc_angle_fill).astype(float)

    bands = preproc_func(df, **config)

    print('Loaded data from: {}'.format(file), flush=True)

    if labeled:
        y = np.array(df["is_iceberg"])
        if soft_val:
            y = np.clip(1 - soft_val, soft_val, y)
    else:
        y = df.id
    return y, bands, df['inc_angle']


if __name__ == '__main__':
    label, data, meta = create_dataset('train.json', True)
    print(label.shape, data.shape, meta.shape)
