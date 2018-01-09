#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script is for training a ConvNet model, which its structure is defined 
at 'models.py' with some parameters of structure and weights' location is 
set at 'param.py'. 

The ConvNet model takes multi-inputs: 1) TWO-channel with capability to perform 
augmentations from 'augmentations.py' and 2) meta info such as 'inc_angle'. 
Four types of augmentations: 'Flip', 'Rotate', 'Shift', 'Zoom' are available.

@author: cttsai (Chia-Ta Tsai), @Oct 2017
"""
import os
#
import numpy as np # linear algebra
import pandas as pd # data processing
import datetime as dt
#
from random import shuffle, uniform, seed
#evaluation
from sklearn.model_selection import train_test_split
#from sklearn.metrics import log_loss
#
from keras.optimizers import Adam, SGD
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, LambdaCallback
#
from augmentations import augment
import models

###############################################################################

def log_loss(t, p):
    h_tp = -(1-t)*np.log(1-p+1.0e-8)-t*np.log(p+1.0e-8)
    return np.mean(h_tp)


def data_generator(data,
                   meta_data,
                   labels,
                   **config):

    indices = [i for i in range(len(labels))]
    use_meta = config.get('use_meta', False)
    batch_size = config.get('batch_size', 16)

    while True:

        x_data = np.copy(data)
        if use_meta:
            x_meta_data = np.copy(meta_data)
        x_labels = np.copy(labels)

        for start in range(0, len(labels), batch_size):
            end = min(start + batch_size, len(labels))
            sel_indices = indices[start:end]

            #select data
            data_batch = x_data[sel_indices]
            if use_meta:
                xm_batch = x_meta_data[sel_indices]
            y_batch = x_labels[sel_indices]

            x_batch = []

            for x in data_batch:
                x = augment(x, **config)
                x_batch.append(x)

            x_batch = np.array(x_batch, dtype=np.float32)

            if use_meta:
                yield [x_batch, xm_batch], y_batch
            else:
                yield x_batch, y_batch


###############################################################################
def train(dataset,
          model,
          **config):
    """ 
    dataset:  (y_train, X_train, X_meta), ]
    """
    np.random.seed(1017)

    name = config.get('name', 'unnamed')
    epochs = config.get('epochs', 250)
    batch_size = config.get('batch_size', 32)
    lr_patience = config.get('lr_patience', 15)
    stop_patience = config.get('stop_patience', 50)
    use_meta = config.get('use_meta', False)
    full_cycls_per_epoch = config.get('full_cycls_per_epoch', 8)
    tmp = config.get('tmp')

    (labels, data, meta) = dataset

    weights_file = os.path.join("../weights/weights_{}_{}.hdf5".format(name, tmp))

    #training
    print('epochs={}, batch={}'.format(epochs, batch_size), flush=True)

    #train, validataion split
    test_ratio = 0.15
    split_seed = 27

    X_train, X_test, Xm_train, Xm_test, y_train, y_test = train_test_split(
        data, meta, labels, test_size=test_ratio, random_state=split_seed)

    print('splitted: {0}, {1}'.format(X_train.shape, X_test.shape), flush=True)
    print('splitted: {0}, {1}'.format(y_train.shape, y_test.shape), flush=True)

    #call backs
    earlystop = EarlyStopping(
        monitor='val_loss', patience=stop_patience, verbose=1, min_delta=1e-4)
    reduce_lr_loss = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.1,
        patience=lr_patience,
        verbose=1,
        epsilon=1e-4)
    model_chk = ModelCheckpoint(
        monitor='val_loss',
        filepath=weights_file,
        save_best_only=True,
        save_weights_only=False)
    flush_logger = LambdaCallback(on_epoch_end=\
        lambda epoch, logs: print(
            'Epoch: {}, '.format(epoch) + \
                ', '.join('{}: {:.4f}'.format(k, v) if v > 1e-3 else '{}: {:.4e}'.format(k, v)
                           for k, v in logs.items()),
            flush=True))

    callbacks = [earlystop, reduce_lr_loss, model_chk, flush_logger]
    ##########

    model.fit_generator(
        generator=data_generator(
            X_train,
            Xm_train,
            y_train,
            **config),
        steps_per_epoch=np.ceil(full_cycls_per_epoch * len(y_train) /(batch_size)),
        epochs=epochs,
        verbose=0,
        callbacks=callbacks,
        validation_data=(X_test,y_test))

    model.load_weights(weights_file)

    if use_meta:
        loss_val, acc_val = model.evaluate(
            [[X_test, Xm_test], y_test], batch_size=batch_size, verbose=0)
        loss_tr, acc_tr = model.evaluate(
            [[X_train, Xm_train], y_train], batch_size=batch_size, verbose=0)
    else:
        loss_val, acc_val = model.evaluate(
            X_test, y_test, batch_size=batch_size, verbose=0)
        loss_tr, acc_tr = model.evaluate(
            X_train, y_train, batch_size=batch_size, verbose=0)

    print(
        '\n\nLoss/Acc in validation data: {:.5f}/{:.5f}'.format(loss_val, acc_val),
        flush=True)
    print(
        'Loss/Acc in training data: {:.5f}/{:.5f}\n'.format(loss_tr, acc_tr),
        flush=True)

    return model


def evaluate(model, dataset, target='is_iceberg', **config):
    use_meta = config.get('use_meta', False)
    batch_size = config.get('batch_size', 16)
    name = config.get('name', 'unnamed')
    tmp = config.get('tmp')

    test_idx, test, test_meta = dataset

    print('\nPredict...', flush=True)

    if use_meta:
        pred = model.predict([test, test_meta], batch_size=batch_size, verbose=2)
    else:
        pred = model.predict(test, batch_size=batch_size, verbose=2)

    pred = np.squeeze(pred)

    file = 'subm_{}_{}.csv'.format(tmp, name)
    model_file = 'subm_{}_{}_model.txt'.format(tmp, name)
    with open('../submit/{}'.format(model_file), 'w') as f:
        model.summary(print_fn=lambda x: f.write(x + '\n'))

    subm = pd.DataFrame({'id': test_idx, target: pred})
    print('Saving submission file: {}'.format(file))
    subm.to_csv('../submit/{}'.format(file), index=False, float_format='%.6f')


if __name__=='__main__':
    print('Run from another script!')
