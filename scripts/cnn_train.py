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
import json
#
from random import shuffle, uniform, seed
#evaluation
from sklearn.model_selection import train_test_split
#from sklearn.metrics import log_loss
#
from keras.optimizers import Adam, SGD
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, LambdaCallback
import tensorflow as tf
#
from augmentations import augment
import models

###############################################################################

def log_loss(t, p):
    h_tp = -(1-t)*np.log(1-p+1.0e-8)-t*np.log(p+1.0e-8)
    return np.mean(h_tp)


def data_generator(data, meta_data, labels, **config):

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


def predict(model, data, data_meta=None, **config):
    """ Predict results 
    data_meta is required only if use_meta = True
    """
    use_meta = config.get('use_meta', False)
    batch_size = config.get('batch_size', 16)
    if use_meta:
        pred = model.predict(
            [data, data_meta], batch_size=batch_size, verbose=2)
    else:
        pred = model.predict(data, batch_size=batch_size, verbose=2)

    return np.squeeze(pred)


def pseudo_generator(data, meta_data, labels, test, test_meta, graph, model,
                     **config):

    use_meta = config.get('use_meta', False)
    batch_size = config.get('batch_size', 16)
    retrain_freq = config.get('pseudo_retrain_freq', 8)
    pseudo_type = config.get('pseudo_type', 'soft') # hard/soft/clipped
    pseudo_proportion = config.get('pseudo_prop', .4)

    m_epoch = 0  # mini epochs are counted as for loop completions

    indices_real = [i for i in range(len(labels))]
    indices_pesudo = [i for i in range(len(test))]

    # shuffle indices in case test is somehow ordered
    np.random.shuffle(indices_real)
    np.random.shuffle(indices_pesudo)

    batch_real = int((1 - pseudo_proportion) * batch_size)
    batch_pseudo = batch_size - batch_real
    print(
        'Batch sizes: real_batch={}, pseudo_batch={}'.format(
            batch_real, batch_pseudo),
        flush=True)

    while True:

        x_data = np.copy(data)
        pseudo_data = np.copy(test)
        if use_meta:
            x_meta_data = np.copy(meta_data)
            pseudo_meta_data = np.copy(test_meta)
        x_labels = np.copy(labels)

        # retrain every retrain_freq mini-epochs
        if retrain_freq > 0 and m_epoch % retrain_freq == 0:
            with graph.as_default():
                pseudo = predict(model, test, test_meta, **config)

            if pseudo_type == 'hard':
                pseudo = np.round(pseudo)
            elif pseudo_type == 'clipped':
                pseudo_clip_val = config.get('pseudo_clip_val', .99)
                pseudo = np.clip(pseudo, 1 - pseudo_clip_val, pseudo_clip_val)
            # else - soft targets, nothing to do

        start_real = 0
        start_pseudo = 0

        while start_real < len(labels):
            end_real = min(start_real + batch_real, len(labels))
            end_pseudo = min(start_pseudo + batch_pseudo, len(test))
            real_indices = indices_real[start_real:end_real]
            pseudo_indices = indices_pesudo[start_pseudo:end_pseudo]

            #select data
            data_batch = np.r_[x_data[real_indices], pseudo_data[
                pseudo_indices]]
            if use_meta:
                xm_batch = np.r_[x_meta_data[real_indices], pseudo_meta_data[
                    pseudo_indices]]
            y_batch = np.r_[x_labels[real_indices], pseudo[pseudo_indices]]

            x_batch = []

            for x in data_batch:
                x = augment(x, **config)
                x_batch.append(x)

            x_batch = np.array(x_batch, dtype=np.float32)

            # update indices
            start_real += batch_real
            start_pseudo += batch_pseudo

            if start_pseudo >= len(test):
                start_pseudo = 0

            if use_meta:
                yield [x_batch, xm_batch], y_batch
            else:
                yield x_batch, y_batch

        # mini_epoch finished
        m_epoch += 1


###############################################################################
def train(dataset, model, **config):
    """ 
    dataset:  (y_train, X_train, X_meta) if pseudo=False, 
        and [(y_train, X_train, X_meta), (_, test, test_meta)] if pseudo=True
    """
    np.random.seed(1017)

    epochs = config.get('epochs', 250)
    batch_size = config.get('batch_size', 32)
    lr = config.get('lr', 1.0e-4)
    lr_patience = config.get('lr_patience', 15)
    stop_patience = config.get('stop_patience', 50)
    use_meta = config.get('use_meta', False)
    full_cycls_per_epoch = config.get('full_cycls_per_epoch', 8)
    pseudo = config.get('pseudo_train', False)
    out_name = config.get('output_name')
    model_w_name = config.get('model_w_name')

    if pseudo:
        ((labels, data, meta), (_, test, test_meta)) = dataset
        # per https://github.com/keras-team/keras/issues/2397#issuecomment-254919212
        graph = tf.get_default_graph()
    else:
        (labels, data, meta) = dataset

    #training
    print('epochs={}, batch={}'.format(epochs, batch_size), flush=True)

    #train, validataion split
    test_ratio = 0.15
    split_seed = 27

    X_train, X_test, Xm_train, Xm_test, y_train, y_test = train_test_split(
        data, meta, labels, test_size=test_ratio, random_state=split_seed)

    if use_meta:
        valid_data = ([X_test, Xm_test], y_test)
    else:
        valid_data = (X_test, y_test)

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
        epsilon=1e-4,
        min_lr=lr / 1000)
    model_chk = ModelCheckpoint(
        monitor='val_loss',
        filepath=model_w_name,
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

    if pseudo:
        model.fit_generator(
            generator=pseudo_generator(X_train, Xm_train, y_train, test,
                                       test_meta, graph, model, **config),
            steps_per_epoch=np.ceil(
                full_cycls_per_epoch * len(y_train) / batch_size),
            epochs=epochs,
            verbose=0,
            callbacks=callbacks,
            validation_data=valid_data)
    else:
        model.fit_generator(
            generator=data_generator(X_train, Xm_train, y_train, **config),
            steps_per_epoch=np.ceil(
                full_cycls_per_epoch * len(y_train) / batch_size),
            epochs=epochs,
            verbose=0,
            callbacks=callbacks,
            validation_data=valid_data)

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
        '\n\nLoss/Acc in validation data: {:.5f}/{:.5f}'.format(
            loss_val, acc_val),
        flush=True)
    print(
        'Loss/Acc in training data: {:.5f}/{:.5f}\n'.format(loss_tr, acc_tr),
        flush=True)

    train_preds = predict(model, data, meta, **config)
    train_log = {'preds': list(train_preds),
                 'labels': list(labels),
                 'meta': list(meta),
                 'val_loss': float(loss_val),
                 'val_acc': float(acc_val)
                 'tr_loss':float(loss_tr),
                 'tr_acc':float(acc_tr)}
    with open(out_name, 'w') as f:
        json.dump(train_log, f)
    print('Training output saved to: {}'.format(out_name))

    return model


def evaluate(model, dataset, target='is_iceberg', **config):
    use_meta = config.get('use_meta', False)
    batch_size = config.get('batch_size', 16)
    name = config.get('name', 'unnamed')
    tmp = config.get('tmp')

    test_idx, test, test_meta = dataset

    print('\nPredict...', flush=True)

    pred = predict(model, test, test_meta, **config)

    file = 'subm_{}_{}.csv'.format(tmp, name)
    model_file = 'subm_{}_{}_model.txt'.format(tmp, name)
    with open('../submit/{}'.format(model_file), 'w') as f:
        model.summary(print_fn=lambda x: f.write(x + '\n'))

    subm = pd.DataFrame({'id': test_idx, target: pred})
    print('Saving submission file: {}'.format(file))
    subm.to_csv('../submit/{}'.format(file), index=False, float_format='%.6f')


if __name__ == '__main__':
    print('Run from another script!')
