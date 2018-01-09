# Random initialization
import numpy as np
import sys
import tensorflow as tf
# Uncomment this to hide TF warnings about allocation
#import os
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


# Data reading and visualization
import pandas as pd

from sklearn.preprocessing import MinMaxScaler

# Training part
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Input, Flatten, GlobalAveragePooling2D, Lambda
from keras.layers import GlobalMaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.merge import Concatenate
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from keras.preprocessing.image import ImageDataGenerator

# Any results you write to the current directory are saved as output.

from random import choice
import cv2
import keras.preprocessing.image as prep



#data augmentations
###############################################################################
def HorizontalFlip(image, u=0.5, v=1.0):

    if v < u:
        image = cv2.flip(image, 1)

    return image


def VerticalFlip(image, u=0.5, v=1.0):

    if v < u:
        image = cv2.flip(image, 0)

    return image


def Rotate90(image, u=0.5, v=1.0):

    if v < u:
        image = np.rot90(image, k=choice([0, 1, 2, 3]), axes=(0, 1))

    return image


def Rotate(image, rotate_rg=10, u=0.5, v=1.0):

    if v < u:
        image = prep.random_rotation(
            image, rg=rotate_rg, row_axis=0, col_axis=1, channel_axis=2)

    return image


def Shift(image, width_rg=0.1, height_rg=0.1, u=0.5, v=1.0):

    if v < u:
        image = prep.random_shift(
            image,
            wrg=width_rg,
            hrg=height_rg,
            row_axis=0,
            col_axis=1,
            channel_axis=2)

    return image


def Zoom(image, zoom_rg=(0.1, 0.1), u=0.5, v=1.0):

    if v < u:
        image = prep.random_zoom(
            image, zoom_range=zoom_rg, row_axis=0, col_axis=1, channel_axis=2)

    return image


def Noise(image, noise_rg=0.02, u=0.5, v=1.0):
    if v < u:
        noise_amp = (image.max() - image.min()) * np.random.rand() * noise_rg
        image = image + noise_amp * np.random.normal(size=image.shape)
    return image



train = pd.read_json("../input/train.json")


def create_dataset(frame, labeled):
    band_1, band_2 = frame['band_1'].values, frame['band_2'].values
    frame['inc_angle'] = frame['inc_angle'].replace('na', -1).astype(float)

    to_arr = lambda x: np.asarray([np.asarray(item) for item in x])
    band_1 = to_arr(band_1)
    band_2 = to_arr(band_2)

    band_3 = (band_1 + band_2) / 2
    
    gray_reshape = lambda x: np.asarray([item.reshape(75, 75) for item in x])
    band_1 = gray_reshape(band_1)
    band_2 = gray_reshape(band_2)
    band_3 = gray_reshape(band_3)

    band = np.stack([band_1, band_2, band_3], axis=3)

    if labeled:
        y = frame["is_iceberg"].values
    else:
        y = None
    return y, band, frame['inc_angle'].values



def get_model_notebook(lr, channels, relu_type='relu', decay=1.e-6):
    # angle variable defines if we should use angle parameter or ignore it
    input_1 = Input(shape=(75, 75, channels))

    fcnn = Conv2D(
        32, kernel_size=(3, 3),
        activation=relu_type)(BatchNormalization()(input_1))
    fcnn = MaxPooling2D((3, 3))(fcnn)
    fcnn = BatchNormalization()(fcnn)

    fcnn = Conv2D(64, kernel_size=(3, 3), activation=relu_type)(fcnn)
    fcnn = MaxPooling2D((2, 2), strides=(2, 2))(fcnn)
    fcnn = BatchNormalization()(fcnn)
    fcnn_1 = Dropout(0.1)(fcnn)

    #Path 1
    fcnn = Conv2D(128, kernel_size=(3, 3), activation=relu_type)(fcnn_1)
    fcnn = MaxPooling2D((2, 2), strides=(2, 2))(fcnn)
    fcnn = Dropout(0.2)(fcnn)

    fcnn = Conv2D(128, kernel_size=(3, 3), activation=relu_type)(fcnn)
    fcnn = MaxPooling2D((2, 2), strides=(2, 2))(fcnn)
    fcnn = Dropout(0.2)(fcnn)
    fcnn = BatchNormalization()(fcnn)

    fcnn = Flatten()(fcnn)

    #Path 2
    fcnn_2 = Conv2D(64, kernel_size=(3, 3), activation=relu_type)(fcnn_1)
    fcnn_2 = BatchNormalization()(fcnn_2)

    fcnn_2 = Conv2D(128, kernel_size=(3, 3), activation=relu_type)(fcnn_2)
    fcnn_2 = BatchNormalization()(fcnn_2)

    fcnn_2 = Conv2D(256, kernel_size=(3, 3), activation=relu_type, padding='same')(fcnn_2)
    fcnn_2 = BatchNormalization()(fcnn_2)

    fcnn_2 = Conv2D(512, kernel_size=(3, 3), activation=relu_type, padding='same')(fcnn_2)
    fcnn_2 = Dropout(0.2)(fcnn_2)
    fcnn_2 = BatchNormalization()(fcnn_2)

    fcnn_2 = Conv2D(256, kernel_size=(3, 3), activation=relu_type, padding='same')(fcnn_2)
    fcnn_2 = MaxPooling2D((2,2), strides=(2,2))(fcnn_2)
    fcnn_2 = Dropout(0.2)(fcnn_2)
    fcnn_2 = BatchNormalization()(fcnn_2)

    fcnn_2 = Conv2D(64, kernel_size=(3, 3), activation=relu_type, padding='same')(fcnn_2)
    fcnn_2 = MaxPooling2D((2,2), strides=(2,2))(fcnn_2)
    fcnn_2 = Dropout(0.2)(fcnn_2)
    fcnn_2 = BatchNormalization()(fcnn_2)

    fcnn_2 = Flatten()(fcnn_2)
    
    
    input_2 = Input(shape=[1],name='angle')
    input_2_bn = BatchNormalization()(input_2)

    fcnn = Concatenate()([fcnn, fcnn_2, input_2_bn])

    dense = Dense(256, activation=relu_type)(fcnn)
    dense = Dropout(0.2)(dense)
    dense = Dense(128, activation=relu_type)(dense)
    dense = Dropout(0.2)(dense)
    dense = Dense(64, activation=relu_type)(dense)
    dense = Dropout(0.2)(dense)

    # For some reason i've decided not to normalize angle data
    output = Dense(1, activation="sigmoid")(dense)
    model = Model([input_1, input_2], output)
    optimizer = Adam(lr=lr, decay=decay)
    model.compile(
        loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"])
    return model, None

def data_generator(data=None, labels=None, batch_size=16):

    indices = [i for i in range(len(labels))]
    loop = 0
    while True:

        x_data = np.copy(data[0])
        x_meta = np.copy(data[1])
        x_labels = np.copy(labels)

        for start in range(0, len(labels), batch_size):
            end = min(start + batch_size, len(labels))
            sel_indices = indices[start:end]

            #select data
            data_batch = x_data[sel_indices]
            y_batch = x_labels[sel_indices]
            x_batch = []

            for x in data_batch:
                #augment
                x = Rotate(x, u=0.2, v=np.random.random())
                x = Rotate90(x, u=0.999, v=np.random.random())
                x = Shift(x, u=0.1, v=np.random.random())
                x = Zoom(x, u=0.15, v=np.random.random())
                x = HorizontalFlip(x, u=0.5, v=np.random.random())
                x = VerticalFlip(x, u=0.5, v=np.random.random())
                x = Noise(x, u=0.5, v=np.random.random())
                x_batch.append(x)
            x_batch = np.array(x_batch, dtype=np.float32)

            yield [x_batch, x_meta[sel_indices]], y_batch



def train_model(model,
                batch_size,
                epochs,
                checkpoint_name,
                X_train,
                y_train,
                val_data,
                verbose=2):

    earlystop = EarlyStopping(
        monitor='val_loss', patience=60, verbose=1, min_delta=1e-4)
    reduce_lr_loss = ReduceLROnPlateau(
        monitor='val_loss', factor=0.1, patience=15, verbose=1, epsilon=1e-4)
    model_chk = ModelCheckpoint(
        monitor='val_loss',
        filepath=checkpoint_name,
        save_best_only=True,
        save_weights_only=False)

    callbacks = [earlystop, reduce_lr_loss, model_chk]

    x_test, y_test = val_data
    try:
        model.fit_generator(
            data_generator(X_train, y_train, batch_size=batch_size),
            epochs=epochs,
            steps_per_epoch=np.ceil(
                8.0 * float(len(y_train)) / float(batch_size)),
            validation_data=(x_test, y_test),
            verbose=2,
            callbacks=callbacks)
    except KeyboardInterrupt:
        if verbose > 0:
            print('Interrupted')
    if verbose > 0:
        print('Loading model')
    model.load_weights(filepath=checkpoint_name)
    return model


def gen_model_weights(lr,
                      channels,
                      relu,
                      batch_size,
                      epochs,
                      path_name,
                      data,
                      verbose=2):
    X_train, y_train, X_val, y_val = data
    model, partial_model = get_model_notebook(lr, channels, relu)
    model.summary()
    print('training', flush=True)
    model = train_model(
        model,
        batch_size,
        epochs,
        path_name,
        X_train,
        y_train, (X_val, y_val),
        verbose=verbose)

    if verbose > 0:
        loss_val, acc_val = model.evaluate(
            X_val, y_val, verbose=0, batch_size=batch_size)

        loss_train, acc_train = model.evaluate(
            X_train, y_train, verbose=0, batch_size=batch_size)

        print('Val/Train Loss:', str(loss_val) + '/' + str(loss_train), \
            'Val/Train Acc:', str(acc_val) + '/' + str(acc_train))
    return model, partial_model


def main(dataset, batch_size, max_epoch, tmp):
    weights_name = '../weights/bandwidth_model_{}.hdf5'.format(tmp)

    y_train, X_b, X_meta = dataset
    y_train, y_val,\
    X_train, X_val, X_meta_train, X_meta_val  = train_test_split(y_train, X_b, X_meta, train_size=0.9)

    print('Training bandwidth network')
    data_b1 = ([X_train, X_meta_train], y_train, [X_val, X_meta_val], y_val)
    model_b, model_b_cut = gen_model_weights(lr=1e-6, channels=3, relu='relu',
          batch_size=batch_size,
          epochs=max_epoch,
          path_name=weights_name,
          data=data_b1)
    return model_b


if __name__ == '__main__':
    import datetime as dt
    tmp = dt.datetime.now().strftime("%Y-%m-%d-%H-%M")
    batch_size = 32
    epochs = 250
    y_train, X_b, X_meta = create_dataset(train, True)
    common_model = main((y_train, X_b, X_meta), batch_size, epochs, tmp + '_' + sys.argv[-1])

    print('Reading test dataset')
    test = pd.read_json("../input/test.json")
    y_fin, X_fin_b, X_meta = create_dataset(test, False)
    prediction = common_model.predict([X_fin_b, X_meta], verbose=2, batch_size=32)
    sub_name = '../submit/keras_018_{}.csv'.format(tmp + '_' + sys.argv[-1])
    print('Submitting ' + sub_name)
    submission = pd.DataFrame({
        'id': test["id"],
        'is_iceberg': prediction.reshape((prediction.shape[0]))
    })

    submission.to_csv(sub_name, index=False)


