# Random initialization
import numpy as np
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

train_band = True
train_img = False
train_com = False


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


# Translate data to an image format
def color_composite(data):
    rgb_arrays = []
    for i, row in data.iterrows():
        band_1 = np.array(row['band_1']).reshape(75, 75)
        band_2 = np.array(row['band_2']).reshape(75, 75)
        band_3 = band_1 / band_2

        r = (band_1 + abs(band_1.min())) / np.max((band_1 + abs(band_1.min())))
        g = (band_2 + abs(band_2.min())) / np.max((band_2 + abs(band_2.min())))
        b = (band_3 + abs(band_3.min())) / np.max((band_3 + abs(band_3.min())))

        rgb = np.dstack((r, g, b))
        rgb_arrays.append(rgb)
    return np.array(rgb_arrays)


train = pd.read_json("../input/train.json")


def create_dataset(frame,
                   labeled,
                   smooth_rgb=0.2,
                   smooth_gray=0.5,
                   weight_rgb=0.05,
                   weight_gray=0.05):
    band_1, band_2, images = frame['band_1'].values, frame[
        'band_2'].values, color_composite(frame)

    to_arr = lambda x: np.asarray([np.asarray(item) for item in x])
    band_1 = to_arr(band_1)
    band_2 = to_arr(band_2)
    band_3 = (band_1 + band_2) / 2
    gray_reshape = lambda x: np.asarray([item.reshape(75, 75) for item in x])

    # Make a picture format from flat vector
    band_1 = gray_reshape(band_1)
    band_2 = gray_reshape(band_2)
    band_3 = gray_reshape(band_3)

    print('RGB done')
    tf_reshape = lambda x: np.asarray([item.reshape(75, 75, 1) for item in x])
    band_1 = tf_reshape(band_1)
    band_2 = tf_reshape(band_2)
    band_3 = tf_reshape(band_3)
    #images = tf_reshape(images)
    band = np.concatenate([band_1, band_2, band_3], axis=3)
    if labeled:
        y = np.array(frame["is_iceberg"])
    else:
        y = None
    return y, band, images


y_train, X_b, X_images = create_dataset(train, True)


def get_model_notebook(lr, decay, channels, relu_type='relu'):
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
    fcnn = Dropout(0.1)(fcnn)

    fcnn = Conv2D(128, kernel_size=(3, 3), activation=relu_type)(fcnn)
    fcnn = MaxPooling2D((2, 2), strides=(2, 2))(fcnn)
    fcnn = Dropout(0.2)(fcnn)

    fcnn = Conv2D(256, kernel_size=(3, 3), activation=relu_type)(fcnn)
    fcnn = MaxPooling2D((2, 2), strides=(2, 2))(fcnn)
    fcnn = Dropout(0.2)(fcnn)
    fcnn = BatchNormalization()(fcnn)

    fcnn = Flatten()(fcnn)

    local_input = input_1

    partial_model = Model(input_1, fcnn)

    dense = Dropout(0.2)(fcnn)
    dense = Dense(256, activation=relu_type)(dense)
    dense = Dropout(0.2)(dense)
    dense = Dense(128, activation=relu_type)(dense)
    dense = Dropout(0.2)(dense)
    dense = Dense(64, activation=relu_type)(dense)
    dense = Dropout(0.2)(dense)

    # For some reason i've decided not to normalize angle data
    output = Dense(1, activation="sigmoid")(dense)
    model = Model(local_input, output)
    optimizer = Adam(lr=lr, decay=decay)
    model.compile(
        loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"])
    return model, partial_model



def data_generator(data=None, labels=None, batch_size=16, data2=None):

    indices = [i for i in range(len(labels))]
    loop = 0
    while True:

        x_data = np.copy(data)
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
                x = Noise(x, u=0.4, v=np.random.random())
                x_batch.append(x)
            x_batch = np.array(x_batch, dtype=np.float32)

            yield x_batch, y_batch


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
    X_train, y_train, X_test, y_test, X_val, y_val = data
    model, partial_model = get_model_notebook(lr, decay, channels, relu)
    model = train_model(
        model,
        batch_size,
        epochs,
        path_name,
        X_train,
        y_train, (X_test, y_test),
        verbose=verbose)

    if verbose > 0:
        loss_val, acc_val = model.evaluate(
            X_val, y_val, verbose=0, batch_size=batch_size)

        loss_train, acc_train = model.evaluate(
            X_test, y_test, verbose=0, batch_size=batch_size)

        print('Val/Train Loss:', str(loss_val) + '/' + str(loss_train), \
            'Val/Train Acc:', str(acc_val) + '/' + str(acc_train))
    return model, partial_model


# Train all 3 models
    y_train, X_b, X_images = dataset
    y_train_full, y_val,\
    X_b_full, X_b_val,\
    X_images_full, X_images_val = train_test_split(y_train, X_b, X_images, random_state=687, train_size=0.9)

    y_train, y_test, \
    X_b_train, X_b_test, \
    X_images_train, X_images_test = train_test_split(y_train_full, X_b_full, X_images_full, random_state=576, train_size=0.85)

    print('Training bandwidth network')
    if train_band:
        data_b1 = (X_b_train, y_train, X_b_test, y_test, X_b_val, y_val)
        model_b, model_b_cut = gen_model_weights(
          lr,
          1e-6,
          3,
          'relu',
          batch_size,
          max_epoch,
          '../weights/bandwidth_model.hdf5',
          data=data_b1,
          verbose=verbose)

    #print('Training image network')
    if train_img:
        data_images = (X_images_train, y_train, X_images_test, y_test, X_images_val,
                   y_val)
        model_images, model_images_cut = gen_model_weights(
          lr,
          1e-6,
          3,
          'relu',
          batch_size,
          max_epoch,
          '../weights/img_model.hdf5',
          data_images,
          verbose=verbose)

    #common_model = combined_model(model_b_cut, model_images_cut, lr / 2, 1e-7)
    #common_x_train = [X_b_full, X_images_full]
    #common_y_train = y_train_full
    #common_x_val = [X_b_val, X_images_val]
    #common_y_val = y_val

    #print('Training common network')
    
    #earlystop = EarlyStopping(
    #    monitor='val_loss', patience=30, verbose=1, min_delta=1e-4)
    #reduce_lr_loss = ReduceLROnPlateau(
    #    monitor='val_loss', factor=0.1, patience=15, verbose=1, epsilon=1e-4)
    #model_chk = ModelCheckpoint(
    #    monitor='val_loss',
    #    filepath='../weights/combined_model.hdf5',
    #    save_best_only=True,
    #    save_weights_only=False)

    #callbacks = [earlystop, reduce_lr_loss, model_chk]
    
    if train_com:
        try:
            common_model.fit_generator(
                data_generator(
                    X_b_full,
                    data2=X_images_full,
                    labels=y_train_full,
                    batch_size=batch_size),
                epochs=max_epoch,
                steps_per_epoch=np.ceil(
                    8.0 * float(len(y_train)) / float(batch_size)),
                validation_data=(common_x_val, common_y_val),
                verbose=2,
                callbacks=callbacks)
        except KeyboardInterrupt:
            pass

    #common_model.load_weights(filepath='../weights/combined_model.hdf5',)
    #loss_val, acc_val = common_model.evaluate(
    #    common_x_val, common_y_val, verbose=0, batch_size=batch_size)
    #loss_train, acc_train = common_model.evaluate(
    #    common_x_train, common_y_train, verbose=0, batch_size=batch_size)
    #if verbose > 0:
    #    print('Loss:', loss_val, 'Acc:', acc_val)
    #return common_model
    return model_b


common_model = train_models((y_train, X_b, X_images), 8e-5, 32, 250, 2)

print('Reading test dataset')
test = pd.read_json("../input/test.json")
y_fin, X_fin_b, X_fin_img = create_dataset(test, False)
print('Predicting')
prediction = common_model.predict(
    X_fin_b, verbose=2, batch_size=32)
print('Submitting')
submission = pd.DataFrame({
    'id': test["id"],
    'is_iceberg': prediction.reshape((prediction.shape[0]))
})

submission.to_csv("../submit/common_model1.csv", index=False)
