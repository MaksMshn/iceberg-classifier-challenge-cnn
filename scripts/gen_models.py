from keras.layers import Activation, Dense, Dropout, Flatten
from keras.layers.normalization import BatchNormalization
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Input, Flatten, GlobalAveragePooling2D, Lambda
from keras.layers import GlobalMaxPooling2D
from keras.layers.merge import Concatenate
from keras.models import Model
from keras.optimizers import Adam






def model1():
    inp = Input(shape=(75,75,2))

    x = Conv2D(256, (3,3), padding='same')(inp)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.0)(x)
    x = MaxPooling2D((2,2), strides=(2,2), padding='same')(x)

    x = Conv2D(16, (3,3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('sigmoid')(x)
    x = Dropout(0.1)(x)
    x = MaxPooling2D((2,2), strides=(2,2), padding='same')(x)

    x = Conv2D(256, (3,3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.25)(x)

    x = Conv2D(64, (3,3), padding='same')(x)
    x = Activation('sigmoid')(x)
    x = Dropout(0.35)(x)

    x = GlobalAveragePooling2D()(x)

    partial_model = Model(inp, x)

    x = Dense(128, activation='linear')(x)
    x = BatchNormalization()(x)
    x = Activation('sigmoid')(x)
    x = Dropout(0.1)(x)

    x = Dense(2, activation='softmax')(x)

    model = Model(inp, x)
    model.compile(loss='binary_crossentropy', optimizer=Adam(), metrics=['accuracy'])
    return model, partial_model

def model2():
    inp = Input(shape=(75,75,2))

    x = Conv2D(256, (3,3), padding='same')(inp)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.0)(x)
    x = MaxPooling2D((2,2), strides=(2,2), padding='same')(x)

    x = Conv2D(16, (3,3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('sigmoid')(x)
    x = Dropout(0.25)(x)
    x = MaxPooling2D((2,2), strides=(2,2), padding='same')(x)

    x = Conv2D(16, (3,3), padding='same')(x)
    x = Activation('relu')(x)
    x = Dropout(0.1)(x)

    x = Conv2D(32, (3,3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.35)(x)
    x = MaxPooling2D((2,2), strides=(2,2), padding='same')(x)

    x = Conv2D(64, (3,3), padding='same')(x)
    x = Activation('sigmoid')(x)
    x = Dropout(0.5)(x)

    x = GlobalAveragePooling2D()(x)

    partial_model = Model(inp, x)

    x = Dense(128, activation='linear')(x)
    x = BatchNormalization()(x)
    x = Activation('sigmoid')(x)
    x = Dropout(0.45)(x)

    x = Dense(2, activation='softmax')(x)

    model = Model(inp, x)
    model.compile(loss='binary_crossentropy', optimizer=Adam(), metrics=['accuracy'])
    return model, partial_model

def model3():
    inp = Input(shape=(75,75,2))

    x = Conv2D(256, (3,3), padding='same')(inp)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.0)(x)
    x = MaxPooling2D((2,2), strides=(2,2), padding='same')(x)

    x = Conv2D(16, (3,3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('sigmoid')(x)
    x = Dropout(0.1)(x)
    x = MaxPooling2D((2,2), strides=(2,2), padding='same')(x)

    x = Conv2D(256, (3,3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.25)(x)

    x = Conv2D(64, (3,3), padding='same')(x)
    x = Activation('relu')(x)
    x = Dropout(0.35)(x)

    x = GlobalAveragePooling2D()(x)

    partial_model = Model(inp, x)

    x = Dense(512, activation='linear')(x)
    x = BatchNormalization()(x)
    x = Activation('sigmoid')(x)
    x = Dropout(0.1)(x)

    x = Dense(2, activation='softmax')(x)

    model = Model(inp, x)
    model.compile(loss='binary_crossentropy', optimizer=Adam(), metrics=['accuracy'])
    return model, partial_model

def model4():
    inp = Input(shape=(75,75,2))

    x = Conv2D(256, (3,3), padding='same')(inp)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.0)(x)
    x = MaxPooling2D((2,2), strides=(2,2), padding='same')(x)

    x = Conv2D(16, (3,3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.1)(x)
    x = MaxPooling2D((2,2), strides=(2,2), padding='same')(x)

    x = Conv2D(256, (3,3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.25)(x)

    x = Conv2D(64, (3,3), padding='same')(x)
    x = Activation('sigmoid')(x)
    x = Dropout(0.35)(x)

    x = GlobalAveragePooling2D()(x)

    partial_model = Model(inp, x)

    x = Dense(128, activation='linear')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.1)(x)

    x = Dense(2, activation='softmax')(x)

    model = Model(inp, x)
    model.compile(loss='binary_crossentropy', optimizer=Adam(), metrics=['accuracy'])
    return model, partial_model

def model5():
    inp = Input(shape=(75,75,2))

    x = Conv2D(64, (3,3), padding='same')(inp)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.15)(x)

    x = Conv2D(32, (3,3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.2)(x)

    x = Conv2D(16, (3,3), padding='same')(x)
    x = Activation('sigmoid')(x)
    x = Dropout(0.05)(x)

    x = Conv2D(64, (3,3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('sigmoid')(x)
    x = Dropout(0.0)(x)
    x = MaxPooling2D((2,2), strides=(2,2), padding='same')(x)

    x = Conv2D(16, (3,3), padding='same')(x)
    x = Activation('sigmoid')(x)
    x = Dropout(0.1)(x)

    x = GlobalAveragePooling2D()(x)

    partial_model = Model(inp, x)

    x = Dense(2, activation='softmax')(x)

    model = Model(inp, x)
    model.compile(loss='binary_crossentropy', optimizer=Adam(), metrics=['accuracy'])
    return model, partial_model

def model6():
    inp = Input(shape=(75,75,2))

    x = Conv2D(256, (3,3), padding='same')(inp)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.15)(x)

    x = Conv2D(32, (3,3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.2)(x)

    x = Conv2D(64, (3,3), padding='same')(x)
    x = Activation('sigmoid')(x)
    x = Dropout(0.0)(x)
    x = MaxPooling2D((2,2), strides=(2,2), padding='same')(x)

    x = Conv2D(16, (3,3), padding='same')(x)
    x = Activation('sigmoid')(x)
    x = Dropout(0.1)(x)

    x = GlobalAveragePooling2D()(x)

    partial_model = Model(inp, x)

    x = Dense(2, activation='softmax')(x)

    model = Model(inp, x)
    model.compile(loss='binary_crossentropy', optimizer=Adam(), metrics=['accuracy'])
    return model, partial_model

def model7():
    inp = Input(shape=(75,75,2))

    x = Conv2D(16, (3,3), padding='same')(inp)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.1)(x)

    x = Conv2D(8, (3,3), padding='same')(x)
    x = Activation('sigmoid')(x)
    x = Dropout(0.1)(x)
    x = MaxPooling2D((2,2), strides=(2,2), padding='same')(x)

    x = Conv2D(32, (3,3), padding='same')(x)
    x = Activation('relu')(x)
    x = Dropout(0.4)(x)

    x = Conv2D(128, (3,3), padding='same')(x)
    x = Activation('relu')(x)
    x = Dropout(0.0)(x)

    x = Conv2D(16, (3,3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.15)(x)
    x = MaxPooling2D((2,2), strides=(2,2), padding='same')(x)

    x = Conv2D(16, (3,3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.15)(x)

    x = Conv2D(16, (3,3), padding='same')(x)
    x = Activation('sigmoid')(x)
    x = Dropout(0.1)(x)

    x = GlobalAveragePooling2D()(x)

    partial_model = Model(inp, x)

    x = Dense(2, activation='softmax')(x)

    model = Model(inp, x)
    model.compile(loss='binary_crossentropy', optimizer=Adam(), metrics=['accuracy'])
    return model, partial_model

def model8():
    inp = Input(shape=(75,75,2))

    x = Conv2D(16, (3,3), padding='same')(inp)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.1)(x)

    x = Conv2D(8, (3,3), padding='same')(x)
    x = Activation('sigmoid')(x)
    x = Dropout(0.1)(x)
    x = MaxPooling2D((2,2), strides=(2,2), padding='same')(x)

    x = Conv2D(32, (3,3), padding='same')(x)
    x = Activation('relu')(x)
    x = Dropout(0.4)(x)

    x = Conv2D(128, (3,3), padding='same')(x)
    x = Activation('relu')(x)
    x = Dropout(0.0)(x)

    x = Conv2D(16, (3,3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.15)(x)
    x = MaxPooling2D((2,2), strides=(2,2), padding='same')(x)

    x = Conv2D(16, (3,3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.15)(x)

    x = Conv2D(16, (3,3), padding='same')(x)
    x = Activation('sigmoid')(x)
    x = Dropout(0.1)(x)

    x = GlobalAveragePooling2D()(x)

    partial_model = Model(inp, x)

    x = Dense(2, activation='softmax')(x)

    model = Model(inp, x)
    model.compile(loss='binary_crossentropy', optimizer=Adam(), metrics=['accuracy'])
    return model, partial_model


def model9():
    inp = Input(shape=(75,75,2))

    x = Conv2D(16, (3,3), padding='same')(inp)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.1)(x)

    x = Conv2D(8, (3,3), padding='same')(x)
    x = Activation('sigmoid')(x)
    x = Dropout(0.0)(x)
    x = MaxPooling2D((2,2), strides=(2,2), padding='same')(x)

    x = Conv2D(32, (3,3), padding='same')(x)
    x = Activation('relu')(x)
    x = Dropout(0.4)(x)

    x = Conv2D(128, (3,3), padding='same')(x)
    x = Activation('relu')(x)
    x = Dropout(0.4)(x)

    x = Conv2D(16, (3,3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.15)(x)
    x = MaxPooling2D((2,2), strides=(2,2), padding='same')(x)

    x = Conv2D(64, (3,3), padding='same')(x)
    x = Activation('sigmoid')(x)
    x = Dropout(0.1)(x)

    x = GlobalAveragePooling2D()(x)

    partial_model = Model(inp, x)

    x = Dense(2, activation='softmax')(x)

    model = Model(inp, x)
    model.compile(loss='binary_crossentropy', optimizer=Adam(), metrics=['accuracy'])
    return model, partial_model

def model10():
    inp = Input(shape=(75,75,2))

    x = Conv2D(64, (3,3), padding='same')(inp)
    x = Activation('relu')(x)
    x = Dropout(0.1)(x)
    x = MaxPooling2D((2,2), strides=(2,2), padding='same')(x)

    x = Conv2D(8, (3,3), padding='same')(x)
    x = Activation('sigmoid')(x)
    x = Dropout(0.1)(x)

    x = Conv2D(128, (3,3), padding='same')(x)
    x = Activation('relu')(x)
    x = Dropout(0.5)(x)

    x = Conv2D(16, (3,3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.15)(x)
    x = MaxPooling2D((2,2), strides=(2,2), padding='same')(x)

    x = Conv2D(16, (3,3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('sigmoid')(x)
    x = Dropout(0.15)(x)

    x = GlobalAveragePooling2D()(x)

    partial_model = Model(inp, x)

    x = Dense(512, activation='linear')(x)
    x = BatchNormalization()(x)
    x = Activation('sigmoid')(x)
    x = Dropout(0.35)(x)

    x = Dense(2, activation='softmax')(x)

    model = Model(inp, x)
    model.compile(loss='binary_crossentropy', optimizer=Adam(), metrics=['accuracy'])
    return model, partial_model

def model11():
    inp = Input(shape=(75,75,2))

    x = Conv2D(16, (3,3), padding='same')(inp)
    x = Activation('relu')(x)
    x = Dropout(0.1)(x)
    x = MaxPooling2D((2,2), strides=(2,2), padding='same')(x)

    x = Conv2D(32, (3,3), padding='same')(x)
    x = Activation('sigmoid')(x)
    x = Dropout(0.4)(x)

    x = Conv2D(128, (3,3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.0)(x)

    x = Conv2D(16, (3,3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.15)(x)
    x = MaxPooling2D((2,2), strides=(2,2), padding='same')(x)

    x = Conv2D(128, (3,3), padding='same')(x)
    x = Activation('relu')(x)
    x = Dropout(0.15)(x)

    x = GlobalAveragePooling2D()(x)

    partial_model = Model(inp, x)

    x = Dense(2, activation='softmax')(x)

    model = Model(inp, x)
    model.compile(loss='binary_crossentropy', optimizer=Adam(), metrics=['accuracy'])
    return model, partial_model

def model12():
    inp = Input(shape=(75,75,2))

    x = Conv2D(256, (3,3), padding='same')(inp)
    x = Activation('relu')(x)
    x = Dropout(0.1)(x)
    x = MaxPooling2D((2,2), strides=(2,2), padding='same')(x)

    x = Conv2D(8, (3,3), padding='same')(x)
    x = Activation('relu')(x)
    x = Dropout(0.1)(x)

    x = Conv2D(32, (3,3), padding='same')(x)
    x = Activation('relu')(x)
    x = Dropout(0.1)(x)

    x = Conv2D(16, (3,3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.05)(x)
    x = MaxPooling2D((2,2), strides=(2,2), padding='same')(x)

    x = Conv2D(128, (3,3), padding='same')(x)
    x = Activation('relu')(x)
    x = Dropout(0.1)(x)
    x = MaxPooling2D((2,2), strides=(2,2), padding='same')(x)

    x = GlobalAveragePooling2D()(x)

    partial_model = Model(inp, x)

    x = Dense(2, activation='softmax')(x)

    model = Model(inp, x)
    model.compile(loss='binary_crossentropy', optimizer=Adam(), metrics=['accuracy'])
    return model, partial_model

def model13():
    inp = Input(shape=(75,75,2))

    x = Conv2D(16, (3,3), padding='same')(inp)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.15)(x)
    x = MaxPooling2D((2,2), strides=(2,2), padding='same')(x)

    x = Conv2D(128, (3,3), padding='same')(x)
    x = Activation('relu')(x)
    x = Dropout(0.45)(x)

    x = Conv2D(64, (3,3), padding='same')(x)
    x = Activation('sigmoid')(x)
    x = Dropout(0.4)(x)

    x = Conv2D(32, (3,3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('sigmoid')(x)
    x = Dropout(0.3)(x)

    x = GlobalAveragePooling2D()(x)

    partial_model = Model(inp, x)

    x = Dense(16, activation='linear')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.05)(x)

    x = Dense(2, activation='softmax')(x)

    model = Model(inp, x)
    model.compile(loss='binary_crossentropy', optimizer=Adam(), metrics=['accuracy'])
    return model, partial_model

def model14():
    inp = Input(shape=(75,75,2))

    x = Conv2D(16, (3,3), padding='same')(inp)
    x = Activation('relu')(x)
    x = Dropout(0.15)(x)

    x = Conv2D(128, (3,3), padding='same')(x)
    x = Activation('relu')(x)
    x = Dropout(0.1)(x)

    x = Conv2D(64, (3,3), padding='same')(x)
    x = Activation('sigmoid')(x)
    x = Dropout(0.4)(x)

    x = Conv2D(64, (3,3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('sigmoid')(x)
    x = Dropout(0.0)(x)
    x = MaxPooling2D((2,2), strides=(2,2), padding='same')(x)

    x = Conv2D(64, (3,3), padding='same')(x)
    x = Activation('sigmoid')(x)
    x = Dropout(0.45)(x)

    x = GlobalAveragePooling2D()(x)

    partial_model = Model(inp, x)

    x = Dense(16, activation='linear')(x)
    x = BatchNormalization()(x)
    x = Activation('sigmoid')(x)
    x = Dropout(0.05)(x)

    x = Dense(2, activation='softmax')(x)

    model = Model(inp, x)
    model.compile(loss='binary_crossentropy', optimizer=Adam(), metrics=['accuracy'])
    return model, partial_model
