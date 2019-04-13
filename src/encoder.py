import os

import numpy as np
import tensorflow as tf
from keras.datasets import mnist
from keras.layers import (
    Input,
    Conv2D,
    MaxPooling2D,
    UpSampling2D
)
from keras.models import (
    Model,
    load_model)


def initialized_model():
    # this is the size of our encoded representations
    input_img = Input(shape=(28, 28, 1))  # adapt this if using `channels_first` image data format

    x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    encoded = MaxPooling2D((2, 2), padding='same')(x)

    # at this point the representation is (4, 4, 8) i.e. 128-dimensional

    x = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(16, (3, 3), activation='relu')(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

    auto_encoder = Model(input_img, decoded)
    auto_encoder.compile(optimizer='adadelta', loss='binary_crossentropy')

    return auto_encoder


def mnist_dataset():
    (x_train, _), (x_test, _) = mnist.load_data()

    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.
    x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))
    x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))

    return x_train, x_test


def loaded_models(path='../saved_models/', prefix='test_'):
    models = []
    for label in ['auto_encoder']:
        model_path = os.path.join(path, ''.join([prefix, label, '.h5']))
        models.append(load_model(model_path))
        print(f'model was loaded from {model_path}')

    return models


def dump_models(models, path='../saved_models/', prefix='test_'):
    for model, label in zip(models, ['auto_encoder']):
        model_path = os.path.join(path, ''.join([prefix, label, '.h5']))
        model.save(model_path)
        print(f'model was dumped to {model_path}')


def trained_auto_encoder(x_train, x_test):
    auto_encoder = initialized_model()
    auto_encoder.fit(x_train, x_train,
                     epochs=50,
                     batch_size=128,
                     shuffle=True,
                     validation_data=(x_test, x_test))

    return auto_encoder


if __name__ == '__main__':
    sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

    x_train, x_test = mnist_dataset()

    if not os.path.isfile('../saved_models/test_auto_encoder.h5'):
        auto_encoder = trained_auto_encoder(x_train, x_test)
        dump_models(models=[auto_encoder])
    else:
        auto_encoder = loaded_models()[0]
