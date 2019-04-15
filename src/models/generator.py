import keras
import numpy as np


class NCGenerator(keras.utils.Sequence):
    def __init__(self, samples, handler, batch_size=32, dimension=(100, 100, 1), n_channels=1):
        self.samples = samples
        self.handler = handler
        self.batch_size = batch_size
        self.dimension = dimension
        self.n_channels = n_channels

    def __len__(self):
        return int(np.floor(len(self.samples) / self.batch_size))

    def __getitem__(self, index):
        batch = self.samples[index * self.batch_size:(index + 1) * self.batch_size]
        # batch = [self.samples[idx] for idx in indexes]

        X, y = self.__data_generation(batch)

        return X, y

    def __data_generation(self, batch):
        X = np.empty((self.batch_size, *self.dimension))
        y = np.empty((self.batch_size, *self.dimension))

        for idx, sample in enumerate(batch):
            field = self.handler.subfield(sample)
            X[idx] = np.stack(arrays=[field], axis=2)
            y[idx] = np.copy(X[idx])

        return X, y
