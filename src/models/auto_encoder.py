from keras import backend as K
from keras.layers import Activation, Dense, Input
from keras.layers import Conv2D, Flatten
from keras.layers import Reshape, Conv2DTranspose
from keras.models import Model


class AutoEncoder:
    def __init__(self, input_shape, kernel_size, latent_dim, layer_filters):
        self.input_shape = input_shape
        self.kernel_size = kernel_size
        self.latent_dim = latent_dim
        self.layer_filters = layer_filters

        self.auto_encoder = None
        self.encoder = None
        self.decoder = None

    def init(self, print_summary=True):
        inputs = Input(shape=self.input_shape, name='encoder_input')
        layer = inputs

        layer = self.__stack_conv_layers(layer)
        shape = K.int_shape(layer)
        layer = Flatten()(layer)
        latent = Dense(self.latent_dim, name='latent_vector')(layer)

        # Instantiate Encoder Model
        encoder = Model(inputs, latent, name='encoder')

        if print_summary:
            encoder.summary()

        # Build the Decoder Model
        latent_inputs = Input(shape=(self.latent_dim,), name='decoder_input')
        layer = Dense(shape[1] * shape[2] * shape[3])(latent_inputs)
        layer = Reshape((shape[1], shape[2], shape[3]))(layer)

        layer = self.__stack_transposed_conv_layers(layer)
        layer = Conv2DTranspose(filters=1,
                                kernel_size=self.kernel_size,
                                padding='same')(layer)

        outputs = Activation('sigmoid', name='decoder_output')(layer)

        # Instantiate Decoder Model
        decoder = Model(latent_inputs, outputs, name='decoder')

        if print_summary:
            decoder.summary()

        # Autoencoder = Encoder + Decoder
        # Instantiate Autoencoder Model
        auto_encoder = Model(inputs, decoder(encoder(inputs)), name='auto_encoder')
        if print_summary:
            auto_encoder.summary()

        self.auto_encoder = auto_encoder
        self.encoder = encoder
        self.decoder = decoder

    def compile(self, loss='mse', optimizer='adam'):
        self.auto_encoder.compile(loss=loss, optimizer=optimizer)

    def fit_generator(self, **kwargs):
        self.auto_encoder.fit_generator(generator=kwargs['generator'],
                                        validation_data=kwargs['validation_data'])

    def __stack_conv_layers(self, layer):
        for filter in self.layer_filters:
            layer = Conv2D(filters=filter,
                           kernel_size=self.kernel_size,
                           strides=2,
                           activation='relu',
                           padding='same')(layer)
        return layer

    def __stack_transposed_conv_layers(self, layer):
        for filters in self.layer_filters[::-1]:
            layer = Conv2DTranspose(filters=filters,
                                    kernel_size=self.kernel_size,
                                    strides=2,
                                    activation='relu',
                                    padding='same')(layer)
        return layer
