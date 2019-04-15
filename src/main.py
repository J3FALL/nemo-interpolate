from sklearn.model_selection import train_test_split
from tqdm import tqdm

from src.dataset import (
    LabelingParams,
    sat_dataset_with_labels,
    sat_images_from_dir
)
from src.flies import (
    NCHandler,
    NCFile
)
from src.models.auto_encoder import AutoEncoder
from src.models.generator import NCGenerator


def train_model():
    label_params = LabelingParams.default_params()

    samples = sat_dataset_with_labels(path='D:\ice_recovered_from_hybrid\conc_satellite', month='May',
                                      label_params=label_params)
    handler = NCHandler(opened_files=sat_images_from_dir(path='D:\ice_recovered_from_hybrid\conc_satellite',
                                                         month='05'))
    # prepare_handler(handler, samples)

    train_set, test_set = train_test_split(samples, test_size=0.2)

    input_shape = (label_params.square_size, label_params.square_size, 1)
    model = AutoEncoder(input_shape=input_shape, kernel_size=3, latent_dim=32, layer_filters=[32, 64])
    model.init(print_summary=True)
    model.compile()

    params = {'batch_size': 64,
              'dimension': input_shape,
              'n_channels': 1}

    training_generator = NCGenerator(train_set, handler, **params)
    validation_generator = NCGenerator(test_set, handler, **params)

    model.fit_generator(generator=training_generator,
                        validation_data=validation_generator)


def prepare_handler(handler, samples):
    for sample in tqdm(samples):
        handler.values(nc_file=NCFile(sample.path), variable_name='ice_conc', time_dim=0)


if __name__ == '__main__':
    train_model()
