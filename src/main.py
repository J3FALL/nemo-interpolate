from sklearn.model_selection import train_test_split

from src.dataset import (
    LabelingParams,
    sat_dataset_with_labels
)
from src.flies import (
    NCHandler,
    NCFile
)
from src.models.auto_encoder import AutoEncoder


def train_model():
    label_params = LabelingParams.default_params()

    samples = sat_dataset_with_labels(path='D:\ice_recovered_from_hybrid\conc_satellite', month='May',
                                      label_params=label_params)

    test = samples[:50]
    handler = NCHandler()

    for sample in test:
        nc_file = NCFile(sample.path)
        _ = handler.values(nc_file=nc_file, variable_name='ice_conc')
    handler.clear()

    train_set, test_set = train_test_split(samples, test_size=0.2)

    input_shape = (label_params.square_size, label_params.square_size, 1)
    model = AutoEncoder(input_shape=input_shape, kernel_size=3, latent_dim=32, layer_filters=[32, 64])
    model.init(print_summary=True)
    model.compile()


if __name__ == '__main__':
    train_model()
