import calendar
import glob
import os

import matplotlib.pyplot as plt

from src.flies import NCFile
from src.viz import filled_map

MONTH_BY_IDX = dict((month, f'{idx:02d}') for idx, month in enumerate(calendar.month_abbr))


class LabelingParams:

    @staticmethod
    def default_params():
        square_size = 100
        full_field_size = {
            'x': 1100,
            'y': 400
        }
        borders = [
            {'x': [100, 800],
             'y': [0, 200]},
            {'x': [200, 800],
             'y': [200, 300]}
        ]

        return LabelingParams(square_size=square_size, full_field_size=full_field_size, borders=borders)

    def __init__(self, square_size, full_field_size, borders):
        self.square_size = square_size
        self.full_field_size = full_field_size
        self.borders = borders

        self.squares_amount = self.__squares_amount()

    def __squares_amount(self):
        amount = 0
        for border in self.borders:
            x_from, x_to = border['x']
            y_from, y_to = border['y']
            squares_by_x = (x_to - x_from) // self.square_size
            squares_by_y = (y_to - y_from) // self.square_size

            amount += squares_by_x * squares_by_y
        return amount

    def is_inside(self, x, y):
        for border in self.borders:
            x_border = border['x']
            y_border = border['y']

            if x_border[0] <= x < x_border[1] and y_border[0] <= y < y_border[1]:
                return True

        return False


class Sample:
    def __init__(self, path_to_file, idx, x, y):
        self.path = path_to_file
        self.idx = idx
        self.x = x
        self.y = y


def sat_dataset_with_labels(path, month, label_params=LabelingParams.default_params()):
    files = sat_images_from_dir(path, month)
    samples = []
    for file in files:
        square_idx = 0
        for y in range(0, label_params.full_field_size['y'], label_params.square_size):
            for x in range(0, label_params.full_field_size['x'], label_params.square_size):
                if label_params.is_inside(x, y):
                    samples.append(Sample(path_to_file=file, idx=square_idx, x=x, y=y))
                    square_idx += 1
    return samples


def sat_images_from_dir(path, month):
    year = '*'
    files = []
    for nc_file in glob.iglob(os.path.join(path, year, month, '*.nc'), recursive=True):
        files.append(nc_file)

    return files


if __name__ == '__main__':
    label_params = LabelingParams.default_params()
    print(f'Amount of squares: {label_params.squares_amount}')

    samples = sat_dataset_with_labels(path='D:\ice_recovered_from_hybrid\conc_satellite', month=MONTH_BY_IDX['May'],
                                      label_params=label_params)
    print(f'May samples total: {len(samples)}')

    print(f'nc_file path: {samples[10000].path}')
    file = NCFile(path=samples[10000].path)

    conc = file.variable(name='ice_conc')[0]

    lat = file.variable(name='nav_lat')
    lon = file.variable(name='nav_lon')

    map = filled_map(values=conc, lon=lon, lat=lat)

    plt.show()
