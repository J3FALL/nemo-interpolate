import random
from datetime import datetime

import numpy as np

random.seed(datetime.now())


class GapsGenerator:
    def __init__(self):
        pass

    def add_gaps_in_random_points(self, source_field, amount):
        x_len, y_len = source_field.shape

        points = []
        while len(points) < amount:
            point = (random.randint(0, x_len - 1), random.randint(0, y_len - 1))

            if point not in points:
                points.append(point)

        gap_mask = np.ones(shape=source_field.shape)

        for point in points:
            x, y = point
            gap_mask[x, y] = 0.0

        return np.multiply(source_field, gap_mask)

    def add_gaps_on_boundaries(self, source_field, side, width):
        x_range, y_range = indexes_of_boarder(source_field.shape, side, width)

        gap_mask = np.ones(shape=source_field.shape)
        gap_mask[x_range[0]:x_range[1], y_range[0]:y_range[1]] = 0.0

        return np.multiply(source_field, gap_mask)


def indexes_of_boarder(field_shape, side, width):
    x_size, y_size = field_shape
    if side is 'top':
        return (0, x_size), (0, width)
    if side is 'bottom':
        return (0, x_size), (y_size - width, -1)
    if side is 'left':
        return (0, width), (0, y_size)
    if side is 'right':
        return (x_size - width, -1), (0, y_size)
