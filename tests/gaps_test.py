import numpy as np

from src.gaps import GapsGenerator


def test_gaps_on_bounds_correct():
    generator = GapsGenerator()

    source = np.ones(shape=(10, 10))
    with_gaps = generator.add_gaps_on_boundaries(source_field=source, side='top', width=2)

    expected_gap = np.zeros(shape=(10, 2))
    actual = with_gaps[:, :2]

    assert np.array_equal(expected_gap, actual)
