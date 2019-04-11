import matplotlib.pyplot as plt

from src.flies import NCFile
from src.gaps import GapsGenerator
from src.viz import filled_map

if __name__ == '__main__':
    file = NCFile(path='../ARCTIC_1h_ice_grid_TUV_20130619-20130619.nc')
    conc = file.variable(name='iceconc')

    lat = file.variable(name='nav_lat')
    lon = file.variable(name='nav_lon')

    generator = GapsGenerator()
    conc = generator.add_gaps_on_boundaries(source_field=conc[0], side='left', width=10)
    map = filled_map(values=conc, lon=lon, lat=lat)

    plt.show()
