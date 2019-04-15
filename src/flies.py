import os
import shutil

import numpy as np
from netCDF4 import Dataset as NetCDF


class NCFile:
    def __init__(self, path):
        self.path = path

    def variable(self, name):
        nc_file = NetCDF(self.path)
        var = nc_file.variables[name][:]
        nc_file.close()

        return self.handle_sat_field(var)

    def handle_sat_field(self, field):
        return field.filled(0) / 100.0


class NCHandler:
    def __init__(self, opened_files=[]):

        self.__opened_files = opened_files
        self.coastline_mask = self.__loaded_mask()

        if not os.path.exists('../handler_tmp'):
            os.mkdir('../handler_tmp')

    def subfield(self, sample, variable_name='ice_conc', size=100, time_dim=0):
        nc_file = NCFile(sample.path)
        full_field = self.values(nc_file=nc_file, variable_name=variable_name, time_dim=time_dim)

        x, y = sample.x, sample.y
        subfield = full_field[y:y + size, x:x + size]

        return subfield

    def values(self, nc_file, variable_name, time_dim=0):
        # TODO: add print as optional flag
        _, name = head_tail(nc_file.path)
        field_name = f'{name}_{variable_name}_{time_dim}.npy'

        if nc_file.path in self.__opened_files:
            field = np.load(os.path.join('../handler_tmp', field_name))

            # print(f'{field_name} was loaded')
            return field * self.coastline_mask
        else:
            field = nc_file.variable(name=variable_name)[time_dim]
            self.__dump_field(field, os.path.join('../handler_tmp', field_name))
            self.__opened_files.append(nc_file.path)

            # print(f'{field_name} was dumped')
            return field * self.coastline_mask

    def __dump_field(self, field, path):
        np.save(path, field)

    def __loaded_mask(self):
        # TODO: clear this somehow
        nc_file = NetCDF('../bathy_meter_mask.nc')
        mask = nc_file.variables['Bathymetry'][:]
        nc_file.close()
        return mask

    def clear(self):
        if os.path.exists('../handler_tmp'):
            shutil.rmtree('../handler_tmp')
        self.__opened_files = []


def head_tail(path):
    head, tail = os.path.split(path)

    return head, tail
