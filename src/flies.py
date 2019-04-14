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
        return field.filled(0)


class NCHandler:
    def __init__(self):
        self.__opened_files = []

    def values(self, nc_file, variable_name, time_dim=0):
        if not os.path.exists('../handler_tmp'):
            os.mkdir('../handler_tmp')

        _, name = head_tail(nc_file.path)
        field_name = f'{name}_{variable_name}_{time_dim}.npy'

        if nc_file.path in self.__opened_files:
            field = np.load(os.path.join('../handler_tmp', field_name))

            print(f'{field_name} was loaded')
            return field
        else:
            field = nc_file.variable(name=variable_name)[time_dim]
            self.__dump_field(field, os.path.join('../handler_tmp', field_name))
            self.__opened_files.append(nc_file.path)

            print(f'{field_name} was dumped')

    def __dump_field(self, field, path):
        np.save(path, field)

    def clear(self):
        if os.path.exists('../handler_tmp'):
            shutil.rmtree('../handler_tmp')
        self.__opened_files = []


def head_tail(path):
    head, tail = os.path.split(path)

    return head, tail
