from netCDF4 import Dataset as NetCDF


class NCFile:
    def __init__(self, path):
        self.path = path

    def variable(self, name):
        nc_file = NetCDF(self.path)
        var = nc_file.variables[name][:]
        nc_file.close()

        return var


file = NCFile(path='../ARCTIC_1h_ice_grid_TUV_20130619-20130619.nc')

print(file.variable('iceconc'))
