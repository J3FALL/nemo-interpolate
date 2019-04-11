from mpl_toolkits.basemap import Basemap


def filled_map(values, lon, lat):
    lat_left_bottom = lat[-1][-1]
    lon_left_bottom = lon[-1][-1]
    lat_right_top = lat[0][0]
    lon_right_top = lon[0][0]

    lat_center = 90
    # 110, 119
    lon_center = 110
    m = Basemap(projection='stere', lon_0=lon_center, lat_0=lat_center, resolution='l',
                llcrnrlat=lat_left_bottom, llcrnrlon=lon_left_bottom,
                urcrnrlat=lat_right_top, urcrnrlon=lon_right_top)

    m.pcolormesh(lon, lat, values, latlon=True, cmap='Blues_r', vmin=0, vmax=1)
    m.drawcoastlines()
    m.drawcountries()
    m.fillcontinents(color='#cc9966', lake_color='#99ffff')

    return m
