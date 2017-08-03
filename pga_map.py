#!/usr/bin/env python
# -*- coding: utf8 -*-
"""
Parses event XML files (ShakeMap) and plots PGAs in mg on a map.

(c) 2017 - Claudio Satriano <satriano@ipgp.fr>
           Felix Léger <leger@ipgp.fr>
"""
from __future__ import print_function
import os
import argparse
try:
    # Python2
    from ConfigParser import ConfigParser
except ModuleNotFoundError:
    # Python3
    from configparser import ConfigParser
try:
    # Python2
    from StringIO import StringIO
except ModuleNotFoundError:
    # Python3
    from io import StringIO
from xml.dom import minidom
from datetime import datetime
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import matplotlib.patheffects as path_effects
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
import cartopy.io.img_tiles as cimgt
import cartopy.crs as ccrs


def parse_event_xml(xml_file):
    """
    Parse an event.xml file (input for ShakeMap).

    Returns an event dictionary.
    """
    event = dict()
    xmldoc = minidom.parse(xml_file)
    tag_earthquake = xmldoc.getElementsByTagName('earthquake')[0]
    event['id'] = tag_earthquake.attributes['id'].value
    event['time'] = datetime.strptime(event['id'], '%Y%m%d%H%M%S')
    locstring = tag_earthquake.attributes['locstring'].value
    event['id_sc3'] = locstring.split(' / ')[0]
    lat = tag_earthquake.attributes['lat'].value
    event['lat'] = float(lat)
    lon = tag_earthquake.attributes['lon'].value
    event['lon'] = float(lon)
    depth = tag_earthquake.attributes['depth'].value
    event['depth'] = float(depth)
    mag = tag_earthquake.attributes['mag'].value
    event['mag'] = float(mag)
    return event


def parse_event_dat_xml(xml_file):
    """
    Parse an event_dat.xml file (input for ShakeMap).

    Returns a dictionary of channel attributes:
        lon, lat, pga, pgv, psa03, psa10, psa30
    """
    xmldoc = minidom.parse(xml_file)
    tag_stationlist = xmldoc.getElementsByTagName('stationlist')
    attributes = dict()
    for slist in tag_stationlist:
        tag_station = slist.getElementsByTagName('station')
        for sta in tag_station:
            stname = sta.attributes['name'].value
            net = sta.attributes['netid'].value
            stla = float(sta.attributes['lat'].value)
            stlo = float(sta.attributes['lon'].value)
            tag_comp = sta.getElementsByTagName('comp')
            for comp in tag_comp:
                cmp_attributes = {'latitude': stla, 'longitude': stlo}
                cmp_name = comp.attributes['name'].value
                cmp_id = '.'.join((net, stname, cmp_name))
                # pga is in percent-g, transform it to milli-g
                tag_acc = comp.getElementsByTagName('acc')[0]
                pga = tag_acc.attributes['value'].value
                cmp_attributes['pga'] = float(pga)*10.
                # pgv is cm/s, transform it to m/s
                tag_vel = comp.getElementsByTagName('vel')[0]
                pgv = tag_vel.attributes['value'].value
                cmp_attributes['pgv'] = float(pgv)/100.
                # psa is in percent-g, transform it to milli-g
                tag_psa03 = comp.getElementsByTagName('psa03')[0]
                tag_psa10 = comp.getElementsByTagName('psa10')[0]
                tag_psa30 = comp.getElementsByTagName('psa30')[0]
                psa03 = tag_psa03.attributes['value'].value
                psa10 = tag_psa10.attributes['value'].value
                psa30 = tag_psa30.attributes['value'].value
                cmp_attributes['psa03'] = float(psa03)*10.
                cmp_attributes['psa10'] = float(psa10)*10.
                cmp_attributes['psa30'] = float(psa30)*10.
                attributes[cmp_id] = cmp_attributes
    return attributes


def colormap():
    # Normalizing color scale
    norm = mpl.colors.Normalize(vmin=0, vmax=5)
    # YlOrRd colormap
    cmap = plt.cm.YlOrRd
    return norm, cmap


def plot_station_name(lon, lat, stname, ax):
    geodetic_transform = ccrs.Geodetic()._as_mpl_transform(ax)

    def text_transform(x):
        return transforms.offset_copy(geodetic_transform,
                                      units='dots', x=x)

    if stname in ['MPLM', 'LAM']:
        trans = text_transform(50)
        ha = 'left'
        if stname == 'LAM':
            va = 'bottom'
        else:
            va = 'center'
    else:
        trans = text_transform(-50)
        ha = 'right'
        va = 'center'
    t = plt.text(lon, lat, stname, size=8, weight='bold',
                 verticalalignment=va, horizontalalignment=ha,
                 transform=trans, zorder=99)
    t.set_path_effects([
        path_effects.Stroke(linewidth=1.5, foreground='white'),
        path_effects.Normal()
    ])


def plot_figure_text(event, pga_text, conf):
    figtitle = 'Peak Ground Acceleration '
    date = event['time'].strftime('%Y-%m-%d %H:%M:%S')
    figtitle += date
    title_y = float(conf['title_offset'])
    plt.figtext(.4, title_y, figtitle,
                horizontalalignment='center',
                verticalalignment='top',
                size=12, weight='bold')

    subtitle_y = title_y - .03
    plt.figtext(.1, subtitle_y, 'Event Information', size=9,
                horizontalalignment='left',
                verticalalignment='top',
                weight='bold')
    event_text = ' Date: {}\n'.format(date)
    event_text += '  Lat: {:8.4f}\n'.format(event['lat'])
    event_text += '  Lon: {:8.4f}\n'.format(event['lon'])
    event_text += 'Depth: {:.3f} km\n'.format(event['depth'])
    event_text += '  Mag: {:.2f}\n'.format(event['mag'])
    text_y = subtitle_y - .02
    plt.figtext(.1, text_y, event_text, size=8,
                horizontalalignment='left',
                verticalalignment='top',
                family='monospace')

    pga_text_x = .35
    plt.figtext(pga_text_x, subtitle_y, 'PGA (mg)', size=9,
                horizontalalignment='left',
                verticalalignment='top',
                weight='bold')
    for n, text in enumerate(pga_text):
        text_x = pga_text_x + n * 0.1
        plt.figtext(text_x, text_y, text, size=8,
                    horizontalalignment='left',
                    verticalalignment='top',
                    family='monospace')


def plotmap(attributes, event, basename, conf):
    # Create a Stamen Terrain instance.
    stamen_terrain = cimgt.StamenTerrain()

    # Create a GeoAxes in the tile's projection.
    fig, ax = plt.subplots(1, figsize=(10, 10),
                           subplot_kw={'projection': stamen_terrain.crs})

    lon0 = float(conf['lon0'])
    lon1 = float(conf['lon1'])
    lat0 = float(conf['lat0'])
    lat1 = float(conf['lat1'])
    extent = (lon0, lon1, lat0, lat1)
    ax.set_extent(extent)

    ax.add_image(stamen_terrain, 11)
    # ax.coastlines('10m')
    ax.gridlines(draw_labels=True, color='#777777', linestyle='--')

    pga_text = []
    pga_text_tmp = ''
    norm, cmap = colormap()
    # sort cmp_ids by station name
    cmp_ids = sorted(attributes, key=lambda x: x.split('.')[1])
    n = 0
    for cmp_id in cmp_ids:
        cmp_attrib = attributes[cmp_id]
        lon = cmp_attrib['longitude']
        lat = cmp_attrib['latitude']
        if not (lon0 <= lon <= lon1 and lat0 <= lat <= lat1):
            continue
        pga = cmp_attrib['pga']
        ax.plot(lon, lat, marker='^', markersize=12,
                markeredgewidth=1, markeredgecolor='k',
                color=cmap(norm(pga)),
                transform=ccrs.Geodetic(), zorder=10)
        stname = cmp_id.split('.')[1]
        plot_station_name(lon, lat, stname, ax)

        if n > 0 and n % 7 == 0:
            pga_text.append(pga_text_tmp)
            pga_text_tmp = ''
        pga_text_tmp += '{:>4s}: {:5.1f}\n'.format(stname, pga)
        n += 1
    pga_text.append(pga_text_tmp)
    plot_figure_text(event, pga_text, conf)

    # Add a colorbar
    ax_divider = make_axes_locatable(ax)
    cax = ax_divider.append_axes('right', size='100%', pad='-30%', aspect=15.,
                                 map_projection=stamen_terrain.crs)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, cax=cax)
    cax.get_yaxis().set_visible(True)
    cax.set_ylabel('PGA (mg)')

    outfile = basename + '_pga_map.png'
    fig.savefig(outfile, dpi=300, bbox_inches='tight')
    print('\nMap plot saved to {}'.format(outfile))
    # outfile = basename + '_map.pdf'
    # fig.savefig(outfile, dpi=300, bbox_inches='tight')
    # print('\nMap plot saved to {}'.format(outfile))


def write_attributes(event, attributes, basename):
    outfile = basename + '_attributes.txt'
    fp = open(outfile, 'w')
    fp.write(
        '#{} {} lon {:8.4f} lat {:8.4f} depth {:8.3f} mag {:.2f}\n'.format(
            event['id_sc3'], event['id'],
            event['lon'], event['lat'], event['depth'], event['mag'])
        )
    fp.write('#id                pga      pgv    psa03   psa10   psa30\n')
    fp.write('#                 (mg)     (m/s)    (mg)    (mg)    (mg)\n')
    fp.write('#\n')
    for cmp_id in sorted(attributes.keys()):
        cmp_attrib = attributes[cmp_id]
        pga = cmp_attrib['pga']
        pgv = cmp_attrib['pgv']
        psa03 = cmp_attrib['psa03']
        psa10 = cmp_attrib['psa10']
        psa30 = cmp_attrib['psa30']
        fp.write('{:14s} {:7.3f} {:.3e} {:7.3f} {:7.3f} {:7.3f}\n'.format(
                    cmp_id, pga, pgv, psa03, psa10, psa30))
    print('\nText file saved to {}'.format(outfile))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('xml_file')
    parser.add_argument('xml_dat_file')
    parser.add_argument('out_dir')
    parser.add_argument('-c', '--config', type=str, default='pga_map.conf',
                        help='Config file name')
    args = parser.parse_args()
    return args


def parse_config(config_file):
    # prepend a [root] namespace
    conf_str = '[root]\n' + open(config_file, 'r').read()
    conf_fp = StringIO(conf_str)
    cfg = ConfigParser()
    cfg.readfp(conf_fp)
    return dict(cfg.items('root'))


def main():
    args = parse_args()
    conf = parse_config(args.config)

    event = parse_event_xml(args.xml_file)
    attributes = parse_event_dat_xml(args.xml_dat_file)

    year = event['id'][:4]
    month = event['id'][4:6]
    day = event['id'][6:8]
    out_path = os.path.join(args.out_dir, year, month, day, event['id_sc3'])
    try:
        os.makedirs(out_path)
    except FileExistsError:
        pass
    basename = '{}_{}'.format(event['id'], event['id_sc3'])
    basename = os.path.join(out_path, basename)
    plotmap(attributes, event, basename, conf)
    write_attributes(event, attributes, basename)


if __name__ == '__main__':
    main()
