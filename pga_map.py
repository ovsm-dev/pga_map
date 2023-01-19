#!/usr/bin/env python
# -*- coding: utf8 -*-
"""
Parses event XML files (ShakeMap) and plots PGAs in mg on a map.

(c) 2017-2018 - Claudio Satriano <satriano@ipgp.fr>
                Félix Léger <leger@ipgp.fr>
                Jean-Marie Saurel <saurel@ipgp.fr>
"""
from __future__ import print_function
import os
script_path = os.path.dirname(os.path.realpath(__file__))
from glob import glob
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
import numpy as np
import matplotlib as mpl
mpl.use('agg')  # NOQA
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
import cartopy.io.img_tiles as cimgt
import cartopy.crs as ccrs
from adjustText import adjust_text
from pyproj import Geod
import pdfkit
from pdf2image import convert_from_path
import re


class PgaMap(object):
    """Class for creating a PGA map report."""

    conf = None
    lon0 = None
    lon1 = None
    lat0 = None
    lat1 = None
    event = None
    attributes = None
    out_path = None
    fileprefix = None
    basename = None
    # markers for soil conditions
    markers = {'R': '^', 'S': 'o', 'U': 's'}

    def parse_config(self, config_file):
        """Parse config file."""
        # Transform initial "=" sign to "#" (comment),
        # then all "|" signs to "="
        conf_str =\
            open(config_file, 'r').read().replace('=', '#').replace('|', '=')
        # Prepend a [root] namespace for ConfigParser compatibility
        conf_str = '[root]\n' + conf_str
        conf_fp = StringIO(conf_str)
        cfg = ConfigParser()
        # Keep variable names in capital letters
        cfg.optionxform = str
        cfg.read_file(conf_fp)
        self.conf = dict(cfg.items('root'))
        self.conf['soil_conditions'] = False
        self.lon0 = float(self.conf['MAP_XYLIM'].split(',')[0])
        self.lon1 = float(self.conf['MAP_XYLIM'].split(',')[1])
        self.lat0 = float(self.conf['MAP_XYLIM'].split(',')[2])
        self.lat1 = float(self.conf['MAP_XYLIM'].split(',')[3])
        legend_locations = {
            'LL': 'lower left',
            'LR': 'lower right',
            'UL': 'upper left',
            'UR': 'upper right',
        }
        self.legend_loc = legend_locations[self.conf['MAP_LEGEND_LOC']]
        try:
            self.colorbar_bcsf = bool(self.conf['COLORBAR_BCSF'])
        except Exception:
            self.colorbar_bcsf = False
        try:
            debug = self.conf['DEBUG']
            self.debug = re.search(
                '^(Y|YES|OK|ON|1)$', debug, re.IGNORECASE) is not None
        except KeyError:
            self.debug = False

    def parse_event_xml(self, xml_file):
        """Parse an event.xml file (input for ShakeMap)."""
        event = dict()
        xmldoc = minidom.parse(xml_file)
        tag_earthquake = xmldoc.getElementsByTagName('earthquake')[0]
        event['year'] = int(tag_earthquake.attributes['year'].value)
        event['month'] = int(tag_earthquake.attributes['month'].value)
        event['day'] = int(tag_earthquake.attributes['day'].value)
        event['hour'] = int(tag_earthquake.attributes['hour'].value)
        event['minute'] = int(tag_earthquake.attributes['minute'].value)
        event['second'] = int(tag_earthquake.attributes['second'].value)
        event['time'] = datetime(
            event['year'], event['month'], event['day'],
            event['hour'], event['minute'], event['second'])
        event['timestr'] = event['time'].strftime('%Y%m%dT%H%M%S')
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
        self.event = event

    def parse_event_dat_xml(self, xml_file, soil_conditions_file=None):
        """
        Parse an event_dat.xml file (input for ShakeMap).

        Creates a dictionary of channel attributes:
            lon, lat, pga, pgv, psa03, psa10, psa30
        """
        soil_conditions = {}
        soil_cnd_codes = {'soil': 'S', 'rock': 'R', 'NA': 'U'}
        if soil_conditions_file is not None:
            self.conf['soil_conditions'] = True
            for line in open(soil_conditions_file, 'r'):
                line = line.strip()
                if not line:
                    continue
                if line[0] == '#':
                    continue
                station, soil_cnd = line.split()
                soil_conditions[station] = soil_cnd_codes[soil_cnd]
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
                    try:
                        soil_cnd = soil_conditions[stname]
                    except KeyError:
                        soil_cnd = 'U'
                    cmp_attributes['soil_cnd'] = soil_cnd
                    attributes[cmp_id] = cmp_attributes
        self.attributes = attributes

    def make_path(self, out_dir):
        """Create the output path."""
        event = self.event
        year = '{:04d}'.format(event['year'])
        month = '{:02d}'.format(event['month'])
        day = '{:02d}'.format(event['day'])
        self.out_path = os.path.join(
            out_dir, year, month, day, event['id_sc3'])
        try:
            os.makedirs(self.out_path)
        except FileExistsError:
            pass
        self.fileprefix = '{}_{}'.format(event['timestr'], event['id_sc3'])
        self.basename = os.path.join(self.out_path, self.fileprefix)

    def _colormap(self):
        if self.colorbar_bcsf:
            colors = [
                '#CCCCCC',
                '#70FFFF',
                '#00FF00',
                '#FCFF00',
                '#FFA800',
                '#FF0000',
                '#C60000',
                '#850000',
                '#A7009B',
                '#18009D'
            ]
            cmap = mpl.colors.LinearSegmentedColormap.from_list(
                'pga_cmap', colors, len(colors))
            # BCSF bounds (in %g)
            bounds = np.array(
                [0.02, 0.07, 0.3, 1.1, 4.7, 8.6, 16, 29, 52, 96, 100])
            # convert bounds to mg
            bounds *= 10.
            norm = mpl.colors.BoundaryNorm(
                boundaries=bounds, ncolors=len(colors))
            return norm, cmap, bounds[:-1]
        else:
            vmin = float(self.conf['COLORBAR_PGA_MIN_MAX'].split(',')[0])
            vmax = float(self.conf['COLORBAR_PGA_MIN_MAX'].split(',')[1])
            # Normalizing color scale
            norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
            colors = [
                '#FFE39A',
                '#FFBE6A',
                '#FF875F',
                '#F3484E',
                '#E6004F',
                '#BD0064',
                '#7B0061'
            ]
            ncols = int(vmax-vmin)
            cmap = mpl.colors.LinearSegmentedColormap.from_list(
                'pga_cmap', colors, ncols)
            return norm, cmap, None

    def _select_stations_pga(self):
        attributes = self.attributes
        lon0 = self.lon0
        lon1 = self.lon1
        lat0 = self.lat0
        lat1 = self.lat1
        # select cmp_ids inside geographic area
        cmp_ids = [cmp_id for cmp_id in attributes
                   if lon0 <= attributes[cmp_id]['longitude'] <= lon1
                   and lat0 <= attributes[cmp_id]['latitude'] <= lat1]
        if len(cmp_ids) == 0:
            raise Exception(
                'No stations in the selected area. No plot generated.')
        # sort cmp_ids by station name
        cmp_ids = sorted(cmp_ids, key=lambda x: x.split('.')[1])
        return cmp_ids

    def _plot_circles(self, ax):
        geodetic_transform = ccrs.PlateCarree()
        g = Geod(ellps='WGS84')
        evlat = self.event['lat']
        evlon = self.event['lon']
        # Following values are for testing
        # evlat = 14.6
        # evlon = -61
        # evlat = 14.9
        # evlon = -60.8
        # evlat = 14.4
        # evlon = -61.2
        # evlat = 12.4
        # evlon = -63.2
        ax.plot(evlon, evlat, marker='*', markersize=12,
                markeredgewidth=1, markeredgecolor='k',
                color='green', transform=geodetic_transform,
                zorder=10)
        evdepth = self.event['depth']
        for hypo_dist in np.arange(10, 500, 10):
            if hypo_dist <= evdepth:
                continue
            dist = (hypo_dist**2 - evdepth**2)**0.5
            azimuths = np.arange(0, 360, 1)
            circle = np.array(
                [g.fwd(evlon, evlat, az, dist*1e3)[0:2] for az in azimuths]
            )
            dlon = (self.lon1-self.lon0)*0.01
            dlat = (self.lat1-self.lat0)*0.01
            circle_visible = circle[
                (circle[:, 0] > self.lon0+dlon) &
                (circle[:, 0] < self.lon1-dlon) &
                (circle[:, 1] > self.lat0+dlat) &
                (circle[:, 1] < self.lat1-dlat)
            ]
            if len(circle_visible) < 2:
                continue
            p0 = circle_visible[np.argmax(circle_visible[:, 0])]
            p1 = circle_visible[np.argmax(circle_visible[:, 1])]
            # check if p1 and p0 are too close
            if sum(np.abs(p1-p0)) <= 1e-2:
                p1 = circle_visible[np.argmin(circle_visible[:, 1])]
            # ignore p1 if it is still too close
            if sum(np.abs(p1-p0)) <= 1e-1:
                p1 = None
            ax.plot(circle[:, 0], circle[:, 1],
                    color='#777777', linestyle='--',
                    transform=geodetic_transform)
            dist_text = '{:d} km'.format(hypo_dist)
            t = plt.text(p0[0], p0[1], dist_text, size=6, weight='bold',
                         verticalalignment='center',
                         horizontalalignment='right',
                         transform=geodetic_transform, zorder=10)
            t.set_path_effects([
                path_effects.Stroke(linewidth=0.8, foreground='white'),
                path_effects.Normal()
            ])
            if p1 is not None:
                t = plt.text(p1[0], p1[1], dist_text, size=6, weight='bold',
                             verticalalignment='center',
                             horizontalalignment='left',
                             transform=geodetic_transform, zorder=10)
                t.set_path_effects([
                    path_effects.Stroke(linewidth=0.8, foreground='white'),
                    path_effects.Normal()
                ])

    def plot_map(self):
        """Plot the PGA map."""
        stamen_terrain = cimgt.StamenTerrain()
        geodetic_transform = ccrs.PlateCarree()

        # Create a GeoAxes
        fig, ax = plt.subplots(1, figsize=(10, 10),
                               subplot_kw={'projection': geodetic_transform})

        extent = (self.lon0, self.lon1, self.lat0, self.lat1)
        ax.set_extent(extent)

        ax.add_image(stamen_terrain, 11)
        # ax.coastlines('10m')
        ax.gridlines(draw_labels=True, color='#777777', linestyle='--')
        self._plot_circles(ax)

        norm, cmap, bounds = self._colormap()

        unknown_soils = False
        cmp_ids = self._select_stations_pga()
        texts = []
        markers = []
        for n, cmp_id in enumerate(cmp_ids):
            cmp_attrib = self.attributes[cmp_id]
            lon = cmp_attrib['longitude']
            lat = cmp_attrib['latitude']
            pga = cmp_attrib['pga']
            marker = '^'
            if self.conf['soil_conditions']:
                soil_cnd = cmp_attrib['soil_cnd']
                if soil_cnd == 'U':
                    unknown_soils = True
                marker = self.markers[soil_cnd]
            m, = ax.plot(
                lon, lat, marker=marker, markersize=12,
                markeredgewidth=1, markeredgecolor='k',
                color=cmap(norm(pga)),
                transform=geodetic_transform, zorder=10)
            markers.append(m)
            stname = cmp_id.split('.')[1]
            t = ax.text(lon, lat, stname, size=8, weight='bold', zorder=99)
            t.set_path_effects([
                path_effects.Stroke(linewidth=1.5, foreground='white'),
                path_effects.Normal()
            ])
            texts.append(t)

        if self.conf['soil_conditions']:
            kwargs = {
                'markersize': 8,
                'markeredgewidth': 1,
                'markeredgecolor': 'k',
                'color': '#cccccc',
                'linewidth': 0,
                'transform': geodetic_transform
            }
            rock_station, = ax.plot(
                -self.lon0, -self.lat0, marker=self.markers['R'],
                label='rock', **kwargs)
            soil_station, = ax.plot(
                -self.lon0, -self.lat0, marker=self.markers['S'],
                label='soil', **kwargs)
            handles = [rock_station, soil_station]
            if unknown_soils:
                unk_station, = ax.plot(
                    -self.lon0, -self.lat0, marker=self.markers['U'],
                    label='unknown', **kwargs)
                handles.append(unk_station)
            legend = ax.legend(handles=handles, loc=self.legend_loc)
            legend.set_zorder(99)

        # Add a colorbar
        ax_divider = make_axes_locatable(ax)
        cax = ax_divider.append_axes('right', size='100%',
                                     pad='-30%', aspect=15.,
                                     map_projection=stamen_terrain.crs)
        cax.background_patch.set_visible(False)
        cax.outline_patch.set_visible(False)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        if self.colorbar_bcsf:
            fig.colorbar(sm, extend='max', ticks=bounds, cax=cax)
        else:
            fig.colorbar(sm, extend='max', cax=cax)
        cax.get_yaxis().set_visible(True)
        cax.set_ylabel('PGA (mg)')

        adjust_text(texts, add_objects=markers)

        outfile = self.basename + '_pga_map_fig.png'
        fig.savefig(outfile, dpi=300, bbox_inches='tight')

    def _b3(self, M, R, uncertainty=False):
        """Compute the B3 law (Beauducel et al., 2011)."""
        a = 0.61755
        b = -0.0030746
        c = -3.3968
        logPGA_uncertainty = 0.47
        PGA = 10.**(a*M + b*R - np.log10(R) + c)
        if uncertainty:
            PGA_lower = 10.**(a*M + b*R - np.log10(R) + c - logPGA_uncertainty)
            PGA_upper = 10.**(a*M + b*R - np.log10(R) + c + logPGA_uncertainty)
            return PGA, PGA_lower, PGA_upper
        else:
            return PGA

    def plot_pga_dist(self):
        """Plot PGA as a function of distance."""
        event = self.event
        fig, ax = plt.subplots(figsize=(8, 2.5))
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.grid(True, which='both', ls='--', color='#bbbbbb')
        ax.set_xlabel('Hypocentral distance (km)')
        ax.set_ylabel('PGA (mg)')
        # plot the b3 law (Beauducel et al., 2011)
        M = event['mag']
        R = np.logspace(-1, 3, 50)
        PGA, PGA_lower, PGA_upper = self._b3(M, R, uncertainty=True)
        b3_curve, = ax.plot(R, PGA*1e3, label=r'$B^3$: M {:.1f}'.format(M))
        kwargs = {
            'color': '#999999', 'linestyle': '--', 'label': 'uncertainty'}
        b3_uncertainty, = ax.plot(R, PGA_lower*1e3, **kwargs)
        b3_uncertainty, = ax.plot(R, PGA_upper*1e3, **kwargs)
        legend_handles = [b3_curve, b3_uncertainty]

        g = Geod(ellps='WGS84')
        evlat = event['lat']
        evlon = event['lon']
        evdepth = event['depth']

        norm, cmap, _ = self._colormap()

        min_hypo_dist = 1e10
        unknown_soils = False
        cmp_ids = self._select_stations_pga()
        for cmp_id in cmp_ids:
            cmp_attrib = self.attributes[cmp_id]
            lon = cmp_attrib['longitude']
            lat = cmp_attrib['latitude']
            _, _, dist = g.inv(lon, lat, evlon, evlat)
            dist /= 1000.
            hypo_dist = (dist**2 + evdepth**2)**0.5
            if hypo_dist < min_hypo_dist:
                min_hypo_dist = hypo_dist
            pga = cmp_attrib['pga']
            marker = 'o'
            if self.conf['soil_conditions']:
                soil_cnd = cmp_attrib['soil_cnd']
                if soil_cnd == 'U':
                    unknown_soils = True
                marker = self.markers[soil_cnd]
            ax.scatter(
                hypo_dist, pga, marker=marker, color=cmap(norm(pga)),
                edgecolor='k', alpha=0.5, zorder=99
            )
        if self.conf['soil_conditions']:
            kwargs = {'color': '#cccccc', 'edgecolor': 'k'}
            rock = ax.scatter(
                0, 0, marker=self.markers['R'], label='rock', **kwargs)
            soil = ax.scatter(
                0, 0, marker=self.markers['S'], label='soil', **kwargs)
            legend_handles += [rock, soil]
            if unknown_soils:
                unk = ax.scatter(
                    0, 0, marker=self.markers['U'], label='unknown', **kwargs)
                legend_handles += [unk]
        ax.legend(handles=legend_handles)
        if min_hypo_dist <= 1:
            ax.set_xlim(0.5, 500)
            ax.set_ylim(1e-2, 1e6)
        else:
            ax.set_xlim(10, 500)
            ax.set_ylim(1e-2, 1e4)
        outfile = self.basename + '_pga_dist_fig.png'
        fig.savefig(outfile, dpi=300, bbox_inches='tight')

    def _build_pga_table_html(self, html):
        """Build the PGA info table for the HTML report."""
        cmp_ids = self._select_stations_pga()
        # find max pga and corresponding station
        pga_list = [(cmp_id.split('.')[1], self.attributes[cmp_id]['pga'])
                    for cmp_id in cmp_ids]
        pga_max_sta, pga_max = max(pga_list, key=lambda x: x[1])
        if self.conf['soil_conditions']:
            _soil_cnds = [
                self.attributes[cmp_id]['soil_cnd'] for cmp_id in cmp_ids]
            if 'U' in _soil_cnds:
                pga_title = 'PGA (mg) (R/S/U: rock/soil/unknown)'
            else:
                pga_title = 'PGA (mg) (R/S: rock/soil)'
        else:
            pga_title = 'PGA (mg)'
        html = html.replace('%PGA_TITLE', pga_title)
        nsta = len(cmp_ids)
        nrows = 7
        ncols = int(np.ceil(nsta/nrows))
        rows = ''
        for nr in range(nrows):
            rows += '\n<tr>'
            for nc in range(ncols):
                placeholder = '%STA{:02d}'.format(nr + nc*nrows)
                rows += \
                    '\n  <td class="left">{}</td>'.format(placeholder)
                placeholder = '%PGA{:02d}'.format(nr + nc*nrows)
                rows += \
                    '\n  <td class="right">{}</td>'.format(placeholder)
            rows += '\n</tr>'
        cmp_ids = sorted(cmp_ids, key=lambda x: x.split('.')[1])
        for n, cmp_id in enumerate(cmp_ids):
            cmp_attrib = self.attributes[cmp_id]
            pga = cmp_attrib['pga']
            pga_text = '{:5.1f}'.format(pga).replace(' ', '&nbsp;')
            stname = cmp_id.split('.')[1]
            if self.conf['soil_conditions']:
                soil_cnd = cmp_attrib['soil_cnd']
                st_text = '{}({}):'.format(stname, soil_cnd)
            else:
                st_text = '{}:'.format(stname)
            if stname == pga_max_sta:
                st_text = '<b>*' + st_text + '</b>'
                pga_text = '<b>' + pga_text + '</b>'
            rows = rows\
                .replace('%STA{:02d}'.format(n), st_text)\
                .replace('%PGA{:02d}'.format(n), pga_text)
        # remove extra rows
        for nn in range(n+1, nrows*ncols):
            rows = rows\
                .replace('%STA{:02d}'.format(nn), '')\
                .replace('%PGA{:02d}'.format(nn), '')
        html = html.replace('%ROWS', rows)
        return html

    def write_html(self):
        """Write the output HTML file."""
        event = self.event
        template_html = os.path.join(script_path, 'template.html')
        html = open(template_html, 'r').read()
        title = 'Peak Ground Acceleration &ndash; ' + self.conf['REGION']
        evid = event['id_sc3']
        subtitle = evid + ' &ndash; '
        date = event['time'].strftime('%Y-%m-%d %H:%M:%S')
        subtitle += date + ' &ndash; '
        subtitle += 'M {:.1f}'.format(event['mag'])

        # Event info table
        lat = '{:8.4f}'.format(event['lat']).replace(' ', '&nbsp;')
        lon = '{:8.4f}'.format(event['lon']).replace(' ', '&nbsp;')
        depth = '{:.3f} km'.format(event['depth']).replace(' ', '&nbsp;')
        mag = '{:.2f}'.format(event['mag']).replace(' ', '&nbsp;')
        html = html\
            .replace('%TITLE', title)\
            .replace('%SUBTITLE', subtitle)\
            .replace('%EVID', evid)\
            .replace('%DATE', date)\
            .replace('%LAT', lat)\
            .replace('%LON', lon)\
            .replace('%DEPTH', depth)\
            .replace('%MAG', mag)

        # PGA info table
        html = self._build_pga_table_html(html)

        # Map file
        map_fig_file = self.basename + '_pga_map_fig.png'
        map_fig_file = os.path.basename(map_fig_file)
        html = html.replace('%MAP', map_fig_file)
        # PGA-dist file
        pga_dist_fig_file = self.basename + '_pga_dist_fig.png'
        pga_dist_fig_file = os.path.basename(pga_dist_fig_file)
        html = html.replace('%PGA_DIST', pga_dist_fig_file)
        # Footer
        datestr = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        footer_text = '(c) OVS-IPGP ' + datestr
        html = html.replace('%FOOTER_TEXT', footer_text)

        # Write HTML file
        html_file = self.basename + '_pga_map.html'
        with open(html_file, 'w') as fp:
            fp.write(html)
        if self.debug:
            print('\nHTML report saved to {}'.format(html_file))

        # Link CSS file and logos
        styles_orig = os.path.join(script_path, 'styles.css')
        styles_link = os.path.join(self.out_path, 'styles.css')
        if os.access(styles_link,  os.F_OK):
            os.remove(styles_link)
        os.symlink(styles_orig, styles_link)
        logos = os.path.join(script_path, 'logos', '*.png')
        for logo in glob(logos):
            logo_link = os.path.join(self.out_path, os.path.basename(logo))
            if os.access(logo_link,  os.F_OK):
                os.remove(logo_link)
            os.symlink(logo, logo_link)

    def write_pdf(self):
        """Convert HTML file to PDF."""
        html_file = self.basename + '_pga_map.html'
        pdf_file = self.basename + '_pga_map.pdf'
        pdfkit_options = {
            'dpi': 300,
            'margin-bottom': '0cm',
            'quiet': '',
            'enable-local-file-access': None
        }
        pdfkit.from_file(html_file, pdf_file, options=pdfkit_options)
        print('\nPDF report saved to {}'.format(pdf_file))

    def write_images(self):
        """Convert PDF file to full size PNG and generate a JPEG thumbnail."""
        pdf_file = self.basename + '_pga_map.pdf'
        png_file = self.basename + '_pga_map.png'
        thumb_file = self.basename + '_pga_map.jpg'
        page = convert_from_path(pdf_file, dpi=300)[0]
        page.save(png_file, 'PNG')
        print('\nPNG report saved to {}'.format(png_file))
        thumb_width = 200
        size = page.size
        ratio = thumb_width/size[0]
        thumb_height = int(size[1]*ratio)
        page_thumb = page.resize((thumb_width, thumb_height))
        page_thumb.save(thumb_file, 'JPEG')
        print('\nThumbnail saved to {}'.format(thumb_file))

    def write_attributes(self):
        """Write attributes text file."""
        event = self.event
        attributes = self.attributes
        outfile = self.basename + '_pga_map.txt'
        fp = open(outfile, 'w')
        fp.write(
            '#{} {} lon {:8.4f} lat {:8.4f} depth {:8.3f} mag {:.2f}\n'.format(
                event['id_sc3'], event['timestr'],
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

    def make_symlinks(self):
        """Create symbolic links."""
        cwd = os.getcwd()
        os.chdir(self.out_path)
        for ext in ['txt', 'jpg', 'png', 'pdf']:
            filename = self.fileprefix + '_pga_map.' + ext
            if not os.access(filename,  os.F_OK):
                continue
            if os.access('pga_map.' + ext,  os.F_OK):
                os.remove('pga_map.' + ext)
            os.symlink(filename, 'pga_map.' + ext)
        os.chdir(cwd)

    def clean_intermediate_files(self):
        if self.debug:
            return
        html_file = self.basename + '_pga_map.html'
        os.remove(html_file)
        map_fig_file = self.basename + '_pga_map_fig.png'
        os.remove(map_fig_file)
        pga_dist_fig_file = self.basename + '_pga_dist_fig.png'
        os.remove(pga_dist_fig_file)
        styles_link = os.path.join(self.out_path, 'styles.css')
        if os.access(styles_link,  os.F_OK):
            os.remove(styles_link)
        logos = os.path.join(script_path, 'logos', '*.png')
        for logo in glob(logos):
            logo_link = os.path.join(self.out_path, os.path.basename(logo))
            if os.access(logo_link,  os.F_OK):
                os.remove(logo_link)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('xml_file')
    parser.add_argument('xml_dat_file')
    parser.add_argument('out_dir')
    parser.add_argument('-s', '--soil_conditions_file', type=str,
                        default=None,
                        help='Soil conditions file (default: None)')
    parser.add_argument('-c', '--config', type=str, default='PROC.PGA_MAP',
                        help='Config file name (default: PROC.PGA_MAP)')
    args = parser.parse_args()
    return args


def main():
    """Run the main code."""
    args = parse_args()
    pgamap = PgaMap()
    pgamap.parse_config(args.config)
    pgamap.parse_event_xml(args.xml_file)
    pgamap.parse_event_dat_xml(args.xml_dat_file, args.soil_conditions_file)
    pgamap.make_path(args.out_dir)
    pgamap.plot_map()
    pgamap.plot_pga_dist()
    pgamap.write_html()
    pgamap.write_pdf()
    pgamap.write_images()
    pgamap.write_attributes()
    pgamap.make_symlinks()
    pgamap.clean_intermediate_files()


if __name__ == '__main__':
    main()
