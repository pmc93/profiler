# -*- coding: utf-8 -*-
"""
Created on 8 October 11:47:54 2023

@author: pmc93
"""

import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
#from colormaps import getAarhusCols, getParulaCols
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap
from matplotlib.collections import PolyCollection
from matplotlib.patches import Polygon
import re
import contextily as cx
import textwrap
from matplotlib import cm


class Model:

    def __init__(self):
        self.ttem_models = {} 
        self.stem_models = []
        self.profiles = []
        self.boreholes = []

    def loadXYZ(self, xyz_path, return_mod_df=False, mod_name=None,
                model_type='tTEM'):
        """

        Parameters
        ----------
        xyz_path : STRING
            Path to Aarhus Workbench .xyz file.
        return_mod_df : BOOLEAN
            Return dataframe with all data

        Returns
        -------
        x : ARRAY OF FLOATS
            x coordinates of model data
        y : ARRAY OF FLOATS
            y coordinates of model data
        elevation : TYPE
            DESCRIPTION.
        rhos : TYPE
            DESCRIPTION.
        depths : TYPE
            DESCRIPTION.
        doi_con : TYPE
            DESCRIPTION.
        doi_standard : TYPE
            DESCRIPTION.

        """

        # Scan through file and find where model part of file starts
        f = open(xyz_path, "r")

        tem_model = {}

        file_lines = []
        line = f.readline()

        while line != '':
            line = f.readline()
            file_lines.append(line)

            if 'DATA TYPE' in line:
                line = f.readline()
                file_lines.append(line)
                tem_model['instrument'] = line.replace("/DT", "").replace("\n", "")

            if 'COORDINATE SYSTEM' in line:
                line = f.readline()
                file_lines.append(line)

                # Use regular expression to find the last number in the string
                espg_code = re.search(r'epsg:(\d+)', line).group(1)

                tem_model['epsg'] = 'epsg:' + espg_code


            if 'LINE_NO' in line:
                row_idx = len(file_lines)

        # Read data into pandas data frame
        mod_df = pd.read_csv(xyz_path, delimiter=None, skiprows=row_idx)

        col_names = mod_df.columns[0].split()[1:]

        mod_df = mod_df[mod_df.columns[0]].str.split(expand=True)

        mod_df.columns = col_names

        if 'DATE' in col_names:
            mod_df = mod_df.drop(['DATE'], axis=1)

        if 'TIME' in col_names:
            mod_df = mod_df.drop(['TIME'], axis=1)

        mod_df = mod_df.astype(float)

        # Extract relevant columns from dataframe
        rho_cols = [col for col in mod_df.columns
                    if 'RHO' in col and 'STD' not in col]

        depth_cols = [col for col in mod_df.columns
                      if 'DEP_BOT' in col and 'STD' not in col]

        rhos = mod_df[rho_cols].values

        depths = mod_df[depth_cols].values

        doi_con = mod_df['DOI_CONSERVATIVE'].values
        doi_standard = mod_df['DOI_STANDARD'].values

        x = mod_df['UTMX'].values
        y = mod_df['UTMY'].values
        elev = mod_df['ELEVATION'].values

        residual = mod_df['RESDATA'].values
        line_num = mod_df['LINE_NO'].astype(int).values

        if mod_name is not None:
            tem_model['mod_name'] = mod_name

        else:
            tem_model['mod_name'] = len(self.tem_models)

        tem_model['x'] = x
        tem_model['y'] = y
        tem_model['elev'] = elev
        tem_model['rhos'] = rhos
        tem_model['depths'] = depths
        tem_model['doi_con'] = doi_con
        tem_model['doi_standard'] = doi_standard
        tem_model['residual'] = residual
        tem_model['line_num'] = line_num

        if return_mod_df:
            tem_model['mod_df'] = mod_df
            
        if model_type == 'tTEM':

            #if self.ttem_models != tem_model:
            self.ttem_models.update(tem_model)

            #else:
            #    print('tTEM data is a duplicate, it was not added.')
            
        elif model_type == 'sTEM':
            
            self.stem_models.append(tem_model)  # Add only if unique
            
        else:
            
            print('Model type not recognized, no data loaded.')
        

    def dict_to_tuple(d):
        """ Convert NumPy arrays inside a dictionary to tuples for comparison. """
        return {k: tuple(v) if isinstance(v, np.ndarray) else v for k, v in d.items()}

    def loadProfileCoords(self, profile_coord_paths, file_type='csv'):

        if file_type == 'csv':

            for profile_coord_path in profile_coord_paths:

                profile_coord = pd.read_csv(profile_coord_path).iloc[:, 0:2].values

                profile = {}
                profile['x'] = profile_coord[:, 0]
                profile['y'] = profile_coord[:, 1]

                self.profiles.append(profile)

        elif file_type == 'shp':

            for profile_coord_path in profile_coord_paths:

                profile_coords = self.readShpFile(profile_coord_path)

                for profile_coord in profile_coords:

                    profile = {}
                    profile['x'] = profile_coord[:, 0]
                    profile['y'] = profile_coord[:, 1]

                    self.profiles.append(profile)

        else:
            print('File type was not recognised, choose "csv" or "shp".')


    def readShpFile(self, shp_file_path):

        gdf = gpd.read_file(shp_file_path)

        sections = []

        for index, row in gdf.iterrows():
            geometry = row['geometry']

            # Check if the geometry is a LineString
            if geometry.geom_type == 'LineString':
                # Access the LineString coordinates
                section_coords = list(geometry.coords)

                sections.append(np.array(section_coords))

        return sections

    def interpCoords(self, x, y, distance=10):
        interpolated_points = []

        for i in range(len(x)-1):
            x1 = x[i]
            y1 = y[i]
            x2 = x[i+1]
            y2 = y[i+1]
            dx = x2 - x1
            dy = y2 - y1
            segments = int(np.sqrt(dx**2 + dy**2) / distance)

            if segments != 0:

                for j in range(segments + 1):
                    xi = x1 + dx * (j / segments)
                    yi = y1 + dy * (j / segments)
                    interpolated_points.append((xi, yi))

        interpolated_points = np.array(interpolated_points)

        xi, yi = interpolated_points[:, 0], interpolated_points[:, 1]
        dists = (np.diff(xi) ** 2 + np.diff(yi) ** 2) ** 0.5
        dists = np.cumsum(np.insert(dists, 0, 0))
        idx = np.unique(dists, return_index=True)[1]

        return xi[idx], yi[idx], dists[idx]

    def interpIDW(self, x, y, z, xi, yi, power=2, interp_radius=10):
        """

        Parameters
        ----------
        x : TYPE
            DESCRIPTION.
        y : TYPE
            DESCRIPTION.
        z : TYPE
            DESCRIPTION.
        xi : TYPE
            DESCRIPTION.
        yi : TYPE
            DESCRIPTION.
        power : TYPE, optional
            DESCRIPTION. The default is 2.
        interp_radius : TYPE, optional
            DESCRIPTION. The default is 10.

        Returns
        -------
        None.

        """
        # Calculate distances between grid points and input points
        dists = np.sqrt((x[:, np.newaxis] - xi[np.newaxis, :])**2 +
                        (y[:, np.newaxis] - yi[np.newaxis, :])**2)

        # Calculate weights based on distances and power parameter
        weights = 1.0 / (dists + np.finfo(float).eps)**power

        # Set weights to 0 for points outside the specified radius
        weights[dists > interp_radius] = 0

        # Normalize weights for each grid point
        weights /= np.sum(weights, axis=0)

        # Interpolate values using weighted average
        zi = np.sum(z[:, np.newaxis] * weights, axis=0)

        return zi

    def interpLIN(self, x, z):
        """


        Parameters
        ----------
        x : TYPE
            DESCRIPTION.
        z : TYPE
            DESCRIPTION.

        Returns
        -------
        zi : TYPE
            DESCRIPTION.

        """

        # Create interpolation function, excluding nan values
        interp_func = interp1d(x[~np.isnan(z)], z[~np.isnan(z)],
                               kind='linear', fill_value="extrapolate")
        zi = z.copy()
        zi[np.isnan(z)] = interp_func(x[np.isnan(z)])

        if np.isnan(zi[-1]):
            zi[-1] = zi[-2]

        return zi

    def createProfiles(self, profile_idx='all', model_spacing=10, interp_radius=40):
        """

        Parameters
        ----------
        model_idx : TYPE
            DESCRIPTION.
        profile_idx : TYPE
            DESCRIPTION.
        model_spacing : TYPE, optional
            DESCRIPTION. The default is 10.
        interp_radius : TYPE, optional
            DESCRIPTION. The default is 40.

        Returns
        -------
        None.

        """

        if profile_idx == 'all':

            profile_idx = range(0, len(self.profiles))

        elif type(profile_idx) != list:

            profile_idx = [profile_idx]

        for idx in profile_idx:

            x = self.profiles[idx]['x']
            y = self.profiles[idx]['y']

            if model_spacing != False:

                xi, yi, dists = self.interpCoords(x, y, distance=model_spacing)

            else:
                xi = self.profiles[idx]['x']
                yi = self.profiles[idx]['y']
                dists = np.concatenate(([0], 
                                        np.cumsum((np.diff(xi)**2 + np.diff(yi)**2)**0.5)))

            rhos = self.ttem_models['rhos']
            depths = self.ttem_models['depths']
            x = self.ttem_models['x']
            y = self.ttem_models['y']
            elev = self.ttem_models['elev']
            doi = self.ttem_models['doi_standard']

            n_layers = rhos.shape[1]
            n_models = len(xi)
            rhos_new = np.zeros((n_models, n_layers))

            for i in range(rhos.shape[1]):
                rhos_new[:, i] = self.interpIDW(x, y, rhos[:, i], xi, yi, power=2,
                                                interp_radius=interp_radius)

            depths_new = np.repeat(depths[0, :][None, :], n_models, axis=0)

            elev_new = self.interpIDW(x, y, elev, xi, yi, power=2,
                                      interp_radius=interp_radius)

            elev_new = self.interpLIN(dists, elev_new)

            doi_new = self.interpIDW(x, y, doi, xi, yi, power=2,
                                     interp_radius=interp_radius)
            doi_new = self.interpLIN(dists, doi_new)

            self.profiles[idx]['rhos'] = rhos_new
            self.profiles[idx]['depths'] = depths_new
            self.profiles[idx]['elev'] = elev_new
            self.profiles[idx]['doi'] = doi_new
            self.profiles[idx]['distances'] = dists
            self.profiles[idx]['xi'] = xi
            self.profiles[idx]['yi'] = yi


    def find_gradient_depths(self, rhos, depths, elev, min_elev, max_elev, mode="Resistivity", log_transform=True):
        """
        Identifies depths where the maximum gradient change occurs within a given elevation range.

        Parameters:
            rhos (np.ndarray): 1D array of resistivity values.
            depths (np.ndarray): 1D array of depth values (same length as rhos).
            elev (float): Surface elevation.
            min_elev (float): Minimum elevation constraint.
            max_elev (float): Maximum elevation constraint.
            mode (str): "resistivity" (default) or "conductivity". 
                        If "conductivity", uses 1/rho.
            log_transform (bool): If True, applies log10 transformation to resistivity/conductivity before computing gradients.

        Returns:
            np.ndarray: Array of depths where the maximum gradient occurs within the elevation range.
        """
        gradient_depths = []

        rhos = rhos[:-1]

        # Convert depths to elevations
        elevs = elev - depths

        # Convert resistivity to conductivity if needed
        if mode == "Conductivity":
            rhos = 1 / rhos  # Convert resistivity to conductivity

        # Apply log-transform if required
        if log_transform:
            rhos = np.log10(rhos)

        # Compute gradient (first derivative)
        gradients = np.abs(np.gradient(rhos, depths))

        # Find indices of local maxima (where gradient changes significantly)
        max_gradient_idx = np.where((gradients[1:-1] > gradients[:-2]) & (gradients[1:-1] > gradients[2:]))[0] + 1

        # Filter by elevation range
        for i in max_gradient_idx:
            if min_elev <= elevs[i] <= max_elev:
                gradient_depths.append(depths[i])

        return np.array(gradient_depths)


    def find_iso_surface_depths(self, rhos, depths, elev, target_rho, min_elev, max_elev, log_transform=True):
        """
        Identifies depths where the resistivity matches the target value within a given elevation range.

        Parameters:
            rhos (np.ndarray): 1D array of resistivity values.
            depths (np.ndarray): 1D array of depth values (same length as rhos).
            elev (float): Surface elevation.
            target_rho (float): The resistivity value to find.
            min_elev (float): Minimum elevation constraint.
            max_elev (float): Maximum elevation constraint.
            log_transform (bool): If True, applies log10 transformation to resistivity values for interpolation.

        Returns:
            np.ndarray: Array of depths where the resistivity matches the target within the elevation range.
        """
        iso_surface_depths = []

        # Convert depths to elevations
        elevs = elev - depths

        # Apply log-transform if required
        if log_transform:
            rhos = np.log10(rhos)
            target_rho = np.log10(target_rho)

        # Loop through each depth and check if resistivity matches the target within elevation range
        for i in range(len(rhos) - 1):
            if min_elev <= elevs[i] <= max_elev:
                # Check if the resistivity crosses the target value between two consecutive points
                if (rhos[i] - target_rho) * (rhos[i + 1] - target_rho) <= 0:
                    # Linearly interpolate to find the exact depth
                    fraction = (target_rho - rhos[i]) / (rhos[i + 1] - rhos[i])
                    interpolated_depth = elevs[i] + fraction * (elevs[i + 1] - elevs[i])
                    iso_surface_depths.append(interpolated_depth)
                else:
                    iso_surface_depths.append(np.nan)

            else: iso_surface_depths.append(np.nan)

        return np.array(iso_surface_depths)

    def loadBoreholes(self, borehole_paths, file_type='dat'):
        """

        Parameters
        ----------
        borehole_path : TYPE
            DESCRIPTION.

        Returns
        -------
        borehole_dict : TYPE
            DESCRIPTION.

        """

        if file_type == 'dat':
            for borehole_path in borehole_paths:
                bh_df = pd.read_csv(borehole_path, sep='\t')

                bh_dict = {}

                bh_dict['id'] = bh_df['id'][0]
                bh_dict['n_layers'] = len(bh_df)
                bh_dict['x'] = bh_df['utm_x'].values[0]
                bh_dict['y'] = bh_df['utm_y'].values[0]
                bh_dict['elevation'] = bh_df['elev'].values[0]
                bh_dict['top_depths'] = bh_df['top_depths'].values
                bh_dict['bot_depths'] = bh_df['bot_depths'].values
                bh_dict['colors'] = bh_df['colors'].values
                bh_dict['descriptions'] = bh_df['lith_descriptions']
                bh_dict['lith_names'] = bh_df['lith_names'].values

                self.boreholes.append(bh_dict)

        elif file_type == 'xlsx':
            for borehole_path in borehole_paths:
                excel_file = pd.ExcelFile(borehole_path)
                sheet_names = excel_file.sheet_names

                for sheet_name in sheet_names:
                    bh_dict = {}
                    bh_df = excel_file.parse(sheet_name)
                    bh_dict['id'] = bh_df['id'][0]
                    bh_dict['n_layers'] = len(bh_df)
                    bh_dict['x'] = bh_df['utm_x'].values[0]
                    bh_dict['y'] = bh_df['utm_y'].values[0]
                    if 'elev' in bh_dict:
                        bh_dict['elevation'] = bh_df['elev'].values[0]
                    else:
                        bh_dict['elevation'] = 0
                    bh_dict['top_depths'] = bh_df['top_depths'].values
                    bh_dict['bot_depths'] = bh_df['bot_depths'].values
                    bh_dict['colors'] = bh_df['colors'].values
                    bh_dict['lith_names'] = bh_df['lith_names'].values
                    self.boreholes.append(bh_dict)

        else:
            print('File type was not recognised, choose "csv" or "xlsx".')


class Plot:

    def __init__(self, model):
        self.model = model

    def profileMap(self, ax=None, background='imagery'):

        if background == 'imagery':
            source = cx.providers.Esri.WorldImagery
            ttem_color = 'w'
            stem_color = 'w'

        elif background == 'osm':
            source = cx.providers.OpenStreetMap.Mapnik
            ttem_color = 'k'
            stem_color = 'w'

        else:
            print('Background not recognised, specify either "imagery" or "osm".')

        if ax is None:
            fig, ax = plt.subplots(figsize=(5, 8))
        else:
            fig = ax.figure

        for i in range(len(self.model.ttem_models)):

            if i == 0:
                ax.scatter(self.model.ttem_models[i]['x'],
                           self.model.ttem_models[i]['y'],
                           marker='.', c=ttem_color, s=1, label='tTEM data')

            else:
                ax.scatter(self.model.tem_models[i]['x'],
                           self.model.tem_models[i]['y'],
                           marker='.', c=ttem_color, s=1)

        for i in range(len(self.model.stem_models)):

            if i == 0:
                ax.scatter(self.model.stem_models[i]['x'],
                           self.model.stem_models[i]['y'],
                           marker='.', c=stem_color, s=10, ec='k',
                           label='sTEM data')

            else:
                ax.scatter(self.model.tem_models[i]['x'],
                           self.model.tem_models[i]['y'],
                           marker='.', c=ttem_color, s=10, ec='k')

        if len(self.model.profiles) > 10:
            colorscale = cm.get_cmap('jet', len(self.model.profiles))

            cols = colorscale(np.linspace(0, 1, len(self.model.profiles)))

            for i in range(len(self.model.profiles)):
                ax.plot(self.model.profiles[i]['x'],
                        self.model.profiles[i]['y'], c=cols[i], lw=2, alpha=0.8,
                        label='Profile ' + str(i+1))

        else:
            for i in range(len(self.model.profiles)):
                ax.plot(self.model.profiles[i]['x'],
                        self.model.profiles[i]['y'],lw=2, alpha=0.8,
                        label='Profile ' + str(i+1))


        cx.add_basemap(ax, crs=self.model.ttem_models[0]['epsg'],  source=source,
                       attribution_size=2)

        #ax.ticklabel_format(useOffset=False, style='plain')
        ax.set_xlabel("Easting [m]")
        ax.set_ylabel("Northing [m]")
        ax.grid()
        leg = ax.legend()
        leg.legendHandles[0]._sizes = [20]
        ax.set_xlabel('Distance [m]')
        fig.tight_layout()

    def getColors(self, rhos, vmin, vmax, cmap=plt.cm.viridis, n_bins=16,
                  log=True, discrete_colors=False):
        """
        Return colors from a color scale based on numerical values

        Parameters
        ----------
        rhos : TYPE
            DESCRIPTION.
        vmin : TYPE
            DESCRIPTION.
        vmax : TYPE
            DESCRIPTION.
        cmap : TYPE, optional
            DESCRIPTION. The default is 'viridis'.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """

        if log:
            rhos = np.log10(rhos)
            vmin = np.log10(vmin)
            vmax = np.log10(vmax)

        # Determine color of each polygon
        cmaplist = [cmap(i) for i in range(cmap.N)]

        if discrete_colors:
            cmap = LinearSegmentedColormap.from_list(
                'Custom cmap', cmaplist, cmap.N)

            # define the bins and normalize
            bounds = np.linspace(vmin, vmax, n_bins)
            norm = BoundaryNorm(bounds, cmap.N)

        else:
            cmap = LinearSegmentedColormap.from_list(
                'Custom cmap', cmaplist, 256)

            # define the bins and normalize
            bounds = np.linspace(vmin, vmax, 256)
            norm = BoundaryNorm(bounds, 256)

        scalar_map = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        return scalar_map.to_rgba(rhos)

    def TEMProfile(self, profile_idx, doi=True, ax=None, vmin=1, vmax=1000,
                   scale=10, cmap=plt.cm.viridis, log=True,
                   flip=False, cbar=True, cbar_orientation='vertical',
                   zmin=None, zmax=None, xmin=None, xmax=None, plot_title='',
                   cbar_label='Resistivity [Ohm.m]'):
        """

        """

        rhos = self.model.profiles[profile_idx]['rhos']
        depths = self.model.profiles[profile_idx]['depths']
        elev = self.model.profiles[profile_idx]['elev']
        dists = self.model.profiles[profile_idx]['distances']
        doi = self.model.profiles[profile_idx]['doi']

        if rhos.shape[1] == depths.shape[1]:

            depths = depths[:,:-1]

        self.plot2D(rhos=rhos, depths=depths, elev=elev, dists=dists,
                    doi=doi, ax=ax, vmin=vmin, vmax=vmax, cmap=cmap,
                    cbar=cbar, log=log,
                    zmin=zmin, zmax=zmax, xmin=xmin, xmax=xmax,
                    plot_title=plot_title, scale=scale, 
                    cbar_orientation=cbar_orientation)


    def plot2D(self, rhos, depths, elev=None, dists=None, doi=None,
               ax=None, vmin=1, vmax=1000, contour=False, scale=10, 
               cmap=plt.cm.viridis, n_bins=16, discrete_colors=False,
               log=True, flip=False, cbar=True, cbar_orientation='vertical',
               zmin=None, zmax=None, xmin=None, xmax=None, plot_title='',
               cbar_label='Resistivity [Ohm.m]'):
        """

        Parameters
        ----------
        rhos : TYPE
            DESCRIPTION.
        depths : TYPE
            DESCRIPTION.
        elev : TYPE
            DESCRIPTION.
        dists : TYPE
            DESCRIPTION.
        doi : TYPE
            DESCRIPTION.
        vmin : TYPE, optional
            DESCRIPTION. The default is 1.
        vmax : TYPE, optional
            DESCRIPTION. The default is 200.
        hx : TYPE, optional
            DESCRIPTION. The default is 3.5.
        scale : TYPE, optional
            DESCRIPTION. The default is 10.
        cmap : TYPE, optional
            DESCRIPTION. The default is plt.cm.viridis.
        n_bins : TYPE, optional
            DESCRIPTION. The default is 11.
        log : TYPE, optional
            DESCRIPTION. The default is True.
        flip : TYPE, optional
            DESCRIPTION. The default is False.
        zmin : TYPE, optional
            DESCRIPTION. The default is None.
        zmax : TYPE, optional
            DESCRIPTION. The default is None.
        xmin : TYPE, optional
            DESCRIPTION. The default is None.
        xmax : TYPE, optional
            DESCRIPTION. The default is None.

        Returns
        -------
        None.

        """

        # Add extra distance, otherwise problem later on, check why this is...
        if dists is not None:
            dists = np.append(dists, dists[-1])
            plot_model_idx = False
        else:
            plot_model_idx = True

        # Add 0 m depth to depths, could be buggy
        depths = -np.c_[np.zeros(depths.shape[0]), depths]

        n_layers = rhos.shape[1]
        n_models = rhos.shape[0]

        # Transform data and lims into log
        if log:
            rhos = np.log10(rhos)
            vmin = np.log10(vmin)
            vmax = np.log10(vmax)

        if plot_model_idx:
            x = np.arange(0, rhos.shape[0]+1)
            elev = np.zeros_like(x)[:-1]

        else:
            x = dists

        if flip:
            x = x[-1] - x[::-1]
            elev = elev[::-1]
            rhos = rhos[::-1, :]
            doi = doi[::-1]

        # Create boundary of polygons to be drawn
        xs = np.tile(np.repeat(x, 2)[1:-1][:, None], n_layers+1)

        depths = np.c_[np.zeros(depths.shape[0]), depths]
        ys = np.repeat(depths, 2, axis=0) + np.repeat(elev, 2, axis=0)[:, None]
        verts = np.c_[xs.flatten('F'), ys.flatten('F')]

        n_vert_row = verts.shape[0]
        connection = np.c_[np.arange(n_vert_row).reshape(-1, 2),
                           2*(n_models) +
                           np.arange(n_vert_row).reshape(-1, 2)[:, ::-1]]

        ie = (connection >= len(verts)).any(1)
        connection = connection[~ie, :]
        coordinates = verts[connection]

        # Determine color of each polygon
        cmaplist = [cmap(i) for i in range(cmap.N)]

        if discrete_colors:
            cmap = LinearSegmentedColormap.from_list(
                'Custom cmap', cmaplist, cmap.N)

            # define the bins and normalize
            bounds = np.linspace(vmin, vmax, n_bins)
            norm = BoundaryNorm(bounds, cmap.N)

        else:
            cmap = LinearSegmentedColormap.from_list(
                'Custom cmap', cmaplist, 256)

            # define the bins and normalize
            bounds = np.linspace(vmin, vmax, 256)
            norm = BoundaryNorm(bounds, 256)

        # Create polygon collection
        coll = PolyCollection(coordinates, array=rhos.flatten('F'),
                              cmap=cmap, norm=norm, edgecolors=None)

        coll.set_clim(vmin=vmin, vmax=vmax)

        # Add polygons to plot
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(15, 5))
        else:
            fig = ax.figure

        if contour:
            max_depth = 100
            centroid = np.mean(coordinates, axis=1)
            centroidx = centroid[:, 0].reshape((-1, n_models))
            centroidz = centroid[:, 1].reshape((-1, n_models))
            xc = np.vstack([centroidx[0, :], centroidx, centroidx[-1, :]])
            zc = np.vstack([np.zeros(n_models), centroidz, -
                           np.ones(n_models)*max_depth])
            val = np.c_[rhos[:, 0], rhos, rhos[:, -1]].T

            levels = np.linspace(vmin, vmax, 15)

            ax.contourf(xc, zc, val, cmap=cmap, levels=levels, extend='both')

        else:
            ax.add_collection(coll)

        # Blank out models below doi
        if doi is not None:
            doi = (np.repeat(elev, 2) - np.repeat(doi, 2)).tolist()

            doi.append(-1000)
            doi.append(-1000)
            doi.append(doi[-1])

            x_doi = xs[:, 0].tolist()

            x_doi.append(x_doi[-1])
            x_doi.append(x_doi[0])
            x_doi.append(x_doi[0])

            ax.fill(np.array(x_doi),  np.array(doi), edgecolor="none",
                    facecolor='w', alpha=0.8)

        if dists is not None:
            ax.set_xlabel('Distance [m]\n')
        else:
            ax.set_xlabel('Index')

        ax.set_ylabel('Elevation [m]')

        if cbar:

            if cbar_orientation == 'vertical':
                cbar = fig.colorbar(coll, label=cbar_label, ax=ax,
                                    orientation=cbar_orientation, shrink=0.8)

            else:
                cbar = fig.colorbar(coll, label=cbar_label, ax=ax,
                                    orientation=cbar_orientation, shrink=0.7,
                                    fraction=0.06, pad=0.2)

            if log:
                tick_locs = np.arange(int(np.floor(vmin)), int(np.ceil(vmax)))

                if tick_locs[-1] < vmax:
                    tick_locs = np.append(tick_locs, vmax)

                if tick_locs[0] < vmin:
                    tick_locs = np.append(vmin, tick_locs[1:])

                cbar.set_ticks(tick_locs)
                cbar.set_ticklabels(np.round(10**tick_locs).astype(int))
                cbar.ax.minorticks_off()

            else:
                tick_locs = np.arange(vmin, vmax+0.00001, int(vmax-vmin)/4)
                cbar.set_ticks(tick_locs)
                cbar.set_ticklabels(np.round(tick_locs))
                cbar.ax.minorticks_off()

        if zmin is None:
            zmin = np.nanmin(elev)+np.min(depths)

        if zmax is None:
            zmax = np.nanmax(elev)

        if xmin is None:
            xmin = 0

        if xmax is None:
            if dists is not None:
                xmax = dists[-1]
            else:
                xmax = n_models

        ax.set_ylim([zmin, zmax])
        ax.set_xlim([xmin, xmax])

        if len(plot_title) != 0:
            ax.set_title(plot_title)

        ax.set_aspect(scale)

        ax.grid(which='both')
        fig.tight_layout()

    def TEMSounding(self, model_type, model_idx, sounding_idx, vmin=0, vmax=1000, ax=None):

        if model_type == 'tTEM':
            rhos = self.model.ttem_models[model_idx]['rhos'][sounding_idx, :]
            depths = self.model.ttem_models[model_idx]['depths'][sounding_idx, :]
            doi = self.model.ttem_models[model_idx]['doi_con'][sounding_idx]

        else:
            rhos = self.model.stem_models[model_idx]['rhos'][sounding_idx, :]
            depths = self.model.stem_models[model_idx]['depths'][sounding_idx, :]
            doi = self.model.stem_models[model_idx]['doi_con'][sounding_idx]

        if len(rhos) == len(depths):

            depths = depths[:-1]

        self.plot1D(rhos, depths, doi, vmin=vmin, vmax=vmax, ax=ax)

    def plot1D(self, rhos, depths, doi, log=True, vmin=0, vmax=1000,
               title=None, label=None, ax=None,
               col='k', ymin=None, ymax=None):

        if ax is None:
            fig, ax = plt.subplots(figsize=(5, 8))
        else:
            fig = ax.figure

        if doi is not None:
            idx = np.where(depths > doi)[0][0]
            ax.step(rhos[:idx], -np.insert(depths, 0, 0)[:idx], where='pre', c=col, label=label)

            ax.step(rhos[idx-1:], -np.insert(depths, 0, 0)[idx-1:],
                    where='pre', c='grey', ls='-',  alpha=0.8)
        else:
            ax.step(rhos, -np.insert(depths, 0, 0), where='pre', c=col, label=label)

        if log == True:
            ax.set_xscale('log')
            if vmin == 0:
                vmin = 1

        ax.set_xlim([vmin, vmax])
        ax.set_ylim([ymin, ymax])

        if label is not None:

            ax.legend()

        if title is not None:

            ax.set_title(title)

        ax.set_xlabel('Resistivity [Ohm.m]')

        ax.set_ylabel('Elevation [m]')

        ax.grid(True, which='major')
        fig.tight_layout()

    def findNearest(self, dict_data, x, y, dists=None, elev=None):
        """

        Parameters
        ----------
        dict_data : TYPE
            DESCRIPTION.
        x : TYPE
            DESCRIPTION.
        y : TYPE
            DESCRIPTION.
        dists : TYPE
            DESCRIPTION.
        elev : TYPE
            DESCRIPTION.

        Returns
        -------
        dist_loc : TYPE
            DESCRIPTION.
        elev_loc : TYPE
            DESCRIPTION.
        min_dist : TYPE
            DESCRIPTION.

        """

        x_loc = dict_data['x']
        y_loc = dict_data['y']

        idx = np.argmin(((x_loc - x) ** 2 + (y_loc - y) ** 2) ** 0.5)
        min_dist = np.min(((x_loc - x) ** 2 + (y_loc - y) ** 2) ** 0.5)

        if elev is not None:
            elev = elev[idx]
        else:
            elev = dict_data['elevation']

        if dists is not None:

            dist_loc = dists[idx]

        else:
            dist_loc = np.nan

        return dist_loc, elev, min_dist, idx

    def addBorehole(self, bh_idx, ax, bh_width=0.2,
                    text_size=12, x_start=None, x_end=None):
        """


        Parameters
        ----------
        bh_list : TYPE
            DESCRIPTION.
        ax : TYPE
            DESCRIPTION.
        bh_width : TYPE, optional
            DESCRIPTION. The default is dists[-1]/100.

        Returns
        -------
        None.

        """

        bh_dict = self.model.boreholes[bh_idx]

        if x_start is None:
            x_start = ax.get_xlim()[0]
            x_end = 10**(np.log10(ax.get_xlim()[1]) * bh_width)

        for i in range(bh_dict['n_layers']):

            coordinates = np.array(([x_start, -bh_dict['top_depths'][i]],
                                    [x_end, -bh_dict['top_depths'][i]],
                                    [x_end, -bh_dict['bot_depths'][i]],
                                    [x_start, -bh_dict['bot_depths'][i]],
                                    [x_start, -bh_dict['top_depths'][i]]))

            p = Polygon(coordinates, facecolor=bh_dict['colors'][i],
                        edgecolor='k', lw=0)

            ax.add_patch(p)

        coordinates = np.array(([x_start, -bh_dict['top_depths'][0]],
                                [x_end, -bh_dict['top_depths'][0]],
                                [x_end, -bh_dict['bot_depths'][-1]],
                                [x_start, -bh_dict['bot_depths'][-1]],
                                [x_start, -bh_dict['top_depths'][-1]]))

        p = Polygon(coordinates, facecolor='none', edgecolor='k', lw=1)

        ax.add_patch(p)

    def addBoreholes(self, profile_idx, ax, elev=None,
                     search_radius=150, bh_width=None, add_label=False,
                     text_size=12, shift=5, print_msg=False):
        """

        Parameters
        ----------
        bh_list : TYPE
            DESCRIPTION.
        ax : TYPE
            DESCRIPTION.
        bh_width : TYPE, optional
            DESCRIPTION. The default is dists[-1]/100.

        Returns
        -------
        None.

        """

        xi = self.model.profiles[profile_idx]['xi']
        yi = self.model.profiles[profile_idx]['yi']
        dists = self.model.profiles[profile_idx]['distances']
        elevs = self.model.profiles[profile_idx]['elev']

        if bh_width is None:

            bh_width = dists[-1] / 40

        for bh in self.model.boreholes:

            dist_loc, elev, min_dist, idx = self.findNearest(bh, xi, yi,
                                                             dists, elevs)

            if min_dist < search_radius:
                x1 = dist_loc - bh_width/2
                x2 = dist_loc + bh_width/2

                for i in range(bh['n_layers']):
                    verts = np.array(([x1, elev - bh['top_depths'][i]],
                                      [x2, elev - bh['top_depths'][i]],
                                      [x2, elev - bh['bot_depths'][i]],
                                      [x1, elev - bh['bot_depths'][i]],
                                      [x1, elev - bh['top_depths'][i]]))

                    p = Polygon(verts, facecolor=bh['colors'][i], lw=0)
                    ax.add_patch(p)

                # add boundary around log
                verts = np.array(([x1, elev - bh['top_depths'][0]],
                                  [x2, elev - bh['top_depths'][0]],
                                  [x2, elev - bh['bot_depths'][-1]],
                                  [x1, elev - bh['bot_depths'][-1]],
                                  [x1, elev - bh['top_depths'][0]]))

                p = Polygon(verts, facecolor='none', edgecolor='k', lw=1)

                ax.add_patch(p)
                if add_label:
                    ax.text(dist_loc, elev+shift,  bh['id'][4:],
                            horizontalalignment='center', weight='bold',
                            verticalalignment='top', fontsize=text_size)
                if print_msg:
                    print('\033[1mBorehole %s is %.3f km from profile, it was included.\033[0m' % (bh['id'], min_dist/1000))

            else:
                if print_msg:
                    print('Borehole %s is %.3f km from profile, it was not included.' % (bh['id'], min_dist/1000))

    def addNMRSoundings(self, profile_idx, nmr_list, param, ax, vmin=1, vmax=1000, elev=None,
                        log=True, cmap=plt.cm.viridis, n_bins=16, discrete_colors=False,
                        search_radius=100, model_width=None, print_msg=False):
        """

        Parameters
        ----------
        bh_list : TYPE
            DESCRIPTION.
        ax : TYPE
            DESCRIPTION.
        bh_width : TYPE, optional
            DESCRIPTION. The default is dists[-1]/100.

        Returns
        -------
        None.

        """

        xi = self.model.profiles[profile_idx]['xi']
        yi = self.model.profiles[profile_idx]['yi']
        dists = self.model.profiles[profile_idx]['distances']
        elevs = self.model.profiles[profile_idx]['elev']

        n_models = len(nmr_list)

        if model_width is None:
            model_width = dists[-1] / 60

        for nmr in nmr_list:

            dist_loc, elev, min_dist, idx = self.findNearest(nmr, xi, yi,
                                                             dists, elevs)

            if min_dist < search_radius:
                x1 = dist_loc - model_width/2
                x2 = dist_loc + model_width/2

                for i in range(nmr['n_layers']-1):
                    verts = np.array(([x1, elev - nmr['top_depths'][i]],
                                      [x2, elev - nmr['top_depths'][i]],
                                      [x2, elev - nmr['bot_depths'][i]],
                                      [x1, elev - nmr['bot_depths'][i]],
                                      [x1, elev - nmr['top_depths'][i]]))

                    nmr['colors'] = self.getColors(nmr[param],
                                                   vmin=vmin, vmax=vmax,
                                                   log=log, cmap=cmap,
                                                   discrete_colors=discrete_colors)

                    if nmr['bot_depths'][i] > nmr['doi']:
                        p = Polygon(verts, facecolor=nmr['colors'][i],
                                    alpha= 0.3, lw=0)

                    else:
                        p = Polygon(verts, facecolor=nmr['colors'][i],
                                    lw=0)

                    ax.add_patch(p)

                verts = np.array(([x1, elev - nmr['top_depths'][0]],
                                  [x2, elev - nmr['top_depths'][0]],
                                  [x2, elev - nmr['bot_depths'][-1]],
                                  [x1, elev - nmr['bot_depths'][-1]],
                                  [x1, elev - nmr['top_depths'][0]]))

                p = Polygon(verts, facecolor='none', edgecolor='k', lw=0.5)

                ax.add_patch(p)
                
                if print_msg:
                    print('\033[1mTEM sounding %s is %.3f km from profile, it was included.\033[0m' % (nmr['id'], min_dist/1000))
                    

            else:
                if print_msg:
                    print('TEM sounding %s is %.3f km from profile, it was not included.' % (nmr['id'], min_dist/1000))

        if log:
            vmin = np.log10(vmin)
            vmax = np.log10(vmax)
            

    def addTEMSoundings(self, profile_idx, stem_model_idx, ax, vmin=1, vmax=1000, elev=None,
                        log=True, cmap=plt.cm.turbo, n_bins=16, discrete_colors=False,
                        search_radius=100, model_width=None, print_msg=False):
        """

        Parameters
        ----------
        bh_list : TYPE
            DESCRIPTION.
        ax : TYPE
            DESCRIPTION.
        bh_width : TYPE, optional
            DESCRIPTION. The default is dists[-1]/100.

        Returns
        -------
        None.

        """

        xi = self.model.profiles[profile_idx]['xi']
        yi = self.model.profiles[profile_idx]['yi']
        dists = self.model.profiles[profile_idx]['distances']
        elevs = self.model.profiles[profile_idx]['elev']

        n_models = len(self.model.stem_models[stem_model_idx]['x'])

        if model_width is None:
            model_width = dists[-1] / 60

        for i in range(n_models):

            sounding = {}

            sounding['id'] = str(i+1)

            sounding['x'] = self.model.stem_models[stem_model_idx]['x'][i]
            sounding['y'] = self.model.stem_models[stem_model_idx]['y'][i]
            sounding['rhos'] = self.model.stem_models[stem_model_idx]['rhos'][i]
            sounding['depths'] = self.model.stem_models[stem_model_idx]['depths'][i]
            sounding['doi'] = self.model.stem_models[stem_model_idx]['doi_con'][i]
            sounding['n_layers'] = len(sounding['rhos'])
            sounding['top_depths'] = np.insert(sounding['depths'], 0, 0)[:-1]
            sounding['bot_depths'] = sounding['depths']

            dist_loc, elev, min_dist, idx = self.findNearest(sounding,
                                                             xi, yi,
                                                             dists, elevs)

            if min_dist < search_radius:
                x1 = dist_loc - model_width/2
                x2 = dist_loc + model_width/2

                for i in range(sounding['n_layers']-1):
                    verts = np.array(([x1, elev - sounding['top_depths'][i]],
                                      [x2, elev - sounding['top_depths'][i]],
                                      [x2, elev - sounding['bot_depths'][i]],
                                      [x1, elev - sounding['bot_depths'][i]],
                                      [x1, elev - sounding['top_depths'][i]]))

                    sounding['colors'] = self.getColors(sounding['rhos'],
                                                        vmin=vmin, vmax=vmax,
                                                        log=log, cmap=cmap,
                                                        discrete_colors=discrete_colors)

                    if sounding['bot_depths'][i] > sounding['doi']:
                        p = Polygon(verts, facecolor=sounding['colors'][i],
                                    alpha= 0.3, lw=0)

                    else:
                        p = Polygon(verts, facecolor=sounding['colors'][i],
                                    lw=0)

                    ax.add_patch(p)

                verts = np.array(([x1, elev - sounding['top_depths'][0]],
                                  [x2, elev - sounding['top_depths'][0]],
                                  [x2, elev - sounding['bot_depths'][-1]],
                                  [x1, elev - sounding['bot_depths'][-1]],
                                  [x1, elev - sounding['top_depths'][0]]))

                p = Polygon(verts, facecolor='none', edgecolor='k', lw=0.5)

                ax.add_patch(p)

                if print_msg:
                    print('\033[1mTEM sounding %s is %.3f km from profile, it was included.\033[0m' % (sounding['id'], min_dist/1000))

            else:
                if print_msg:
                    print('TEM sounding %s is %.3f km from profile, it was not included.' % (sounding['id'], min_dist/1000))

        if log:
            vmin = np.log10(vmin)
            vmax = np.log10(vmax)

    def lithKey(self, ax=None, max_line_length=50, title='Geological Key',
                label_names=None, drop_idx=None):

        lith_cols = []
        lith_names = []
        top_depths = []

        # Iterate through the list of dictionaries and extract unique values
        for bh in self.model.boreholes:
            lith_cols.append(bh['colors'].tolist())
            lith_names.append(bh['lith_names'].tolist())
            top_depths.append(bh['top_depths'].tolist())

        lith_cols = np.concatenate(lith_cols)
        lith_names = np.concatenate(lith_names)
        top_depths = np.concatenate(top_depths)

        unique_cols = np.unique(lith_cols)

        lith_key = []
        lith_depth = []
        for col in unique_cols:

            idx = np.where(lith_cols == col)[0]

            lith_key.append(np.unique(lith_names[idx]))
            lith_depth.append(np.mean(top_depths[idx]))

        idx = np.argsort(lith_depth)
        
        print(idx)

        print(lith_key)

        lith_key = np.array(lith_key)[idx]
        unique_cols = np.array(unique_cols)[idx]

        if label_names is not None:
            
            lith_key = label_names

        if ax is None:

            fig, ax = plt.subplots(1, 1)
            
        if drop_idx is not None:
            label_names.pop(drop_idx)
            unique_cols=unique_cols[drop_idx+1:]
            #print(lith_key)
            #lith_key.pop(drop_idx)
            
        # Iterate through the geological units and plot colored squares
        y_position = 0.5  # Initial y-position for the first square
        for i in range(len(unique_cols)):

            # Plot a colored square
            ax.add_patch(plt.Rectangle((0, y_position), 0.4, 0.37, color=unique_cols[i]))
            if len(lith_key) > 1:
                label = ' / '.join(lith_key[i])
                
            else:
                label = lith_key[i]
                print(label)
            wrapped_label = '\n'.join(textwrap.wrap(label, max_line_length))

            ax.text(0.6, y_position + 0.15, wrapped_label, va='center', fontsize=12)

            # Update the y-position for the next square
            y_position -= 0.7

        # Set axis limits and labels
        ax.set_xlim(0, 4)
        ax.set_ylim(y_position, 1)
        ax.axis('off')  # Turn off axis
        ax.set_title(title)