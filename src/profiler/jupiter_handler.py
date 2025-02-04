# -*- coding: utf-8 -*-
"""
Created on Tue Mar  8 10:02:11 2022

@author: au701230
"""

import fdb
import pandas as pd
import os
import geopandas as gp
import numpy as np
import matplotlib.pyplot as plt
import utm
import ast
import contextily as cx
import re
from matplotlib.patches import Polygon
import textwrap

__location__ = os.path.realpath(
    os.path.join(os.getcwd(), os.path.dirname(__file__))
)

class Jupiter:
    """ 
    A class to handle data from the GEUS database.
    
    Attributes
    ----------
    fdb_path : str
        Path to the .fdb file
    df : pd.DataFrame
        DataFrame to store Jupiter data
    start_date : str
        Start date for water level records (format: dd.mm.yyyy)
    end_date : str
        End date for water level records (format: dd.mm.yyyy)
    utm_x_range : list
        UTM x-coordinate range
    utm_y_range : list
        UTM y-coordinate range
    latlong_lat : float
        Latitude for lat/long range
    latlong_long : float
        Longitude for lat/long range
    query_type : str
        Type of query ('geo', 'hydro', etc.)
    con : fdb.Connection
        Connection to the Firebird database 
    utm_x_surveylocs : list
        UTM x-coordinates of survey locations
    utm_y_surveylocs : list
        UTM y-coordinates of survey locations
    """
    
    def __init__(self):
        self.fdb_path = None
        self.df = None
        self.start_date = None
        self.end_date = None
        self.utm_x_range = None
        self.utm_y_range = None
        self.latlong_lat = None
        self.latlong_long = None
        self.query_type = None
        self.con = None
        self.utm_x_surveylocs = None
        self.utm_y_surveylocs = None

    def load_geo_dicts(self):
        """Loads geological dictionaries from files."""
        
        file_path = os.path.join(__location__, "dictionaries/rock_colors.txt")
        with open(file_path, "r") as file:
            self.rock_colors_dict = ast.literal_eval(file.read())

        file_path = os.path.join(__location__, "dictionaries/rock_name.txt")
        with open(file_path, "r") as file:
            self.rock_name_dict = ast.literal_eval(file.read())

        file_path = os.path.join(__location__, "dictionaries/rock_simple.txt")
        with open(file_path, "r") as file:
            self.rock_simple_dict = ast.literal_eval(file.read())

    def connect_database(self, fdb_path=None):
        """Establishes a connection to the Firebird database."""
        
        self.fdb_path = fdb_path
        
        if not self.fdb_path:
            print('Path to Firebird database has not been specified.')
            return
        
        if not os.path.isfile(self.fdb_path):
            print('Path to Firebird database is incorrect.')
            return
        
        print(f'Firebird database is located at {fdb_path}.')
        self.con = fdb.connect(dsn=self.fdb_path, user='sysdba', password='masterkey')
        print('Connection to Firebird database was successful.')
        
    def set_query_type(self, query_type=None):
        """Sets the type of query to be used."""
        
        self.query_type = query_type
        
        if self.query_type == 'geo':
            print('Geology query string will be used.')
        elif self.query_type == 'hydro':
            print('Hydrogeology string will be used.')
        else:
            print('Query type was not recognised.')

    def set_bounds(self, utm_x_range=None, utm_y_range=None, latlong_lat_range=None, latlong_long_range=None):
        """Sets the boundaries for the query."""
        
        if utm_x_range and utm_y_range:
            self.utm_x_range = utm_x_range
            self.utm_y_range = utm_y_range
        elif latlong_lat_range and latlong_long_range:
            x_min, y_min = utm.from_latlon(latlong_lat_range[0], latlong_long_range[0])
            x_max, y_max = utm.from_latlon(latlong_lat_range[1], latlong_long_range[1])
            self.utm_x_range = [x_min, x_max]
            self.utm_y_range = [y_min, y_max]
        else:
            print('Coordinates have not been provided, all of Denmark will be queried.')
            self.utm_x_range = [0, 999999999999]
            self.utm_y_range = [0, 999999999999]

    def set_survey_locs(self, utm_x_locs=None, utm_y_locs=None, latlong_lat_locs=None, latlong_long_locs=None, survey_labels=None):
        """Sets survey locations using either UTM or lat/long coordinates."""
        
        if utm_x_locs and utm_y_locs:
            self.utm_x_surveylocs = utm_x_locs
            self.utm_y_surveylocs = utm_y_locs
        elif latlong_lat_locs and latlong_long_locs:
            self.utm_x_surveylocs = []
            self.utm_y_surveylocs = []
            for lat, lon in zip(latlong_lat_locs, latlong_long_locs):
                x, y = utm.from_latlon(lat, lon)
                self.utm_x_surveylocs.append(x)
                self.utm_y_surveylocs.append(y)
        else:
            print('Coordinates have not been provided.')
            return
        
        self.surveylabels = survey_labels

    def create_query_string(self):
        """Creates the SQL query string based on the query type."""
        
        if self.query_type == 'geo':
            self.query_string = (
                f'SELECT BOREHOLE.BOREHOLENO, BOREHOLE.XUTM, BOREHOLE.YUTM, BOREHOLE.ELEVATION, '
                f'LITHSAMP.TOP, LITHSAMP.BOTTOM, LITHSAMP.ROCKTYPE, LITHSAMP.ROCKSYMBOL '
                f'FROM BOREHOLE '
                f'INNER JOIN LITHSAMP ON BOREHOLE.BOREHOLENO = LITHSAMP.BOREHOLENO '
                f'WHERE BOREHOLE.XUTM > {self.utm_x_range[0]} AND BOREHOLE.XUTM < {self.utm_x_range[1]} '
                f'AND BOREHOLE.YUTM > {self.utm_y_range[0]} AND BOREHOLE.YUTM < {self.utm_y_range[1]};'
            )
            print('Query string has been set.')

        elif self.query_type == 'hydro':
            self.query_string = (
                f'SELECT BOREHOLE.BOREHOLENO, BOREHOLE.XUTM, BOREHOLE.YUTM, BOREHOLE.ELEVATION, '
                f'WATLEVEL.WATLEVMSL, WATLEVEL.WATLEVGRSU, WATLEVEL.TIMEOFMEAS '
                f'FROM BOREHOLE '
                f'INNER JOIN WATLEVEL ON BOREHOLE.BOREHOLENO = WATLEVEL.BOREHOLENO '
                f'WHERE BOREHOLE.XUTM > {self.utm_x_range[0]} AND BOREHOLE.XUTM < {self.utm_x_range[1]} '
                f'AND BOREHOLE.YUTM > {self.utm_y_range[0]} AND BOREHOLE.YUTM < {self.utm_y_range[1]};'
            )

        elif self.query_type == 'screen':
            self.query_string = (
                f'SELECT BOREHOLE.BOREHOLENO, BOREHOLE.XUTM, BOREHOLE.YUTM, BOREHOLE.ELEVATION, '
                f'SCREEN.TOP, SCREEN.BOTTOM, SCREEN.DIAMETERMM '
                f'FROM BOREHOLE '
                f'INNER JOIN WATLEVEL ON BOREHOLE.BOREHOLENO = WATLEVEL.BOREHOLENO '
                f'WHERE BOREHOLE.XUTM > {self.utm_x_range[0]} AND BOREHOLE.XUTM < {self.utm_x_range[1]} '
                f'AND BOREHOLE.YUTM > {self.utm_y_range[0]} AND BOREHOLE.YUTM < {self.utm_y_range[1]};'
            )
            print('Query string has been set.')

    def filter_dataframe(self, header_name='ROCKSYMBOL'):
        """Filters the DataFrame to remove boreholes with missing or specific values."""

        # Count and display the number of boreholes
        self.num_boreholes = len(self.df['BOREHOLENO'].unique())
        print(f'{self.num_boreholes} boreholes were found.')
        
        # Identify boreholes with missing or specific data
        
        print(f'Boreholes with null entries in {header_name} will be removed from dataframe.')
        borehole_id = self.df.loc[self.df[header_name].isnull(), 'BOREHOLENO'].unique()
       
        # Remove identified boreholes from the DataFrame
        for borehole in borehole_id:
            self.df = self.df[self.df.BOREHOLENO != borehole]
        
        self.num_boreholes = len(self.df['BOREHOLENO'].unique())
        
        # Update record of borehole ids
        self.borehole_ids = self.df['BOREHOLENO'].unique()
        print(f'{self.num_boreholes} boreholes remain after filtering.')


    def plot_cyklo_log(self, borehole_id, ax=None, bounds=None):
        """Plots a circular log of geological layers for a given borehole."""

        # Create a new figure and axes if ax is None
        if ax is None:
            fig, ax = plt.subplots()  
        
        # Filter data for the selected borehole
        bh_idx = np.where(self.borehole_ids==borehole_id)[0][0]
        bh = self.bh_list[bh_idx]
        depth = np.max(bh['bot_depths'])

        outer_vals = []
        outer_cols = []
    
        inner_vals = []
        inner_cols = []

        min_elev = np.min(bh['elevation'] - bh['bot_depths'])

        if min_elev < -100:
            plt.close()
            return('Not implemented for depths exceeding 100 m')
        
        num_layers_bsl = len(np.where(bh['elevation'] - bh['bot_depths'] < 0)[0])
        num_layers_asl = len(np.where(bh['elevation']- bh['top_depths'] > 0)[0])

        thk = np.diff(bh['top_depths'])
        thk_bsl = np.array(thk[-num_layers_bsl:])
        thk_asl = np.array(thk[:num_layers_asl])

        if num_layers_bsl + num_layers_asl > bh['n_layers']:
            bg = np.sum(thk_asl) - bh['elevation']
            if bg < 0:
                return('Error! Something fucked up.')
            thk_asl[-1] = thk_asl[-1] - bg
            thk_bsl[0] = bg

        # No layers below sea level, set outer layer to white
        if num_layers_bsl == 0: 
            outer_vals = [100]  
            outer_cols = ['white']
        
        # Define sections for log bsl
        else:
            white_space = 100 - np.sum(thk_bsl)
            outer_vals = thk_bsl.tolist()
            outer_vals.append(white_space)
            if num_layers_bsl + num_layers_asl > bh['n_layers']:
                outer_cols = bh['colors'][num_layers_asl-1:].tolist()
            else:
                outer_cols = bh['colors'][num_layers_asl:].tolist()
            outer_cols.append('white')

        # No layers above sea level, set inner layer to white
        if num_layers_asl == 0: 
            inner_vals = [100]  
            inner_cols = ['white']

        # Layers above sea level, set inner layer to white    
        if num_layers_asl > 0: 
            if bh['elevation'] - np.sum(thk_asl) > 0: # need to add white to both sides
    
                white_space = 100 - np.sum(thk_asl)
                upper_white_space = 100 - bh['elevation']
                lower_white_space = white_space - upper_white_space
    
                inner_vals = thk_asl.tolist()
                inner_cols = bh['colors'][:num_layers_asl].tolist()
    
                inner_vals.insert(0, upper_white_space)
                inner_vals.append(lower_white_space)
    
                inner_cols.insert(0, 'white')
                inner_cols.append('white')

            # Only add white to upper surface
            else: 
                
                white_space = 100 - np.sum(thk_asl)
                upper_white_space = 100 - bh['elevation']
    
                inner_vals = thk_asl.tolist()
                inner_cols = bh['colors'][:num_layers_asl].tolist()
    
                inner_vals.insert(0, upper_white_space)
                inner_cols.insert(0, 'white')
    
        #flip inner values for plotting
        #inner_cols = inner_cols[::-1]
        #inner_vals = inner_vals[::-1]
    
        size = 0.2

        if bounds:
            ec = 'k'
        
        else:
            ec = None

        ax.pie(outer_vals, radius=1, colors=outer_cols, counterclock=False, startangle=180,
               wedgeprops=dict(width=size, edgecolor=ec, linewidth=0.1))
    
        ax.pie(inner_vals, radius=1-size, colors=inner_cols, counterclock=False, startangle=180,
               wedgeprops=dict(width=size/1.75, edgecolor=ec, linewidth=0.1))
    
        ax.pie([25, 25, 25, 25], radius=1, colors=['#FF000000','#FF000000'], counterclock=True, startangle=180,
               wedgeprops=dict(width=size, edgecolor='lightgrey', linewidth=0.5))   
    
        ax.pie([25, 25, 25, 25], radius=1-size, colors=['#FF000000','#FF000000'], counterclock=True, startangle=180,
               wedgeprops=dict(width=size/1.75, edgecolor='lightgrey', linewidth=0.5))
        
        ax.set_title(bh['id'])
        ax.set(aspect="equal")
        

    def plot_strat_log(self, borehole_id, ax=None, bounds=False):
        """Plots a stratigraphic log of geological layers for a given borehole."""

        # Create a new figure and axes if ax is None
        if ax is None:
            fig, ax = plt.subplots()  
        
        # Filter data for the selected borehole
        bh_idx = np.where(self.borehole_ids==borehole_id)[0][0]
        bh = self.bh_list[bh_idx]
        depth = np.max(bh['bot_depths'])
        
        # Set up plot
        ax.set_title(f'{borehole_id}', fontsize=14)
        ax.set_ylim([depth, 0])

        if bounds:
            ec = 'k'
        
        else:
            ec = None
        
        # Plot each geological layer as a rectangle
        for i in range(bh['n_layers']):
            layer = np.array([[-1/2, bh['top_depths'][i]], 
                              [1/2, bh['top_depths'][i]], 
                              [1/2, bh['bot_depths'][i]], 
                              [-1/2, bh['bot_depths'][i]], 
                              [-1/2, bh['top_depths'][i]]])
            poly = Polygon(layer, facecolor = bh['colors'][i], edgecolor=ec)
            ax.add_patch(poly)
            print(poly)
        
        layer = np.array([[-1/2, bh['top_depths'][0]], 
                          [1/2, bh['top_depths'][0]], 
                          [1/2, bh['bot_depths'][-1]], 
                          [-1/2, bh['bot_depths'][-1]], 
                          [-1/2, bh['top_depths'][0]]])
        
        poly = Polygon(layer, facecolor = 'None', edgecolor='k')
        ax.add_patch(poly)

        ax.set_ylabel('Depth [m]')

        ax.set_xlim([-1/2, 1/2]) 
        ax.set_xticks([])

    def plot_map(self, ax=None, bounds=None, basemap=None):
        """Plots a map with the locations of boreholes."""
        
        # Create a new figure and axes if ax is None
        if ax is None:
            fig, ax = plt.subplots()  
        
        # Plot borehole locations
        ax.scatter(self.df['XUTM'], self.df['YUTM'], c='r', 
                   marker='o', s=5, label='Boreholes')
        
        if bounds is not None:
            min_x, min_y = bounds[0]
            max_x, max_y = bounds[1]
            ax.set_xlim([min_x, max_x])
            ax.set_ylim([min_y, max_y])

        if basemap is not None:
             cx.add_basemap(ax, crs='epsg:32632', 
                            source=cx.providers.Esri.WorldImagery, 
                            attribution=False)
        
        ax.set_xlabel('UTM X')
        ax.set_ylabel('UTM Y')
        ax.legend()

    def df_to_dict(self):
        """Returns list of bh dictionaries from the borehole dataframe.
        
       Parameters
       ----------
       df : pandas dataframe
           Dataframe of geo data, must have ELEVATION, TOP, BOTTOM, and ROCKSYMBOL as headers
            
        """

        self.bh_list = []
        
        for borehole_id in self.borehole_ids:
            df_borehole = self.df[self.df['BOREHOLENO'] == borehole_id]
        
            # Create a dictionary for bh log
            bh = {'id' : df_borehole['BOREHOLENO'].values[0]} 
            bh['id'] = re.sub(pattern = ' ', repl='', string = bh['id'])
        
            bh['x'] = df_borehole['XUTM'].values[0]
            bh['y'] = df_borehole['YUTM'].values[0]
            bh['elevation'] = df_borehole['ELEVATION'].values[0]
    
            # Sort rows in df by top of unit
            df_borehole = df_borehole.sort_values(by=['TOP'], axis=0) 
    
            # Assign values to dictionary
            bh['top_depths'] = df_borehole['TOP'].values
            bh['bot_depths'] = df_borehole['BOTTOM'].values
        
            if np.isnan(bh['bot_depths'][-1]):
                bh['bot_depths'][-1] = np.floor(bh['top_depths'][-1])
    
            bh['rock_symbol'] = df_borehole['ROCKSYMBOL'].values
                              
            rock_type = []
            rock_summary = []
            rock_col = []
        
            for i in range(len(bh['rock_symbol'])):
                if bh['rock_symbol'][i] is None:
                    return(bh)
                
                rock_type.append(self.rock_name_dict[bh['rock_symbol'][i]])
                rock_summary.append(self.rock_simple_dict[bh['rock_symbol'][i]])
                rock_col.append(self.rock_colors_dict[rock_summary[i]])
        
            bh['lith_names'] = np.array(rock_type)
            bh['simple'] = rock_summary
            bh['colors'] = np.array(rock_col)
            
            bh['n_layers'] = int(len(bh['bot_depths']))

            self.bh_list.append(bh)
        
    def query_database(self):
        """Queries the database and loads the data into a DataFrame."""
        
        self.create_query_string()
        self.load_geo_dicts()
        
        # Execute the query
        self.df = pd.read_sql(self.query_string, self.con)
        
        # Convert DataFrame to GeoDataFrame
        self.df = gp.GeoDataFrame(
            self.df, geometry=gp.points_from_xy(self.df['XUTM'], self.df['YUTM'])
        )

        self.borehole_ids = self.df['BOREHOLENO'].unique()
        
        # Filter DataFrame and create list of bh dictionaries for geo query
        if self.query_type == 'geo':
            self.filter_dataframe(header_name='ROCKSYMBOL')
            self.df_to_dict()
        
        #self.filter_dataframe(header_name='ELEVATION')

    def filter_by_depth(self, min_depth=None, max_depth=None):
        """
        Filters the boreholes in the DataFrame based on their depth.
        
        Parameters:
        min_depth (float): Minimum depth to filter boreholes. Boreholes with depths greater than this value will be kept.
        max_depth (float): Maximum depth to filter boreholes. Boreholes with depths less than this value will be kept.
        """
        if self.df is None:
            print("Dataframe was not found.")
            return
        
        if min_depth is not None:
            self.df = self.df[self.df['BOTTOM'] > min_depth]
            print(f"Boreholes with depths greater than {min_depth} have been selected.")
        
        if max_depth is not None:
            self.df = self.df[self.df['BOTTOM'] < max_depth]
            print(f"Boreholes with depths less than {max_depth} have been selected.")
        
        self.num_boreholes = len(self.df['BOREHOLENO'].unique())
        print(f'{self.num_boreholes} boreholes remain after depth filtering.')


    def plot_lith_key(self, ax=None, max_line_length=50, title='Geological Key',
                      label_names=None, drop_idx=None):

        lith_cols = []
        lith_names = []
        top_depths = []

        # Iterate through the list of dictionaries and extract unique values
        for bh in self.bh_list:
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
  
        lith_key[:] = [lith_key[i] for i in idx] 
        unique_cols[:] = [unique_cols[i] for i in idx] 

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
