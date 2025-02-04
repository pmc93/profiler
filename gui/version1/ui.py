import contextily as cx
import geopandas as gpd
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk
import sys
import re
import os

import profiler
from colormaps import getAarhusCols, getParulaCols



class ProfilerUI:

    def __init__(self, root):
        self.root = root
        self.root.title("Profiler")
        #self.root.iconbitmap("Profiler.ico")

        self.plotter = UIPlot()

        self.setup_ui()

    def setup_ui(self):
        self.snap = 0
        self.misfit = 0
        self.discrete = 0
        self.canvas_frame = ttk.Frame(self.root)
        self.canvas_frame.pack(padx=1, pady=1)

        self.canvas = FigureCanvasTkAgg(
            self.plotter.fig, master=self.canvas_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.toolbar = NavigationToolbar2Tk(self.canvas, self.canvas_frame)
        self.toolbar.update()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Button Frame 1
        button_frame1 = ttk.Frame(self.root)
        button_frame1.pack(fill=tk.BOTH, expand=True)

        map_style = ["None", "Aerial Imagery", "OSM"]
        self.plotter.bg_map = tk.StringVar()
        self.plotter.bg_map.set("Map Background")
        self.cmap_select = tk.OptionMenu(
            button_frame1, self.plotter.bg_map, *map_style)
        self.cmap_select.pack(side=tk.LEFT)

        self.load_button = ttk.Button(
            button_frame1, text="Add tTEM data", command=self.open_tTEM_file)
        self.load_button.pack(side=tk.LEFT, padx=10, pady=5)

        self.load_button = ttk.Button(
            button_frame1, text="Add walkTEM Data",
            command=self.plotter.open_walkTEM_file)
        self.load_button.pack(side=tk.LEFT, padx=10, pady=5)

        self.show_misfit = tk.IntVar()
        self.misfit_button = ttk.Checkbutton(button_frame1,
                                             text="Plot Residual",
                                             variable=self.show_misfit,
                                             onvalue=1, offvalue=0,
                                             command=self.misfit_selection)
        self.misfit_button.pack(side=tk.LEFT, padx=10, pady=5)

        self.load_button = ttk.Button(
            button_frame1, text="Add Borehole Data",
            command=self.plotter.open_borehole_files)
        self.load_button.pack(side=tk.LEFT, padx=10, pady=5)

        self.load_button = ttk.Button(
            button_frame1, text="Convert Jupiter Borehole Data",
            command=self.plotter.open_borehole_files)
        self.load_button.pack(side=tk.LEFT, padx=10, pady=5)

        self.load_button["state"] = "disabled"

        # Button Frame 2
        button_frame2 = ttk.Frame(self.root)
        button_frame2.pack(fill=tk.BOTH, expand=True)

        label_xmin = tk.StringVar()
        label_xmin.set("x min")
        tk.Label(button_frame2, textvariable=label_xmin).pack(side=tk.LEFT)
        self.xmin_entry = ttk.Entry(
            button_frame2, text=self.plotter.xmin, width=5)
        self.xmin_entry.pack(side=tk.LEFT, padx=10, pady=5)

        label_xmax = tk.StringVar()
        label_xmax.set("x max")
        tk.Label(button_frame2, textvariable=label_xmax).pack(side=tk.LEFT)
        self.xmax_entry = ttk.Entry(
            button_frame2, text=self.plotter.xmax, width=5)
        self.xmax_entry.pack(side=tk.LEFT, padx=10, pady=5)

        label_ymin = tk.StringVar()
        label_ymin.set("y min")
        tk.Label(button_frame2, textvariable=label_ymin).pack(side=tk.LEFT)
        self.ymin_entry = ttk.Entry(
            button_frame2, text=self.plotter.ymin, width=5)
        self.ymin_entry.pack(side=tk.LEFT, padx=10, pady=5)

        label_ymax = tk.StringVar()
        label_ymax.set("y max")
        tk.Label(button_frame2, textvariable=label_ymax).pack(side=tk.LEFT)
        self.ymax_entry = ttk.Entry(
             button_frame2, text=self.plotter.ymax, width=5)
        self.ymax_entry.pack(side=tk.LEFT, padx=10, pady=5)

        self.load_button = ttk.Button(
            button_frame2, text="Set Map Limits", command=self.set_map_limits)
        self.load_button.pack(side=tk.LEFT, padx=10, pady=5)

        self.load_button["state"] = "disabled"

        self.load_button = ttk.Button(
            button_frame2, text="Reset Map Limits",
            command=self.reset_map_limits)
        self.load_button.pack(side=tk.LEFT, padx=10, pady=5)

        self.load_button["state"] = "disabled"

        # Button Frame 3
        button_frame3 = ttk.Frame(self.root)
        button_frame3.pack(fill=tk.BOTH, expand=True)

        self.drawing_button = ttk.Button(
            button_frame3, text="Toggle Profile Drawer",
            command=self.toggle_drawing_mode)
        self.drawing_button.pack(side=tk.LEFT, padx=10, pady=5)

        self.snap_on = tk.IntVar()
        self.snap_button = ttk.Checkbutton(
            button_frame3, text="Snap", variable=self.snap_on, onvalue=1,
            offvalue=0, command=self.snap_selection)
        self.snap_button.pack(side=tk.LEFT, padx=10, pady=5)
        self.snap_on.set(1)

        self.open_shp_button = ttk.Button(
            button_frame3, text="Open Shape File",
            command=self.open_shp)
        self.open_shp_button.pack(side=tk.LEFT, padx=10, pady=5)

        self.open_shp_button["state"] = "disabled"

        self.save_coords = ttk.Button(
            button_frame3, text="Save Profile Coordinates",
            command=self.save_profile_points)
        self.save_coords.pack(side=tk.LEFT, padx=10, pady=5)

        # Button Frame 4
        button_frame4 = ttk.Frame(self.root)
        button_frame4.pack(fill=tk.BOTH, expand=True)

        label = tk.StringVar()
        label.set("Lateral\nDiscretization")
        tk.Label(button_frame4, textvariable=label).pack(side=tk.LEFT)

        self.plotter.model_spacing = tk.DoubleVar()
        self.plotter.model_spacing.set(25)
        self.model_spacing_entry = ttk.Entry(
            button_frame4, text=self.plotter.model_spacing, width=5)
        self.model_spacing_entry.pack(side=tk.LEFT, padx=10, pady=5)

        label = tk.StringVar()
        label.set("Search\nRadius")
        tk.Label(button_frame4, textvariable=label).pack(side=tk.LEFT)

        self.plotter.interp_radius = tk.DoubleVar()
        self.plotter.interp_radius.set(25)
        self.interp_radius_entry = ttk.Entry(
            button_frame4, text=self.plotter.interp_radius, width=5)
        self.interp_radius_entry.pack(side=tk.LEFT, padx=10, pady=5)

        self.create_button = ttk.Button(
            button_frame4, text="Create Profile", command=self.create_profile)
        self.create_button.pack(side=tk.LEFT, padx=10, pady=5)


    def open_tTEM_file(self):
        self.plotter.open_tTEM_file()

    def set_map_limits(self):
        self.plotter.xmin = self.xmin_entry.get()
        self.plotter.xmax = self.xmax_entry.get()
        self.plotter.ymin = self.ymin_entry.get()
        self.plotter.ymax = self.ymax_entry.get()
        self.plotter.plot_coordinates()

    def reset_map_limits(self):
        self.plotter.xmin = None
        self.plotter.xmax = None
        self.plotter.ymin = None
        self.plotter.ymax = None
        self.plotter.plot_coordinates()

    def toggle_drawing_mode(self):
        self.plotter.toggle_drawing_mode()

    def save_profile_points(self):
        self.plotter.save_profile_points()

    def create_profile(self):
        self.plotter.create_profile()
        self.plotter.launch_profile_plot()

    def snap_selection(self):
        self.snap = self.snap_on.get()
        self.plotter.set_snap(self.snap)

    def misfit_selection(self):
        self.misfit = self.show_misfit.get()
        self.plotter.set_misfit(self.misfit)
        self.plotter.plot_coordinates()

    def open_shp(self):
        self.plotter.open_shp()


class UIPlot:

    def __init__(self):

        self.fig, self.ax = plt.subplots(dpi=100, figsize=(8, 4))
        self.xmin = None
        self.xmax = None
        self.ymin = None
        self.ymax = None
        self.bg_map = None
        self.current_line = []
        self.file_paths = []
        self.drawing_mode = 0
        self.line_length_text = None
        self.snap = 1
        self.misfit = 0
        self.flip = 0
        self.discrete = 0
        self.walkTEM_list = None
        self.bh_list = None
        self.profile_coords = None
        self.fig.canvas.mpl_connect('button_press_event', self.on_canvas_click)
        self.model = profiler.Model()
        self.model_types = []

    def plot_coordinates(self):
        self.ax.clear()

        if self.current_line:
            x_vals, y_vals = zip(*self.current_line)
            line, = self.ax.plot(x_vals, y_vals, color='r', marker='x')
            self.calculate_length(line)

        for ttem_model in self.model.ttem_models:

            if self.misfit:
                sc = self.ax.scatter(ttem_model['x'],
                                     ttem_model['y'],
                                     c=ttem_model['residual'],
                                     s=0.5, vmin=0, vmax=3, cmap='RdYlGn_r')

                if len(plt.gcf().axes) < 2:
                    self.cb = plt.colorbar(sc).set_label('Residual')

            else:
                self.ax.scatter(ttem_model['x'],
                                ttem_model['y'],
                                c='cyan', s=0.5)

        for stem_model in self.model.stem_models:

            if self.misfit:
                sc = self.ax.scatter(stem_model['x'],
                                     stem_model['y'],
                                     c=stem_model['residual'],
                                     s=8, vmin=0, vmax=3, cmap='RdYlGn_r')

                if len(plt.gcf().axes) < 2:
                    self.cb = plt.colorbar(sc).set_label('Residual')

            else:
                self.ax.scatter(stem_model['x'],
                                stem_model['y'],
                                c='blue', s=8)

        if self.profile_coords is not None:
            x_vals = self.profile_coords[:, 0]
            y_vals = self.profile_coords[:, 1]
            line, = self.ax.plot(x_vals, y_vals, color='r', marker='x')
            self.calculate_length(line)

        if len(self.model.boreholes) != 0:
            for bh in self.model.boreholes:
                self.ax.scatter(bh['x'],
                                bh['y'], c='orange',
                                marker='*', s=20)

        if self.bg_map.get() != 'None':

            if self.bg_map.get() == 'OSM':
                cx.add_basemap(
                   self.ax, crs=self.crs,
                   source=cx.providers.OpenStreetMap.Mapnik,
                   attribution_size=4)

            if self.bg_map.get() == 'Aerial Imagery':
                cx.add_basemap(
                    self.ax, crs=self.crs,
                    source=cx.providers.Esri.WorldImagery,
                    attribution_size=4)

        self.ax.set_aspect(1)
        self.ax.set_xlim(self.xmin, self.xmax)
        self.ax.set_ylim(self.ymin, self.ymax)
        self.ax.set_xlabel("Easting [m]")
        self.ax.set_ylabel("Northing [m]")
        self.ax.grid(True)
        self.fig.tight_layout()
        self.fig.canvas.draw()

    def save_profile_points(self):
        self.file_path = filedialog.asksaveasfilename(
            defaultextension=".txt", filetypes=[("Text Files", "*.txt")])
        self.file_paths.append(self.file_path)
        if self.file_path:
            with open(self.file_path, "w") as f:
                f.write("X, Y\n")  # Header
                for point in self.current_line:
                    f.write(f"{point[0]}, {point[1]}\n")

    def create_profile(self):

        if len(self.current_line) != 0:
            self.current_line = np.unique(self.current_line, axis=0)
            x_values = np.array(self.current_line)[:, 0]
            y_values = np.array(self.current_line)[:, 1]
        else:
            x_values = self.profile_coords[:, 0]
            y_values = self.profile_coords[:, 1]

        self.model.profiles.append({})
        self.model.profiles[0]['x'] = x_values
        self.model.profiles[0]['y'] = y_values

        self.model.createProfiles(ttem_model_idx=0, profile_idx = [0],
                                  interp_radius=self.interp_radius.get(),
                                  model_spacing=self.model_spacing.get())

    def launch_profile_plot(self):
        profile_window = tk.Toplevel(root)
        profile_window.title("Profile")
        plot = profiler.Plot(self.model)

        fig, ax = plt.subplots()

        # Embed the plot into the tkinter window
        self.canvas = FigureCanvasTkAgg(fig, master=profile_window)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack()

        # Add default buttons
        button_frame = ttk.Frame(profile_window)
        button_frame.pack(fill=tk.BOTH, expand=True)

        label = tk.StringVar()
        label.set("vmin")
        tk.Label(button_frame, textvariable=label).pack(side=tk.LEFT)

        self.vmin = tk.DoubleVar()
        self.vmin.set(1)
        self.vmin_entry = ttk.Entry(
            button_frame, text=self.vmin, width=5)
        self.vmin_entry.pack(side=tk.LEFT, padx=10, pady=5)

        label = tk.StringVar()
        label.set("vmax")
        tk.Label(button_frame, textvariable=label).pack(side=tk.LEFT)

        self.vmax = tk.DoubleVar()
        self.vmax.set(1000)
        self.vmax_entry = ttk.Entry(
            button_frame, text=self.vmax, width=5)
        self.vmax_entry.pack(side=tk.LEFT, padx=10, pady=5)

        cmap_list = ["aarhus", "viridis", "jet", "rainbow", "bone", "parula",
                     "turbo"]
        self.cmap = tk.StringVar()
        self.cmap.set("Select cmap")

        self.cmap_select = tk.OptionMenu(
            button_frame, self.cmap, *cmap_list)
        self.cmap_select.pack(side=tk.LEFT)

        self.var3 = tk.IntVar()
        self.discrete_button = ttk.Checkbutton(
            button_frame, text="Discrete\nColors", variable=self.var3,
            onvalue=1, offvalue=0)
        self.discrete_button.pack(side=tk.LEFT, padx=10, pady=5)

        self.var4 = tk.IntVar()
        self.flip_button = ttk.Checkbutton(
            button_frame, text="Flip\nModel", variable=self.var4,
            onvalue=1, offvalue=0)

        self.flip_button.pack(side=tk.LEFT, padx=10, pady=5)

        zmin = np.min(self.model.profiles[0]['elev']
                     - self.model.profiles[0]['doi'])

        scale = 10
        vmin=self.vmin.get(); vmax=self.vmax.get()

        plot.TEMProfile(profile_idx=0, cmap=getAarhusCols(),
                        cbar_orientation='horizontal',
                        zmin=-90, zmax=10, ax=ax, scale=scale, vmax=vmax)

        if len(self.model.stem_models) !=0:

            for i in range(len(self.model.stem_models)):

                plot.addTEMSoundings(profile_idx=0,  stem_model_idx=i,
                                     search_radius=100, cmap=getAarhusCols(),
                                     ax=ax, print_msg=True, vmax=vmax)

        if len(self.model.boreholes) != 0:
            plot.addBoreholes(profile_idx=0, ax=ax, print_msg=True)

       # Function to refresh the plot
        def refresh_plot():

            ax.clear()

            # Embed the plot into the tkinter window

            if self.cmap.get() == 'aarhus' or self.cmap.get() == 'Select cmap':
                cmap = getAarhusCols()
            if self.cmap.get() == 'viridis':
                cmap = plt.cm.viridis
            if self.cmap.get() == 'jet':
               cmap = plt.cm.jet
            if self.cmap.get() == 'rainbow':
               cmap = plt.cm.rainbow
            if self.cmap.get() == 'bone':
               cmap = plt.cm.bone
            if self.cmap.get() == 'parula':
               cmap = getParulaCols()

            zmin = np.min(self.model.profiles[0]['elev']
                         - self.model.profiles[0]['doi'])

            scale = 10
            vmin=self.vmin.get(); vmax=self.vmax.get()

            plot.TEMProfile(profile_idx=0, cmap=cmap,
                            cbar_orientation='horizontal',
                            zmin=-90, zmax=10, ax=ax, scale=scale, vmax=vmax)

            self.canvas.draw()

        def add_walkTEM():
            if self.cmap.get() == 'aarhus' or self.cmap.get() == 'Select cmap':
                cmap = getAarhusCols()
            if self.cmap.get() == 'viridis':
                cmap = plt.cm.viridis
            if self.cmap.get() == 'jet':
               cmap = plt.cm.jet
            if self.cmap.get() == 'rainbow':
               cmap = plt.cm.rainbow
            if self.cmap.get() == 'bone':
               cmap = plt.cm.bone
            if self.cmap.get() == 'parula':
               cmap = getParulaCols()


            plot.addTEMSoundings(profile_idx=0, model_idx=1, cmap=cmap,
                                 search_radius=100,
                                 ax=ax, print_msg=True, vmax=vmax)

        def add_borehole():
            plot.addBoreholes(model_idx=0, ax=ax, print_msg=True)

        def plot_key():
            key_window = tk.Toplevel(profile_window)
            key_window.title("Key")
            plot = profiler.Plot(self.model)
            fig, ax = plt.subplots()
            canvas = FigureCanvasTkAgg(fig, master=key_window)
            canvas.draw()
            canvas.get_tk_widget().pack()
            plot.lithKey(ax=ax)


        refresh_button = tk.Button(button_frame,
                                   text="Refresh Plot",
                                   command=refresh_plot)
        refresh_button.pack(side=tk.LEFT, padx=10, pady=5)

        add_walkTEM_button = tk.Button(button_frame,
                                   text="Add walkTEM",
                                   command=add_walkTEM)
        add_walkTEM_button.pack(side=tk.LEFT, padx=10, pady=5)

        add_borehole_button = tk.Button(button_frame,
                                   text="Add Borehole",
                                   command=add_borehole)
        add_borehole_button.pack(side=tk.LEFT, padx=10, pady=5)

        plot_key_button = tk.Button(button_frame,
                                   text="Lithological Key",
                                   command=add_borehole)
        plot_key_button.pack(side=tk.LEFT, padx=10, pady=5)

    def calculate_length(self, line):
        if len(self.current_line) > 1:
            x_vals, y_vals = zip(*self.current_line)
            length = np.sum(np.sqrt(np.diff(x_vals)**2 + np.diff(y_vals)**2))
            if self.line_length_text:
                self.line_length_text.remove()
            self.line_length_text = self.ax.text(
                0.05, 0.95, f"Profile Length: {length:.2f} m",
                transform=self.ax.transAxes, verticalalignment='top', c='w')

            self.fig.canvas.draw()

    def toggle_drawing_mode(self):
        self.drawing_mode = not self.drawing_mode
        if not self.drawing_mode:
            self.current_line = []
            self.plot_coordinates()

    def set_snap(self, snap):
        self.snap = snap

    def set_misfit(self, misfit):
        self.misfit = misfit

    def set_flip(self, flip):
        self.flip = flip

    def set_discrete(self, discrete):
        self.discrete = discrete

    def on_canvas_click(self, event):
        if self.drawing_mode:
            if event.button == 1:  # Left-click to add a point
                x_click, y_click = event.xdata, event.ydata
                if self.snap:
                    idx = np.argmin(
                        ((self.x - x_click) ** 2 +
                         (self.y - y_click) ** 2) ** 0.5)

                    x_click = self.x[idx]
                    y_click = self.y[idx]
                self.current_line.append((x_click, y_click))
            elif event.button == 3:  # Right-click to remove the previous point
                if self.current_line:
                    self.current_line.pop()
            self.plot_coordinates()

    def open_tTEM_file(self):

        self.crs = None
        self.tem_path = filedialog.askopenfilename(
            filetypes=[("Text Files", "*.xyz")])

        self.model.loadXYZ(self.tem_path, mod_name='tTEM', model_type='tTEM')

        self.crs = self.model.ttem_models[-1]['epsg']

        self.plot_coordinates()
        self.x = self.model.ttem_models[0]['x']
        self.y = self.model.ttem_models[0]['y']


    def open_walkTEM_file(self):

        self.walktem_path = filedialog.askopenfilename(
            filetypes=[("Text Files", "*.xyz")])

        self.model.loadXYZ(self.walktem_path, mod_name='sTEM',
                           model_type='sTEM')

        self.plot_coordinates()

    def open_borehole_files(self):

        self.borehole_paths = filedialog.askopenfilenames(
            filetypes=[("Text Files", "*.dat")])

        self.model.loadBoreholes(self.borehole_paths)

        self.plot_coordinates()

    def open_shp(self):

        # Select path to shapefile (.shp)
        self.shape_path = filedialog.askopenfilename()

        self.profile_coords = self.model.loadShp(self.shape_path)

        self.plot_coordinates()

    def find_coord_line(self):
        try:
            with open(self.tem_path, 'r') as file:
                lines = file.readlines()
                for i, line in enumerate(lines):
                    if '/COORDINATE SYSTEM' in line or '/EPSG' in line:
                        if i + 1 < len(lines):

                            return lines[i + 1].strip()
                        else:
                            return "No line after the matched line"
        except FileNotFoundError:
            return "File not found"

        return "Text not found in the file"

    def get_coord(self, line):
        match = re.findall(r'\(.*?\)', line)
        if match:
            return match[0][1:-1]
        return "No brackets found"

    def write_py(self):
        print('No good')


if __name__ == "__main__":
    root = tk.Tk()
    app = ProfilerUI(root)
    root.mainloop()
