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

sys.path.append(r'c:\users\au701230\onedrive - aarhus universitet\desktop\pytem\pytem2\profiler\src/profiler')
import profiler
from colormaps import getAarhusCols, getParulaCols



class InteractivePlotApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Profiler")

        self.plotter = InteractivePlot()

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

        button_frame1 = ttk.Frame(self.root)
        button_frame1.pack(fill=tk.BOTH, expand=True)

        self.load_button = ttk.Button(
            button_frame1, text="Add tTEM data", command=self.open_TEM_file)
        self.load_button.pack(side=tk.LEFT, padx=10, pady=5)
        
        self.load_button = ttk.Button(
            button_frame1, text="Add walkTEM Data", command=self.plotter.open_walkTEM_file)
        self.load_button.pack(side=tk.LEFT, padx=10, pady=5)

        self.load_button = ttk.Button(
            button_frame1, text="Add Borehole Data", command=self.plotter.open_borehole_files)
        self.load_button.pack(side=tk.LEFT, padx=10, pady=5)

        #self.load_button = ttk.Button(
        #    button_frame1, text="Export .py",
         #                            command=self.plotter.write_py)

        #self.load_button.pack(side=tk.RIGHT, padx=10, pady=5)
        
        self.var2 = tk.IntVar()
        self.misfit_button = ttk.Checkbutton(button_frame1,
                                             text="Plot Residual",
                                             variable=self.var2, onvalue=1,
                                             offvalue=0,
                                             command=self.misfit_selection)

        self.misfit_button.pack(side=tk.LEFT, padx=10, pady=5)
        
        
        button_frame2 = ttk.Frame(self.root)
        button_frame2.pack(fill=tk.BOTH, expand=True)
        
        map_style = ["aarhus", "viridis", "jet", "rainbow",
                     "bone", "parula", "turbo"]

        self.plotter.cmap = tk.StringVar()
        self.plotter.cmap.set("Select cmap")
        
        self.cmap_select = tk.OptionMenu(
            button_frame2, self.plotter.cmap, *map_style)
        self.cmap_select.pack(side=tk.LEFT)

        label = tk.StringVar()
        label.set("x min")
        tk.Label(button_frame2, textvariable=label).pack(side=tk.LEFT)

        self.plotter.lat_disc = tk.DoubleVar()
        self.plotter.lat_disc.set(None)
        self.xmin_entry = ttk.Entry(
            button_frame2, text=self.plotter.xmin, width=5)
        self.xmin_entry.pack(side=tk.LEFT, padx=10, pady=5)
        
        label = tk.StringVar()
        label.set("x min")
        tk.Label(button_frame2, textvariable=label).pack(side=tk.LEFT)

        self.plotter.lat_disc = tk.DoubleVar()
        self.plotter.lat_disc.set(None)
        self.xmin_entry = ttk.Entry(
            button_frame2, text=self.plotter.xmin, width=5)
        self.xmin_entry.pack(side=tk.LEFT, padx=10, pady=5)
        
        label = tk.StringVar()
        label.set("x min")
        tk.Label(button_frame2, textvariable=label).pack(side=tk.LEFT)

        self.plotter.lat_disc = tk.DoubleVar()
        self.plotter.lat_disc.set(None)
        self.xmin_entry = ttk.Entry(
            button_frame2, text=self.plotter.xmin, width=5)
        self.xmin_entry.pack(side=tk.LEFT, padx=10, pady=5)
        
        label = tk.StringVar()
        label.set("x min")
        tk.Label(button_frame2, textvariable=label).pack(side=tk.LEFT)

        self.plotter.lat_disc = tk.DoubleVar()
        self.plotter.lat_disc.set(None)
        self.xmin_entry = ttk.Entry(
            button_frame2, text=self.plotter.xmin, width=5)
        self.xmin_entry.pack(side=tk.LEFT, padx=10, pady=5)

        self.load_button = ttk.Button(
            button_frame2, text="Set Map Limits", command=self.open_TEM_file)
        self.load_button.pack(side=tk.LEFT, padx=10, pady=5)
        
        
        self.load_button = ttk.Button(
            button_frame2, text="Reset Map Limits", command=self.open_TEM_file)
        self.load_button.pack(side=tk.LEFT, padx=10, pady=5)
        

        button_frame3 = ttk.Frame(self.root)
        button_frame3.pack(fill=tk.BOTH, expand=True)
        
        self.drawing_button = ttk.Button(
            button_frame3, text="Toggle Profile Drawer",
            command=self.toggle_drawing_mode)
        self.drawing_button.pack(side=tk.LEFT, padx=10, pady=5)

        self.var1 = tk.IntVar()
        self.snap_button = ttk.Checkbutton(
            button_frame3, text="Snap", variable=self.var1, onvalue=1,
            offvalue=0, command=self.snap_selection)
        
        self.snap_button.pack(side=tk.LEFT, padx=10, pady=5)
        
        self.var1.set(1)
        
        self.open_shp_button = ttk.Button(
            button_frame3, text="Open Shape File",
            command=self.open_shp)

        self.open_shp_button.pack(side=tk.LEFT, padx=10, pady=5)

        self.save_button = ttk.Button(
            button_frame3, text="Save Profile Coordinates",
            command=self.save_profile_points)

        self.save_button.pack(side=tk.LEFT, padx=10, pady=5)


        button_frame4 = ttk.Frame(self.root)
        button_frame4.pack(fill=tk.BOTH, expand=True)

        label = tk.StringVar()
        label.set("Lateral\nDiscretization")
        tk.Label(button_frame4, textvariable=label).pack(side=tk.LEFT)

        self.plotter.lat_disc = tk.DoubleVar()
        self.plotter.lat_disc.set(25)
        self.lat_disc_entry = ttk.Entry(
            button_frame4, text=self.plotter.lat_disc, width=5)
        self.lat_disc_entry.pack(side=tk.LEFT, padx=10, pady=5)

        label = tk.StringVar()
        label.set("Search\nRadius")
        tk.Label(button_frame4, textvariable=label).pack(side=tk.LEFT)

        self.plotter.interp_dist = tk.DoubleVar()
        self.plotter.interp_dist.set(25)
        self.interp_dist_entry = ttk.Entry(
            button_frame4, text=self.plotter.interp_dist, width=5)
        self.interp_dist_entry.pack(side=tk.LEFT, padx=10, pady=5)

        #label = tk.StringVar()
        #label.set("vmin")
        #tk.Label(button_frame2, textvariable=label).pack(side=tk.LEFT)

        #self.plotter.vmin = tk.DoubleVar()
        #self.plotter.vmin.set(1)
        #self.vmin_entry = ttk.Entry(
        #    button_frame2, text=self.plotter.vmin, width=5)
        #self.vmin_entry.pack(side=tk.LEFT, padx=10, pady=5)

        #label = tk.StringVar()
        #label.set("vmax")
        #tk.Label(button_frame2, textvariable=label).pack(side=tk.LEFT)

        #self.plotter.vmax = tk.DoubleVar()
        #self.plotter.vmax.set(1000)
        #self.vmax_entry = ttk.Entry(
        #    button_frame2, text=self.plotter.vmax, width=5)
        #self.vmax_entry.pack(side=tk.LEFT, padx=10, pady=5)

        #cmap_list = ["aarhus", "viridis", "jet", "rainbow",
        #             "bone", "parula", "turbo"]

        #self.plotter.cmap = tk.StringVar()
        #self.plotter.cmap.set("Select cmap")

        #self.cmap_select = tk.OptionMenu(
        #    button_frame2, self.plotter.cmap, *cmap_list)
        #self.cmap_select.pack(side=tk.LEFT)

        #self.var3 = tk.IntVar()
        #self.discrete_button = ttk.Checkbutton(
        #    button_frame2, text="Discrete\nColors", variable=self.var3,
        #    onvalue=1, offvalue=0, command=self.discrete_selection)

        #self.discrete_button.pack(side=tk.LEFT, padx=10, pady=5)

        #self.var4 = tk.IntVar()
        #self.flip_button = ttk.Checkbutton(
        #    button_frame2, text="Flip\nModel", variable=self.var4,
        #    onvalue=1, offvalue=0, command=self.flip_selection)

        #self.flip_button.pack(side=tk.LEFT, padx=10, pady=5)

        self.create_button = ttk.Button(
            button_frame4, text="Create Profile", command=self.create_profile)
        self.create_button.pack(side=tk.LEFT, padx=10, pady=5)
        
        self.create_button = ttk.Button(
            button_frame4, text="Plot Profile", command=self.create_profile)
        self.create_button.pack(side=tk.LEFT, padx=10, pady=5)


       
    def open_TEM_file(self):
        self.plotter.open_TEM_file()

    def add_TEM_file(self):
        self.plotter.add_TEM_file()

    def toggle_drawing_mode(self):
        self.plotter.toggle_drawing_mode()

    def save_profile_points(self):
        self.plotter.save_profile_points()

    def create_profile(self):
        self.plotter.create_profile()

    def snap_selection(self):
        self.snap = self.var1.get()
        self.plotter.set_snap(self.snap)

    def misfit_selection(self):
        self.misfit = self.var2.get()
        self.plotter.set_misfit(self.misfit)

    def open_shp(self):
        self.plotter.open_shp()

    def discrete_selection(self):
        self.discrete = self.var3.get()
        self.plotter.set_discrete(self.discrete)

    def flip_selection(self):
        self.flip = self.var4.get()
        self.plotter.set_flip(self.flip)


class InteractivePlot:
    def __init__(self):

        self.fig, self.ax = plt.subplots(dpi=100, figsize=(8, 6))
        
        self.xmin=None
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


    def plot_coordinates(self):
        self.ax.clear()

        if self.current_line:
            x_vals, y_vals = zip(*self.current_line)
            line, = self.ax.plot(x_vals, y_vals, color='r', marker='x')
            self.calculate_length(line)

        for tem_model in self.model.tem_models:

            if tem_model['mod_name'] == 'tTEM':
                s = 0.5; c = 'cyan'

            if tem_model['mod_name'] == 'walkTEM':
                s = 8; c = 'blue'

            if self.misfit:
                sc = self.ax.scatter(tem_model['x'],
                                     tem_model['y'],
                                     c=tem_model['residual'],
                                     s=s, vmin=0, vmax=3, cmap='RdYlGn_r')
    
                if len(plt.gcf().axes) < 2:
                    self.cb = plt.colorbar(sc).set_label('Residual')

            else:
                self.ax.scatter(tem_model['x'],
                                tem_model['y'],
                                c=c, s=s)

        if self.profile_coords is not None:
            x_vals = self.profile_coords[:,0]
            y_vals = self.profile_coords[:,1]
            line, = self.ax.plot(x_vals, y_vals, color='r', marker='x')
            self.calculate_length(line)

        if len(self.model.boreholes) != 0:
            for bh in self.model.boreholes:
                self.ax.scatter(bh['x'],
                                bh['y'], c='orange', s=7)

       # cx.add_basemap(
        #    self.ax, crs=self.crs, source=cx.providers.Esri.WorldImagery,
        #    attribution_size=4)

        self.ax.set_xlabel("Easting [m]")
        self.ax.set_ylabel("Northing [m]")
        self.ax.grid(True)
        self.fig.tight_layout()
        self.fig.canvas.draw()

    def save_profile_points(self):
        #self.current_line = list(set(map(tuple, self.current_line)))
        self.file_path = filedialog.asksaveasfilename(
            defaultextension=".txt", filetypes=[("Text Files", "*.txt")])
        self.file_paths.append(self.file_path)
        if self.file_path:
            with open(self.file_path, "w") as f:
                f.write("X, Y\n")  # Header
                for point in self.current_line:
                    f.write(f"{point[0]}, {point[1]}\n")

    def create_profile(self):

        #self.current_line = list(set(map(tuple, self.current_line)))

        if len(self.current_line) != 0:
            self.current_line = np.unique(self.current_line, axis=0)
            x_values = np.array(self.current_line)[:, 0]
            y_values = np.array(self.current_line)[:, 1]
        else:
            x_values = self.profile_coords[:,0]
            y_values = self.profile_coords[:,1]
            
        self.model.profiles.append({})
        self.model.profiles[0]['x'] = x_values
        self.model.profiles[0]['y'] = y_values
        print(y_values)

        interp_radius = self.interp_dist.get()

        self.model.createProfiles(model_idx=0, profile_idx = [0],
                                  interp_radius=interp_radius,
                                  model_spacing=self.lat_disc.get())
        
    def plot_profile(self):

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

        fig, ax = plt.subplots(1, 1)

        zmin = np.min(elev_new - doi_new)

        pt.plot_model2D(rhos_new, depths_new, elev_new, dists,
                        doi_new, zmin=zmin, ax=ax, vmin=self.vmin.get(),
                        vmax=self.vmax.get(), log=True, flip=self.flip,
                        cmap=cmap, discrete_colors=self.discrete)

        if self.walkTEM_list is not None:
            pt.add_walkTEM(
                self.walkTEM_list, xi, yi, dists,  ax=ax, elev=elev_new,
                walkTEM_width=dists[-1]/75, cmap=cmap, search_radius=100,
                vmin=self.vmin.get(), vmax=self.vmax.get())

        if self.bh_list is not None:
            pt.add_boreholes(
                self.bh_list, xi, yi, dists, ax=ax, elev=elev_new,
                search_radius=dists[-1]/10, bh_width=dists[-1]/20)

        ax.set_aspect('auto')
        fig.tight_layout()

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

    def open_TEM_file(self):

        self.crs = None
        self.tem_path = filedialog.askopenfilename(
            filetypes=[("Text Files", "*.xyz")])

        self.model.loadXYZ(self.tem_path, mod_name='tTEM')

        self.crs = self.model.tem_models[-1]['epsg']

        self.plot_coordinates()
        
        self.x = self.model.tem_models[0]['x']
        self.y = self.model.tem_models[0]['y']


    def add_TEM_file(self):
        # add function to merge
        self.tem_paths = [self.tem_path]
        self.tem_paths.append(filedialog.askopenfilename(filetypes=[("Text Files", "*.xyz")]))
        self.model.loadXYZ(self.tem_paths[-1])
        if self.crs != self.model.tem_models[-1]['epsg']:

            print('Wrong CRS')

        self.plot_coordinates()


    def open_walkTEM_file(self):

        self.walktem_path = filedialog.askopenfilename(
            filetypes=[("Text Files", "*.xyz")])

        self.model.loadXYZ(self.walktem_path, mod_name='walkTEM')

        self.plot_coordinates()

    def open_borehole_files(self):

        self.borehole_paths = filedialog.askopenfilenames(
            filetypes=[("Text Files", "*.dat")])

        self.model.loadBoreholes(self.borehole_paths)

        self.plot_coordinates()

    def open_shp(self):

        # Select path to shapefile (.shp)
        self.shape_path = filedialog.askopenfilename()

        self.profile_coords = pt.load_shp_file(self.shape_path)

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
        
        print('')



if __name__ == "__main__":
    root = tk.Tk()
    app = InteractivePlotApp(root)
    root.mainloop()
