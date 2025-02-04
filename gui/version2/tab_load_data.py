#%%
import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import numpy as np
import contextily as cx
import pickle
import threading
import sys
from matplotlib.widgets import RectangleSelector

sys.path.append(r"C:\Users\pamcl\OneDrive - Danmarks Tekniske Universitet\Dokumenter\Projects\Python\profiler")

import model_handler

padx = 2
pady = 2

class LoadDataTab:
    def __init__(self, parent_frame):
        self.profile_coords = [] # move to model
        self.parent = parent_frame
        self.drawing_mode = False
        self.current_line = []
        self.xmin, self.xmax, self.ymin, self.ymax = None, None, None, None
        self.model = model_handler.Model()

        ctk.set_appearance_mode("dark")  # Options: "System", "Light", "Dark"
        ctk.set_default_color_theme("blue")

        self.setup_ui()

    def setup_ui(self):
        """ Set up UI elements """
        project_frame = ctk.CTkFrame(self.parent)
        project_frame.pack(fill=tk.BOTH, expand=False, padx=padx, pady=pady)
        ctk.CTkButton(project_frame, text="Create Project", 
                      command=self.create_object).pack(side=tk.LEFT, padx=padx, pady=pady)
        ctk.CTkButton(project_frame, text="Load Project", 
                      command=self.load_object).pack(side=tk.LEFT, padx=padx, pady=pady)
        self.create_plot_area()
        self.create_controls()

    def create_plot_area(self):
        """ Embeds Matplotlib into the Tkinter frame """
        
        self.fig, self.ax = plt.subplots(dpi=100, figsize=(6, 6))
        
        self.canvas_frame = ctk.CTkFrame(self.parent)
        self.canvas_frame.pack(fill=tk.BOTH, expand=False, padx=padx, pady=pady)

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.canvas_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=False)

        self.fig.canvas.mpl_connect('button_press_event', self.on_canvas_click)

        #self.toolbar = NavigationToolbar2Tk(self.canvas, self.canvas_frame)
        #self.toolbar.update()

        # Add zoom selection tool
        self.rect_selector = RectangleSelector(
            self.ax, self.on_zoom_select,
            useblit=True, interactive=True
        )

         # Customize rectangle appearance AFTER creation
        self.rect_selector.artists[0].set_edgecolor('red')   # Border color
        self.rect_selector.artists[0].set_facecolor('None')   # Border color
        self.rect_selector.artists[0].set_alpha(1)          # Transparency
        self.rect_selector.artists[0].set_linestyle('-')     # Dashed border
        self.rect_selector.artists[0].set_linewidth(2)        # Border thickness

        self.rect_selector.set_active(False)  # Start disabled

        # Ensure proper event handling for zooming
        self.canvas.get_tk_widget().bind("<ButtonRelease-1>", lambda event: self.apply_zoom())

    def on_zoom_select(self, eclick, erelease):
        """ Stores the coordinates of the zoom box when selection is complete """
        if eclick.xdata is None or erelease.xdata is None:
            return  # Ignore clicks outside plot area

        self.xmin, self.xmax = sorted([eclick.xdata, erelease.xdata])
        self.ymin, self.ymax = sorted([eclick.ydata, erelease.ydata])

        # Keep track that selection has started
        self.zoom_selection_active = True

    def finalize_zoom(self, event):
        """ Applies zoom on the second click """
        if self.zoom_selection_active:
            self.apply_zoom()
            self.zoom_selection_active = False  # Reset for the next selection

    def apply_zoom(self):
        """ Applies zoom limits to the plot and disables the selector after zooming """
        if None not in [self.xmin, self.xmax, self.ymin, self.ymax]:
            self.ax.set_xlim(self.xmin, self.xmax)
            self.ax.set_ylim(self.ymin, self.ymax)
            self.canvas.draw_idle()

            # Disable the selector after applying zoom
            self.rect_selector.set_active(False)
            self.toggle_button.configure(text="Enable Rectangle Zoom")

    def toggle_rectangle_selector(self):
        """ Toggles the rectangle drawer on/off """
        if self.rect_selector.active:
            self.rect_selector.set_active(False)
            self.toggle_button.configure(text="Enable Rectangle Zoom")
        else:
            self.rect_selector.set_active(True)
            self.toggle_button.configure(text="Disable Rectangle Zoom")

    def toggle_drawing_mode(self):
        """ Toggles the profile drawing mode """
        if self.drawing_mode == 1:
            self.drawing_mode = 0  # Disable drawing
            self.draw_toggle_button.configure(text="Draw Profile Coordinates")
            if len(self.current_line) != 0:
                self.profile_coords.append(np.array(self.current_line))
                self.current_line = []
                self.update_map()  # Refresh map display
        else:
            self.drawing_mode = 1  # Enable drawing
            self.draw_toggle_button.configure(text="Save Profile Coordinates")
            self.current_line = []  # Reset any drawn lines
            self.update_map()  # Refresh map display
            
    def create_controls(self):
        """ Creates UI controls for file handling and map settings """

        # Frame for Map Settings
        map_frame = ctk.CTkFrame(self.parent)
        map_frame.pack(fill=tk.BOTH, expand=False, padx=padx, pady=pady)

        # Dropdown for background maps
        map_styles = ["None", "Aerial Imagery", "OSM"]
        self.bg_map = ctk.StringVar(value="Map Background")
        self.bg_map_select = ctk.CTkOptionMenu(map_frame, values=map_styles, 
                                               variable=self.bg_map)
        self.bg_map_select.pack(side=tk.LEFT, padx=padx, pady=pady)

        # X Limits
        self.xmin_entry = ctk.CTkEntry(map_frame, 
                                       placeholder_text="X Min", width=100)
        self.xmin_entry.pack(side=tk.LEFT, padx=padx, pady=pady)

        self.xmax_entry = ctk.CTkEntry(map_frame, 
                                       placeholder_text="X Max", width=100)
        self.xmax_entry.pack(side=tk.LEFT, padx=padx, pady=pady)

        # Y Limits
        self.ymin_entry = ctk.CTkEntry(map_frame, 
                                       placeholder_text="Y Min", width=100)
        self.ymin_entry.pack(side=tk.LEFT, padx=padx, pady=pady)

        self.ymax_entry = ctk.CTkEntry(map_frame, 
                                       placeholder_text="Y Max", width=100)
        self.ymax_entry.pack(side=tk.LEFT, padx=padx, pady=pady)

        self.zoom_toggle_button = ctk.CTkButton(map_frame, 
                                           text="Enable Rectangle Zoom",
                                           command=self.toggle_rectangle_selector)
        self.zoom_toggle_button.pack(side=tk.LEFT, padx=padx, pady=pady)

        update_map = ctk.CTkButton(map_frame, text="Update Map", 
                                   command=self.set_map_limits)
        update_map.pack(side=tk.LEFT, padx=padx, pady=pady)
        
        reset_map = ctk.CTkButton(map_frame, text="Reset Limits",
                                  command=self.reset_map_limits)
        reset_map.pack(side=tk.LEFT, padx=padx, pady=pady)

        # Frame for Map & Data Controls
        button_frame = ctk.CTkFrame(self.parent)
        button_frame.pack(fill=tk.BOTH, expand=False, padx=padx, pady=pady)

        # Buttons for adding data
        add_ttem = ctk.CTkButton(button_frame, text="Add tTEM Models", 
                                 command=lambda: self.open_file("tTEM"))
        add_ttem.pack(side=tk.LEFT, padx=padx)
        add_stem = ctk.CTkButton(button_frame, text="Add sTEM Models", 
                                 command=lambda: self.open_file("sTEM"))
        add_stem.pack(side=tk.LEFT, padx=padx)
        add_bh = ctk.CTkButton(button_frame, text="Add Borehole Data", 
                               command=lambda: self.open_file("borehole"))
        add_bh.pack(side=tk.LEFT, padx=padx)
        
        load_shp = ctk.CTkButton(button_frame, text="Load Shp File", 
                                 command=print('load shape'))
        load_shp.pack(side=tk.LEFT, padx=padx)

        # Toggle Drawing Mode
        self.draw_toggle_button = ctk.CTkButton(button_frame, 
                                                text="Draw Profile Coordinates", 
                                                command=self.toggle_drawing_mode)
        self.draw_toggle_button.pack(side=tk.LEFT, padx=padx)

        # Snap Mode Checkbox
        self.snap = tk.BooleanVar(value=False)
        snap_checkbox = ctk.CTkCheckBox(button_frame, 
                                        text="Snap to Closet Data Point", 
                                        variable=self.snap)
        snap_checkbox.pack(side=tk.LEFT, padx=padx)

        prof_frame = ctk.CTkFrame(self.parent)
        prof_frame.pack(fill=tk.BOTH, expand=False, padx=padx, pady=pady)

        prof_methods = ["Line Number", "Shp File", "Map Selection"]
        self.prof_method = ctk.StringVar(value="Select Method")
        self.cmap_select = ctk.CTkOptionMenu(prof_frame, values=prof_methods, variable=self.prof_method)
        self.cmap_select.pack(side=tk.LEFT, padx=padx, pady=pady)

        self.int_rad_entry = ctk.CTkEntry(prof_frame,
                                          placeholder_text="Interpolation Radius")
        self.int_rad_entry.pack(side=tk.LEFT, padx=padx, pady=pady)

        self.lat_dis_entry = ctk.CTkEntry(prof_frame,
                                          placeholder_text="Lateral Discretization")
        self.lat_dis_entry.pack(side=tk.LEFT, padx=padx, pady=pady)

        create_profiles = ctk.CTkButton(prof_frame, text="Create Profiles", 
                                        command=lambda: threading.Thread(target=self.create_profile, daemon=True).start())
        create_profiles.pack(side=tk.LEFT, padx=padx, pady=pady)

        self.buttons = [self.bg_map_select, self.xmin_entry, self.xmax_entry, 
                        self.ymin_entry, self.ymax_entry, self.zoom_toggle_button, 
                        update_map, reset_map, add_ttem, add_stem, add_bh, load_shp,
                        self.draw_toggle_button, self.snap, self.prof_method, self.cmap_select,
                        create_profiles]
        
        for button in self.buttons:
            if isinstance(button, ctk.CTkButton) or isinstance(button, ctk.CTkEntry):
                button.configure(state="disabled") 
    
    def enable_buttons(self):
        for button in self.buttons:
            if isinstance(button, ctk.CTkButton) or isinstance(button, ctk.CTkEntry):
                button.configure(state="normal") 

    def create_profile(self):
        if self.prof_method.get() == "Line Number":
            lines = np.unique(self.model.ttem_models['line_num'])

        else:
            lines = self.profile_coords

        # Ensure the popup window appears immediately
        progress_window = ctk.CTkToplevel(self.parent)
        progress_window.title("Creating Profiles")
        progress_window.geometry("300x100")
        progress_window.grab_set()  # Makes the popup modal (blocks interaction with the main window)

        # Progress label
        progress_label = ctk.CTkLabel(progress_window, text="Progress...")
        progress_label.pack(pady=pady)

        # Progress bar
        progress_bar = ctk.CTkProgressBar(progress_window)
        progress_bar.pack(pady=pady, padx=pady)
        progress_bar.set(0)  # Initialize at 0%

        # Force the popup to render immediately
        progress_window.update()

        self.model.profiles = []  # Ensure profiles list is cleared

        for line_idx, line in enumerate(lines):

            if self.prof_method.get() == "Line Number":

                idx = np.where(self.model.ttem_models['line_num'] == line)

                # Extract x and y coordinates
                x = self.model.ttem_models['x'][idx[0]]
                y = self.model.ttem_models['y'][idx[0]]

            else:

                x = self.profile_coords[line_idx][:,0]
                y = self.profile_coords[line_idx][:,1]

            # Compute cumulative distance along the path
            dists = np.concatenate(([0], np.cumsum((np.diff(x) ** 2 + np.diff(y) ** 2) ** 0.5)))

            # Define new equally spaced points
            new_distance = np.arange(0, dists[-1], float(self.lat_dis_entry.get()))

            # Interpolate x and y at the new distances
            new_x = np.interp(new_distance, dists, x)
            new_y = np.interp(new_distance, dists, y)

            if dists[-1] > 50.:
                profile = {'x': new_x, 'y': new_y}
                self.model.profiles.append(profile)

            # Processing profiles
            total_profiles = len(self.model.profiles)

            for i, profile in enumerate(self.model.profiles):
                self.model.createProfiles(interp_radius=float(self.int_rad_entry.get()),
                                          profile_idx=[i],
                                          model_spacing=False)

                # Update progress bar
                value = (i + 1) / total_profiles

                self.model.profiles[i]['geo_features'] = {}

                self.parent.after(10, lambda v=value: progress_bar.set(v))

                print(f"\rProgress: {(i+1)/total_profiles*100:.2f}%", end='', flush=True)

            print('\nProfiles created, saving object.')

            self.save_object()

            # Close the popup after completion
            progress_window.destroy()


    def open_file(self, file_type):
        """ Opens a file for tTEM, walkTEM, or borehole data """
        file_types = {
            "tTEM": ("tTEM Data Files", "*.xyz"),
            "sTEM": ("sTEM Data Files", "*.xyz"),
            "borehole": ("Borehole Data Files", "*.dat *.fbd")
        }
        
        if file_type == 'tTEM':
            file_path = filedialog.askopenfilename(filetypes=[file_types[file_type]])
            self.model.loadXYZ(file_path, mod_name='tTEM', model_type='tTEM')
            self.crs = self.model.ttem_models['epsg']

            self.update_map()
            self.x = self.model.ttem_models['x']
            self.y = self.model.ttem_models['y']

            self.save_object()

        if file_type == 'sTEM':
            file_path = filedialog.askopenfilename(filetypes=[file_types[file_type]])

            self.model.loadXYZ(file_path, mod_name='sTEM', model_type='sTEM')

            self.update_map()

            self.save_object()

        if file_type == 'borehole':
            file_paths = filedialog.askopenfilenames(filetypes=[file_types[file_type]])

            if isinstance(file_paths, (list, tuple)):
                if len(file_paths) == 1:
                    print("A single file was selected.")
                    if file_paths.lower().endswith(".fbd"):
                        print("Processing FBD file...")
                elif len(file_paths) > 1:
                    print("Multiple files were selected.")
                    self.model.loadBoreholes(file_paths)
            
            self.update_map()

            self.save_object()

    def set_map_limits(self):
        """ Sets map limits based on user input """
        try:
            self.xmin = float(self.xmin_entry.get()) if self.xmin_entry.get() else None
            self.xmax = float(self.xmax_entry.get()) if self.xmax_entry.get() else None
            self.ymin = float(self.ymin_entry.get()) if self.ymin_entry.get() else None
            self.ymax = float(self.ymax_entry.get()) if self.ymax_entry.get() else None
            self.update_map()
        except ValueError:
            print("Invalid input! Please enter numeric values.")

    def reset_map_limits(self):
        """ Resets map limits to default """
        self.xmin, self.xmax, self.ymin, self.ymax = None, None, None, None
        self.update_map()

    def update_map(self):
        """ Clears and redraws the plot with new limits """
        self.ax.clear()
        if self.current_line:
            x_vals, y_vals = zip(*self.current_line)
            self.ax.plot(x_vals, y_vals, color='r', marker='x')

        if self.model.ttem_models: 
            self.ax.scatter(self.model.ttem_models['x'],
                            self.model.ttem_models['y'],
                            c='cyan', s=0.5, label='tTEM')
            
        if self.model.stem_models: 
            x_coords = [stem_model['x'] for stem_model in self.model.stem_models]
            y_coords = [stem_model['y'] for stem_model in self.model.stem_models]
            self.ax.scatter(x_coords, y_coords, c='k', marker='*', s=20, label='sTEM')
                 
        if self.model.boreholes:
            x_coords = [bh['x'] for bh in self.model.boreholes]
            y_coords = [bh['y'] for bh in self.model.boreholes]
            self.ax.scatter(x_coords, y_coords, c='orange', marker='*', s=20, label='Boreholes')
                
        if len(self.profile_coords) != 0:
            for i, profile_coord in enumerate(self.profile_coords):
                self.ax.plot(profile_coord[:,0], profile_coord[:,1], 
                             label = f"P{i+1}")
            
        if self.bg_map.get() != 'None':
            if self.bg_map.get() == 'OSM':
                cx.add_basemap(
                   self.ax, crs=self.crs,
                   source=cx.providers.OpenStreetMap.Mapnik,
                   attribution=False)

            if self.bg_map.get() == 'Aerial Imagery':
                cx.add_basemap(
                    self.ax, crs=self.crs,
                    source=cx.providers.Esri.WorldImagery,
                    attribution=False)
                
        self.ax.set_aspect(1)
        self.ax.set_xlim(self.xmin, self.xmax)
        self.ax.set_ylim(self.ymin, self.ymax)
        self.ax.set_xlabel("Easting [m]")
        self.ax.set_ylabel("Northing [m]")
        self.ax.grid(True)
        self.ax.legend()
        self.fig.canvas.draw()

    def on_canvas_click(self, event):
        """Handles drawing mode for adding/removing points with optional snapping."""
        if not self.drawing_mode or event.xdata is None or event.ydata is None:
            return
        
        if event.button == 1:  # Left-click to add a point
            x, y = event.xdata, event.ydata
            if self.snap.get():
                idx = np.argmin(np.hypot(self.x - x, self.y - y))
                x, y = self.x[idx], self.y[idx]
            self.current_line.append((x, y))

        elif event.button == 3 and self.current_line:  # Right-click to remove last point
            self.current_line.pop()

        self.update_map()

    def create_object(self):
        """ Opens a file dialog to save an object as a .pkl file """
        self.pkl_path = filedialog.asksaveasfilename(
            defaultextension=".pkl",
            filetypes=[("Pickle Files", "*.pkl"), ("All Files", "*.*")],
            title="Save Pickle File"
        )
        if self.pkl_path:  # Ensure the user didn't cancel
            with open(self.pkl_path, "wb") as f:
                pickle.dump(self.model, f)
            print(f"Object saved to {self.pkl_path}")

        self.enable_buttons()

    def save_object(self):
            """ Opens a file dialog to save an object as a .pkl file """
            if self.pkl_path:  
                with open(self.pkl_path, "wb") as f:
                    pickle.dump(self.model, f)
                print(f"Object saved to {self.pkl_path}")

    def load_object(self):
        """ Opens a file dialog to load a .pkl file and return the object """
        self.pkl_path = filedialog.askopenfilename(
            filetypes=[("Pickle Files", "*.pkl"), ("All Files", "*.*")],
            title="Load Pickle File"
        )
        if self.pkl_path:  # Ensure the user selected a file
            with open(self.pkl_path, "rb") as f:
                self.model = pickle.load(f)
            print(f"Object loaded from {self.pkl_path}")
        
        self.crs = self.model.ttem_models['epsg']

        self.x = self.model.ttem_models['x']
        self.y = self.model.ttem_models['y']
        
        self.update_map()
        self.enable_buttons()
        

if __name__ == "__main__":
    root = ctk.CTk()
    root.title("Load Data Tab")
    root.geometry("1000x700")
    LoadDataTab(root)
    root.mainloop()
