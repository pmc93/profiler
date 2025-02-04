#%%

import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import numpy as np
import pickle

import sys

padx=2
pady=2

sys.path.append(r"C:\Users\pamcl\OneDrive - Danmarks Tekniske Universitet\Dokumenter\Projects\Python\profiler")
import model_handler

class SectionPlotTab:
    """ UI for creating, selecting, and annotating sections with color scale & aspect ratio options. """

    def __init__(self, parent_frame):
        self.parent = parent_frame
        self.current_line = []
        self.drawing_mode = False
        self.profile_idx = 0
        
        
        self.color_scales = ["viridis", "jet", "rainbow", "bone"]#, "parula"]
        self.annotations = []  # Store annotations
        pkl_path = r"c:\Users\pamcl\OneDrive - Danmarks Tekniske Universitet\Dokumenter\Projects\Python\Pytem0\profiler\data\tem\out1.pkl"

        with open(pkl_path, "rb") as f:
                self.model = pickle.load(f)
    
        self.plot = model_handler.Plot(self.model)
        #self.sections = np.unique(self.model.ttem_models[0]['line_num']).tolist()
        n = len(self.model.profiles)
        self.profiles = [str(i) for i in range(1, n + 1)]
        
        # Create two independent Matplotlib figures
        self.fig1, self.ax1 = plt.figure(dpi=100, figsize=(4, 4)), plt.axes()
        self.fig2, self.ax2 = plt.figure(dpi=100, figsize=(8, 4)), plt.axes()

        self.fig2.canvas.mpl_connect("button_press_event", self.on_canvas_click)

        self.setup_ui()

    def setup_ui(self):
        """ Sets up UI elements for section selection, color scale, aspect ratio, and annotation. """
        self.create_plot_area()
        self.create_controls()
        self.update_plots()

    def create_plot_area(self):
        """ Embeds two independent Matplotlib figures into the Tkinter frame """
        self.canvas_frame = ctk.CTkFrame(self.parent)
        self.canvas_frame.pack(fill=tk.BOTH, expand=False, 
                               padx=padx, pady=pady)

        left_frame = ctk.CTkFrame(self.canvas_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Frame for the second canvas and its toolbar (right side)
        right_frame = ctk.CTkFrame(self.canvas_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # First independent plot
        self.canvas1 = FigureCanvasTkAgg(self.fig1, master=left_frame)
       
        # Second independent plot
        self.canvas2 = FigureCanvasTkAgg(self.fig2, master=right_frame)
        
        # Add toolbars for both figures
        #self.toolbar1 = NavigationToolbar2Tk(self.canvas1, left_frame)
        #self.toolbar1.update()
        #self.toolbar2 = NavigationToolbar2Tk(self.canvas2, right_frame)
        #self.toolbar2.update()

        self.canvas1.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.canvas2.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)

    def create_controls(self):
        """ Creates UI controls for section selection, color scale, aspect ratio, and annotations. """

        controls_frame = ctk.CTkFrame(self.parent)
        controls_frame.pack(fill=tk.BOTH, expand=False, padx=padx, pady=pady)

        # Section Selection
        ctk.CTkLabel(controls_frame, text="Select Profile:").pack(side=tk.LEFT, padx=5)
        

        first_button = ctk.CTkButton(controls_frame, text="<<", command=self.first_profile, width=10)
        first_button.pack(side=tk.LEFT, padx=padx)

        prev_button = ctk.CTkButton(controls_frame, text="<", command=self.previous_profile, width=10)
        prev_button.pack(side=tk.LEFT, padx=padx)

        self.section_var = tk.StringVar(value=self.profiles[0])
        self.section_menu = ctk.CTkOptionMenu(controls_frame,
                                              values=self.profiles,
                                              variable=self.section_var,
                                              width=60,
                                              command=lambda _: self.update_profile_idx())
        self.section_menu.pack(side=tk.LEFT, padx=padx)

        next_button = ctk.CTkButton(controls_frame, text=">", command=self.next_profile, width=10)
        next_button.pack(side=tk.LEFT, padx=padx)

        last_button = ctk.CTkButton(controls_frame, text=">>", command=self.last_profile, width=10)
        last_button.pack(side=tk.LEFT, padx=padx)

        ctk.CTkButton(controls_frame, text="Update Plot", command=self.update_plots).pack(side=tk.LEFT, padx=padx)

        self.auto_update_var = tk.BooleanVar(value=True) 
        self.auto_update_checkbox = ctk.CTkCheckBox(controls_frame, 
                                                    text="Auto Update", variable=self.auto_update_var)

        self.auto_update_checkbox.pack(side=tk.LEFT, padx=padx)

        controls_frame2 = ctk.CTkFrame(self.parent)
        controls_frame2.pack(fill=tk.BOTH, expand=False, padx=padx, pady=pady)
                
        # Color Scale Selection # Entry boxes for vmin, vmax, and scale

        self.color_var = tk.StringVar(value="Color Map")
        self.color_menu = ctk.CTkOptionMenu(controls_frame2, 
                                            values=self.color_scales, 
                                            variable=self.color_var, width=100)
        self.color_menu.pack(side=tk.LEFT, padx=2)

        self.vmin_entry = ctk.CTkEntry(controls_frame2, width=120, 
                                       placeholder_text="Min Rho [Ohm.m]")
        self.vmin_entry.pack(side=tk.LEFT, padx=padx)
        self.vmax_entry = ctk.CTkEntry(controls_frame2, width=120,
                                       placeholder_text="Max Rho [Ohm.m]")
        self.vmax_entry.pack(side=tk.LEFT, padx=padx)

        self.zmin_entry = ctk.CTkEntry(controls_frame2, width=80, 
                                       placeholder_text="Min Elev.")
        #self.zmin_entry.insert(0, "-70")  # Default value
        self.zmin_entry.pack(side=tk.LEFT, padx=padx)
        self.zmax_entry = ctk.CTkEntry(controls_frame2, width=80,
                                       placeholder_text="Max Elev.")
        self.zmax_entry.pack(side=tk.LEFT, padx=padx)

        self.scale_entry = ctk.CTkEntry(controls_frame2, width=120,
                                        placeholder_text="Vert. Exaggeration")
        #self.scale_entry.insert(0, "10")  # Default value
        self.scale_entry.pack(side=tk.LEFT, padx=padx)

        # Buttons
        controls_frame3 = ctk.CTkFrame(self.parent)
        controls_frame3.pack(fill=tk.BOTH, expand=False, padx=padx, pady=pady)
        
        self.draw_toggle_button = ctk.CTkButton(controls_frame3, text="Draw Feature", command=self.toggle_drawing_mode)
        self.draw_toggle_button.pack(side=tk.LEFT, padx=padx)

        ctk.CTkButton(controls_frame3, text="Add Iso-Contour", command=lambda: self.open_boundary_popup(method='iso')).pack(side=tk.LEFT, padx=5)
        #ctk.CTkButton(controls_frame3, text="Add Iso-Contour", command=self.add_iso_contour).pack(side=tk.LEFT, padx=5)
        ctk.CTkButton(controls_frame3, text="Add Gradient Feature", 
                      command=lambda: self.open_boundary_popup(method='gradient')).pack(side=tk.LEFT, padx=padx)
        ctk.CTkButton(controls_frame3, text="Copy Previous Feature Definitions", 
                      command=lambda: self.open_boundary_popup(method='gradient')).pack(side=tk.LEFT, padx=padx)

        self.create_geo_feature_editor()

    def create_geo_feature_editor(self, selected_feature=None):
        """
        Creates an interface within the main window to edit geological features, including:
        - Dropdown menu to select a geological feature.
        - Checkbox to toggle editing.
        - Button to delete the selected feature.
        """

        parent_frame = self.parent

        # Create or reuse the geo_frame
        if not hasattr(self, 'geo_frame'):
            self.geo_frame = ctk.CTkFrame(parent_frame)

        geo_frame = self.geo_frame

        for widget in geo_frame.winfo_children():
            widget.destroy()
        geo_frame.pack(fill=tk.X, padx=padx, pady=pady)

        # Check if there are any geological features
        features_exist = len(self.model.profiles[self.profile_idx]["geo_features"]) > 0

        # Prepare dropdown options and initial values
        if features_exist:
            feature_names = list(self.model.profiles[self.profile_idx]["geo_features"].keys())
            feature_liths = [feature["lith"] for feature in self.model.profiles[self.profile_idx]["geo_features"].values()]
            feature_types = [feature["type"] for feature in self.model.profiles[self.profile_idx]["geo_features"].values()]
        else:
            feature_names = ["No Features"]
            feature_liths = [""]
            feature_types = [""]

        if selected_feature is not None:
            feature_idx = feature_names.index(selected_feature)
            print(feature_idx)
        else:
            feature_idx = 0

        # Variables to hold user input
        selected_feature_var = ctk.StringVar(value=feature_names[feature_idx])
        lithology_var = ctk.StringVar(value=feature_liths[feature_idx] if features_exist else "Lithology")
        feature_type_var = ctk.StringVar(value=feature_types[feature_idx] if features_exist else "Feature Type:")

        # Dropdown for feature selection
        feature_dropdown = ctk.CTkOptionMenu(geo_frame, 
                                             variable=selected_feature_var, 
                                             values=feature_names, 
                                             width=120,
                                             command=self.create_geo_feature_editor)
        feature_dropdown.grid(row=0, column=1, padx=padx, pady=pady, sticky="e")
        # Checkbox to toggle editing mode
        edit_mode_var = tk.BooleanVar(value=False)
        edit_checkbox = ctk.CTkCheckBox(
            geo_frame, text="Edit Feature Points", variable=edit_mode_var, 
            command=lambda: print(f"Editing {selected_feature_var.get()}")
        )
        edit_checkbox.grid(row=0, column=2, padx=padx, pady=pady, sticky="w")

        # Lithology text field
        lithology_entry = ctk.CTkEntry(geo_frame, 
                                       textvariable=lithology_var,
                                       placeholder_text='Lithology', width=80)
        lithology_entry.grid(row=0, column=4, padx=5, pady=5, sticky="e")

        # Feature type dropdown
        feature_type_menu = ctk.CTkOptionMenu(geo_frame, 
                                              variable=feature_type_var, 
                                              values=["Top", "Base", "Lens"],
                                              width=80)
        
        feature_type_menu.grid(row=0, column=6, 
                               padx=padx, pady=pady, sticky="e")

        # Delete button
        def delete_feature():
            if not features_exist:
                print("No features to delete.")
                return
            selected_feature = selected_feature_var.get()
            print(selected_feature)
            del self.model.profiles[self.profile_idx]["geo_features"][selected_feature]
            print(f"{selected_feature} deleted.")
            self.create_geo_feature_editor()  # Refresh UI after deletion

        delete_button = ctk.CTkButton(geo_frame, text="Delete Feature", 
                                      command=delete_feature, width=80)
        delete_button.grid(row=0, column=7, padx=padx, pady=pady, sticky="e")

        # Save changes button
        def save_changes():
            if not features_exist:
                print("No features to save.")
                return
            self.model.profiles[self.profile_idx]["geo_features"][selected_feature]["lith"] = lithology_var.get()
            self.model.profiles[self.profile_idx]["geo_features"][selected_feature]["type"] = feature_type_var.get()
            print(f"{selected_feature} updated.")
            print("All Geological Features:")
            for i, feature in enumerate(self.model.profiles[self.profile_idx]["geo_features"].values(), start=1):
                print(f"Feature {i}: {feature}")

        save_button = ctk.CTkButton(geo_frame, text="Save Changes", 
                                    command=save_changes, width=80)
        save_button.grid(row=0, column=8, padx=padx, pady=pady, sticky="e")

        # Toggle editing functionality
        def toggle_editing():
            state = "normal" if edit_mode_var.get() else "disabled"

        edit_checkbox.configure(command=toggle_editing)

    def next_profile(self):
        current_index = self.profiles.index(self.section_var.get())
        if current_index < len(self.profiles) - 1:  # Ensure we don't go out of bounds
            self.section_var.set(self.profiles[current_index + 1])

        self.profile_idx = int(self.section_var.get()) - 1

        self.current_line = []

        if self.auto_update_var.get() == True:
            self.update_plots()

    def first_profile(self):
        current_index = self.profiles.index(self.section_var.get())
        if current_index != 0:  # Ensure we don't go out of bounds
            self.section_var.set(self.profiles[0])
        
        self.profile_idx = int(self.section_var.get()) - 1

        self.current_line = []

        if self.auto_update_var.get() == True:
            self.update_plots()

    def last_profile(self):
        current_index = self.profiles.index(self.section_var.get())
        if current_index != -1:  # Ensure we don't go out of bounds
            self.section_var.set(self.profiles[-1])

        self.profile_idx = int(self.section_var.get()) - 1

        self.current_line = []

        if self.auto_update_var.get() == True:
            self.update_plots()


    def open_boundary_popup(self, method='iso'):
        """
        Opens a popup window for user input before running the iso or gradient method.

        Parameters:
            method (str): Either "iso" for the iso-surface method or "gradient" for the gradient method.
        """
        popup = ctk.CTkToplevel(self.parent)
        popup.title(f"Parameters for {method.capitalize()} Method")
        popup.geometry("350x150")
        popup.grab_set()  # Makes the popup modal

        # Create a grid layout for three rows and four columns
        popup.rowconfigure([0, 1, 2], weight=1)  # Configure rows
        popup.columnconfigure([0, 1, 2, 3], weight=1)  # Configure columns

        # Target Resistivity Input (Row 0)
        if method == 'iso':
            ctk.CTkLabel(popup, text="Target Rho:").grid(row=0, column=0, 
                                                         padx=padx, pady=pady, sticky="w")
            resistivity_entry = ctk.CTkEntry(popup, width=55)
            resistivity_entry.grid(row=0, column=1, 
                                   padx=padx, pady=pady, columnspan=2, sticky="w")

            # Log Spacing Checkbox (Row 1)
            log_var = ctk.BooleanVar(value=False)
            ctk.CTkLabel(popup, text="Use Log(Rho):").grid(row=1, column=0,
                                                           padx=padx, pady=pady, sticky="w")

            log_checkbox = ctk.CTkCheckBox(popup, text="", variable=log_var)
            log_checkbox.grid(row=1, column=1, padx=padx, pady=pady, sticky="w")

        if method == 'gradient':
            pos_grad_var = ctk.BooleanVar(value=False)
            ctk.CTkLabel(popup, text="Positive Gradient:").grid(row=1, column=0, padx=padx, pady=pady, sticky="w")
            pos_grad_checkbox = ctk.CTkCheckBox(popup, text="", variable=pos_grad_var)
            pos_grad_checkbox.grid(row=1, column=1, padx=padx, pady=pady, sticky="w")

            use_cond_var = ctk.BooleanVar(value=False)
            ctk.CTkLabel(popup, text="Use Conductivity:").grid(row=1, column=2, padx=5, pady=5, sticky="w")
            use_cond_checkbox = ctk.CTkCheckBox(popup, text="", variable=use_cond_var)
            use_cond_checkbox.grid(row=1, column=3, padx=padx, pady=pady, sticky="w")

            ctk.CTkLabel(popup, text="Target Rho:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
            resistivity_entry = ctk.CTkEntry(popup, width=55)
            resistivity_entry.grid(row=0, column=1, padx=padx, pady=pady, columnspan=2, sticky="w")

        # Elevation Limits Input (Row 2)
        ctk.CTkLabel(popup, text="Elevation Limits:").grid(row=2, column=0, padx=5, pady=5, sticky="w")

        min_elev_entry = ctk.CTkEntry(popup, width=55)
        min_elev_entry.grid(row=2, column=1, padx=padx, pady=pady, sticky="w")

        max_elev_entry = ctk.CTkEntry(popup, width=55)
        max_elev_entry.grid(row=2, column=2, padx=padx, pady=pady, sticky="w")

        current_index = self.profiles.index(self.section_var.get())
        # Submit Button
        def submit():
            
            use_log = log_var.get()
            min_elev = float(min_elev_entry.get())
            max_elev = float(max_elev_entry.get())

            feature_elev = []
            idx = []

            # Call the appropriate method with user inputs
            if method == "iso":
                target_rho = float(resistivity_entry.get())
                rhos = self.model.profiles[current_index]['rhos']
                depths = self.model.profiles[current_index]['depths']
                elevs = self.model.profiles[current_index]['elev']

                for i in range(rhos.shape[0]):

                    out = self.model.find_iso_surface_depths(rhos=rhos[i], depths=depths[i], 
                                                            elev=elevs[i],  target_rho=target_rho, 
                                                            min_elev=min_elev, max_elev=max_elev)
                    feature_elev.append(out)
                    idx.append(i)

                feature_elev = np.array(feature_elev)
                feature_elev = feature_elev[:, ~np.all(np.isnan(feature_elev), axis=0)]
                print(feature_elev)
                idx = np.array(idx)

                geo_feature = self.model.profiles[current_index]['geo_features']

                for i in range(feature_elev.shape[1]):
                    feature_name = f"feature_{i+1}"  # Unique key for each feature

                    geo_feature[feature_name] = {
                        "idx": idx,
                        "elev": feature_elev[:, i],
                        "lith": 'Unnamed Lithology',
                        "type": "Unknown"
                    }

                self.model.profiles[current_index]['geo_features']  = geo_feature

                self.update_plots()

                self.update_geofeature_editor()

    
            elif method == "gradient":
                self.find_gradient_iso_surface_depths(min_elev, max_elev, use_log)

            popup.destroy()

        submit_button = ctk.CTkButton(popup, text="Submit", command=submit)
        submit_button.grid(row=3, column=0, columnspan=3, padx=padx, pady=pady)


    def add_grad_feature(self):
        current_index = self.profiles.index(self.section_var.get())

        rhos = self.model.profiles[current_index]['rhos']
        depths = self.model.profiles[current_index]['depths']
        elevs = self.model.profiles[current_index]['elev']
        dists = self.model.profiles[current_index]['distances']

        print(rhos.shape)

        idx = []
        feature_elev = []
        

        for i in range(rhos.shape[0]):

            out = self.model.find_gradient_depths(rhos=rhos[i], depths=depths[i], 
                                                  elev=elevs[i], 
                                                  min_elev = -100, max_elev=100)
            
            feature_elev.append(out)

            print(out)

            idx.append(i)

        feature_elev = np.array(feature_elev)
        idx = np.array(idx)

        self.model.profiles[current_index]['geo_features'] = {}

        geo_feature = self.model.profiles[current_index]['geo_features']


        for i in range(feature_elev.shape[1]):
            feature_name = f"feature_{i+1}"  # Unique key for each feature

            geo_feature[feature_name] = {
                "idx": idx,
                "elev": feature_elev[:, i],
                "lith": 'Unamed',
                "type": "Unknown"
            }

        self.model.profiles[current_index]['geo_features']  = geo_feature

        print( self.model.profiles[current_index]['geo_features'])

        self.update_plots()
        
    # Function to move to the previous profile
    def previous_profile(self):
        current_index = self.profiles.index(self.section_var.get())
        if current_index > 0:  # Ensure we don't go out of bounds
            self.section_var.set(self.profiles[current_index - 1])

        self.profile_idx = int(self.section_var.get()) - 1

        self.current_line = []

        if self.auto_update_var.get() == True:
            self.update_plots()

    def update_profile_idx(self):

        self.profile_idx = int(self.section_var.get()) - 1

        self.current_line = []

        if self.auto_update_var.get() == True:
            self.update_plots()


    def get_plot_params(self, param=None):
        try:
            params = {
                "vmin": float(self.vmin_entry.get()),
                "vmax": float(self.vmax_entry.get()),
                "zmin": float(self.zmin_entry.get()),
                "zmax": float(self.zmax_entry.get()),
                "scale": float(self.scale_entry.get()),
            }
        except ValueError:
            params = {"vmin": 1, "vmax": 200, "zmin": -100, "zmax": 10, "scale": 10}  # Default values

        return params[param] if param else (params["vmin"], params["vmax"], params["zmin"], params["zmax"], params["scale"])

    def update_map(self, ax):
        """ Clears and redraws the plot with new limits """
        ax.clear()

        ax.scatter(self.model.ttem_models['x'], self.model.ttem_models['y'], 
                   c='cyan', s=0.5, label='tTEM Data')
            
        if len(self.model.stem_models) != 0:
            x_coords = [stem_model['x'] for stem_model in self.model.stem_models]
            y_coords = [stem_model['y'] for stem_model in self.model.stem_models]
            ax.scatter(x_coords, y_coords, c='k', marker='*', s=20, label='sTEM')
                 
        if len(self.model.boreholes) != 0:
            x_coords = [bh['x'] for bh in self.model.boreholes]
            y_coords = [bh['y'] for bh in self.model.boreholes]
            ax.scatter(x_coords, y_coords, c='orange', marker='*', s=20, label='Boreholes')
                

        ax.plot(self.model.profiles[self.profile_idx]['x'], 
                self.model.profiles[self.profile_idx]['y'],
                c='k', label='Current Line')
        
        ax.legend()

    
        #if self.bg_map.get() != 'None':
         #   if self.bg_map.get() == 'OSM':
         #       cx.add_basemap(
         #          self.axs[0], crs=self.crs,
         #          source=cx.providers.OpenStreetMap.Mapnik,
         #          attribution=False)

         #   if self.bg_map.get() == 'Aerial Imagery':
         #       cx.add_basemap(
         #           self.axs[0], crs=self.crs,
         #           source=cx.providers.Esri.WorldImagery,
         #           attribution=False)
                
        ax.set_aspect(1)
        #ax.set_xlim(self.xmin, self.xmax)
        #ax.set_ylim(self.ymin, self.ymax)
        self.ax1.set_xticks([])
        self.ax1.set_yticks([])

        self.fig1.tight_layout()
        
        self.fig1.canvas.draw()

    def toggle_drawing_mode(self):
    
        """ Toggles the profile drawing mode """
        if self.drawing_mode == 1:
            
            self.drawing_mode = 0  # Disable drawing
            self.draw_toggle_button.configure(text="Draw Feature")

            geo_feature = self.model.profiles[self.profile_idx]['geo_features']

            num_features = len(geo_feature)

            x_vals, y_vals = zip(*self.current_line)

            geo_feature[f'Feature {num_features+1}'] = {"distance": x_vals,
                                                        "elev": y_vals,
                                                        "lith": 'Unspecified Lithology',
                                                        "type": "Unknown"}

            self.model.profiles[self.profile_idx]['geo_features'] = geo_feature

            print(self.model.profiles[self.profile_idx]['geo_features'])

            self.create_geo_feature_editor()

            if len(self.current_line) != 0:
                self.current_line = []

            for line in self.plotted_lines:
                line.remove()

            self.update_lines()
                
        else:
            self.plotted_lines = []
            self.drawing_mode = 1  # Enable drawing
            self.draw_toggle_button.configure(text="Save Feature")

    def update_lines(self, geo_feature=True):

        if self.current_line:
            x_vals, y_vals = zip(*self.current_line)
            self.plotted_lines.append(self.ax2.plot(x_vals, y_vals, color='r', marker='x')[0])

        if geo_feature:
        
            if self.model.profiles[self.profile_idx]['geo_features']:
                
                geo_features = self.model.profiles[self.profile_idx]['geo_features']
                linewidths = [1] * len(geo_features)
                linewidths[0] = 2
                for i, feature in enumerate(geo_features):

                    self.ax2.plot(self.model.profiles[self.profile_idx]['geo_features'][feature]['distance'],
                                  self.model.profiles[self.profile_idx]['geo_features'][feature]['elev'], lw=2, 
                                  ls = '--', c='grey',marker='o')
        self.canvas2.draw()

    def update_plots(self):

        self.profile_idx = int(self.section_var.get()) - 1

        self.update_map(ax=self.ax1)
        self.update_profile()

        self.update_lines()

        self.create_geo_feature_editor()
                
    def update_profile(self):
        """ Refreshes the profile plot """

         # Remove the existing axis
        self.fig2.clear()
        self.ax2 = self.fig2.subplots(1, 1) 
    
        #
        # Set color limits
        vmin, vmax, zmin, zmax, scale = self.get_plot_params()

        scale = int(self.model.profiles[self.profile_idx]['distances'][-1] * 0.002)

        # Redraw the profile plot
        if self.color_var.get() == 'viridis':
            cmap = plt.cm.viridis
        if self.color_var.get() == 'jet':
            cmap = plt.cm.jet
        if self.color_var.get() == 'rainbow':
            cmap = plt.cm.rainbow
        if self.color_var.get() == 'bone':
            cmap = plt.cm.bone
        if self.color_var.get() == 'Color Map':
            cmap = plt.cm.viridis

        self.plot.TEMProfile(profile_idx=self.profile_idx, cmap=cmap,
                            cbar_orientation='horizontal', cbar=True,
                            zmin=zmin, zmax=zmax, ax=self.ax2, 
                            scale=scale, vmin=vmin, vmax=vmax)
        
        if self.model.stem_models: 
            for i, _ in enumerate(self.model.stem_models):  
                self.plot.addTEMSoundings(
                    profile_idx=self.profile_idx, stem_model_idx=i,
                    search_radius=100, cmap=cmap,
                    ax=self.ax2, print_msg=False, vmax=vmax
                )

        if self.model.boreholes: 
            self.plot.addBoreholes(profile_idx=self.profile_idx, 
                                   ax=self.ax2, 
                                   print_msg=False)
            
        self.ax2.set_title(f"Vertical exaggeration: x {scale}")
        self.fig2.tight_layout()
        self.canvas2.draw()
        
    def on_canvas_click(self, event):
        """Handles drawing mode for adding/removing points with optional snapping."""
        if not self.drawing_mode or event.xdata is None or event.ydata is None:
            return
        
        if event.button == 1:  # Left-click to add a point
            self.current_line.append((event.xdata, event.ydata))
            self.update_lines(geo_feature=False)

        elif event.button == 3 and self.current_line:  # Right-click to remove last point
            self.current_line.pop()
            self.update_lines(geo_feature=False)

    def enable_line_selection(self):
        """ Enables line selection mode. Clicking a line will highlight it. """
        self.selecting_line = True
        self.selected_line = None
        self.fig.canvas.mpl_connect('button_press_event', self.select_line)

    def select_line(self, event):
        """ Selects a line when clicked and highlights it in red. """
        if not self.selecting_line:
            return

        for line in self.lines:
            contains, _ = line.contains(event)
            if contains:
                # Reset previous line color
                if self.selected_line:
                    self.selected_line.set_color("black")

                # Highlight the new selected line
                self.selected_line = line
                self.selected_line.set_color("red")
                self.fig.canvas.draw()
                break  # Stop checking once a line is selected

        self.selecting_line = False  # Exit selection mode after selecting a line

    def edit_line(self, event):
        """ Allows moving or deleting points on a selected line """
        if self.selected_line is None:
            return

        x_data, y_data = self.selected_line.get_data()
        points = list(zip(x_data, y_data))

        # Find the closest point
        distances = [(abs(event.xdata - x), abs(event.ydata - y)) for x, y in points]
        closest_idx = np.argmin([sum(d) for d in distances])

        if event.button == 1:  # Left click: Move point
            points[closest_idx] = (event.xdata, event.ydata)
        elif event.button == 3 and len(points) > 2:  # Right click: Delete point
            points.pop(closest_idx)

        # Update the line
        new_x, new_y = zip(*points)
        self.selected_line.set_data(new_x, new_y)
        self.fig.canvas.draw()

if __name__ == "__main__":
    root = ctk.CTk()
    root.title("Standalone Section Plotting")
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()

    mx = 0.9
    my = 0.9

    x_position = (screen_width - screen_width*mx) // 2  # Center horizontally
    y_position = (screen_height - screen_height*my) // 2  # Center vertically
    # Set the window position

    root.geometry(f"{screen_width*mx}x{screen_height*my}+{x_position}+{y_position}")
    SectionPlotTab(root)
    root.mainloop()
