#%%
import customtkinter as ctk
import tkinter as tk
import threading
import gempy as gp
import gempy_viewer as gpv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

class GeologicalModelingTab:
    """ UI tab for launching GemPy and running the tutorial."""

    def __init__(self, parent_frame):
        self.parent = parent_frame
        self.geo_model = None  # Store the geological model
        np.random.seed(1234)

        # CustomTkinter Theming
        ctk.set_appearance_mode("dark")  # Options: "System", "Light", "Dark"
        ctk.set_default_color_theme("blue")

        self.setup_ui()

    def setup_ui(self):
        """ Sets up UI elements."""
        control_frame = ctk.CTkFrame(self.parent)
        control_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ctk.CTkButton(control_frame, text="Run GemPy Tutorial", command=self.run_gempy_tutorial).pack(side=tk.LEFT, padx=5)
        
        self.create_plot_area()
    
    def create_plot_area(self):
        """ Embeds Matplotlib into the Tkinter frame."""
        self.canvas_frame = ctk.CTkFrame(self.parent)
        self.canvas_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.fig, self.ax = plt.subplots(figsize=(6, 4))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.canvas_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.canvas_frame)
        self.toolbar.update()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def run_gempy_tutorial(self):
        """ Runs the GemPy tutorial in a separate thread to avoid UI freezing."""
        threading.Thread(target=self.run_gempy_model, daemon=True).start()
    
    def run_gempy_model(self):
        """ Executes the GemPy tutorial and updates the plot."""
        
        data_path = 'https://raw.githubusercontent.com/cgre-aachen/gempy_data/master/'
        
        # Load example model
        self.geo_model = gp.create_geomodel(
            project_name='Tutorial_ch1_1_Basics',
            extent=[0, 2000, 0, 2000, 0, 750],
            resolution=[20, 20, 20],
            refinement=4,
            importer_helper=gp.data.ImporterHelper(
                path_to_orientations=data_path + "/data/input_data/getting_started/simple_fault_model_orientations.csv",
                path_to_surface_points=data_path + "/data/input_data/getting_started/simple_fault_model_points.csv",
                hash_surface_points="4cdd54cd510cf345a583610585f2206a2936a05faaae05595b61febfc0191563",
                hash_orientations="7ba1de060fc8df668d411d0207a326bc94a6cdca9f5fe2ed511fd4db6b3f3526"
            )
        )

        gp.map_stack_to_surfaces(
            gempy_model=self.geo_model,
            mapping_object={
                "Fault_Series": 'Main_Fault',
                "Strat_Series": ('Sandstone_2', 'Siltstone', 'Shale', 'Sandstone_1')
            }
        )

        gp.set_is_fault(
            frame=self.geo_model.structural_frame,
            fault_groups=['Fault_Series']
        )

        gp.set_section_grid(
            grid=self.geo_model.grid,
            section_dict={
                'section1': ([0, 0], [2000, 2000], [100, 80]),
                'section2': ([800, 0], [800, 2000], [150, 100]),
                'section3': ([0, 200], [1500, 500], [200, 150])
            }
        )

        gp.set_topography_from_random(
            grid=self.geo_model.grid,
            fractal_dimension=1.2,
            d_z=np.array([300, 750]),
            topography_resolution=np.array([50, 50])
        )

        self.geo_model.interpolation_options.mesh_extraction = False
        sol = gp.compute_model(self.geo_model)

        # Update plot
        self.ax.clear()
        gpv.plot_2d(self.geo_model, section_names=['section1', 'section2', 'section3', 'topography'], show_topography=True)
        self.fig.canvas.draw()

if __name__ == "__main__":
    root = ctk.CTk()
    root.title("GemPy Launcher Tab")
    root.geometry("900x600")
    GeologicalModelingTab(root)
    root.mainloop()
