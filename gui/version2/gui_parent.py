#%%

import customtkinter as ctk
from tab_load_data import LoadDataTab
from tab_plot_sections import SectionPlotTab
from tab_geological_modeling import GeologicalModelingTab
#from tab_settings import SettingsTab
#from tab_search import SearchTab

class MyApp(ctk.CTk):
    """ Main application that loads tabs dynamically """

    def __init__(self):
        super().__init__()
        self.title("CustomTkinter Modular App")
        self.geometry("1050x850")

        self.create_tabs()

    def create_tabs(self):
        """ Creates the tab structure and loads modules dynamically """
        self.tabview = ctk.CTkTabview(self, fg_color="#2A2D2E", corner_radius=15)
        self.tabview.pack(fill="both", expand=True, padx=20, pady=20)

        # Create tab instances
        self.tab1 = self.tabview.add("Load Files")
        self.tab2 = self.tabview.add("Plot Profiles")
        #self.tab3 = self.tabview.add("Settings")
        self.tab4 = self.tabview.add("Search")

        # Load each module inside the corresponding tab
        LoadDataTab(self.tab1)
        SectionPlotTab(self.tab2)
        #GeologicalModelingTab(self.tab3)
        #SearchTab(self.tab4)


if __name__ == "__main__":
    app = MyApp()
    app.mainloop()

    