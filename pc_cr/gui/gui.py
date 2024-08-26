#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" 
Created on Mon May 13 12:13:16 2024
@author: yiyanliu

PC-Cr GUI module

This module provides a graphical user interface (GUI) for detecting and measuring cracks 
in segmented structural components using point clouds taken before and after a deformation event, 
based on the PC-Cr method. The method is described in Chapter 8 of Liu's DPhil thesis on 
Displacement and Damage Monitoring for Masonry Buildings Subjected to Ground Movements Induced by Underground Construction.
"""

import os
import pickle
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from pc_cr.main_processes import feature_registration_process, crack_detect_measure_process, crack_plot_process
from pc_cr.func_collections import pc_utilities

def main():
    app = GUI()
    app.mainloop()

class GUI(tk.Tk):
    
    """
    A graphical user interface (GUI) for the PC-Cr method, inheriting from tkinter's Tk class.

    This class provides a interface for performing tasks related to feature registration, crack detection 
    and measurement in structural components using methods descirbed in Liu's DPhil thesis. It includes functionalities for 
    selecting directories and files, configuring processing parameters, and initiating the 
    registration, crack detection, and crack plotting processes.

    Attributes:
        work_dir_path (str): The path to the working directory selected by the user.
        raw_cloud_files (dict): A dictionary storing the file paths for the pre-test and post-test point clouds.
        registration_entries (dict): A dictionary of tkinter Entry widgets for registration configuration parameters.
        crack_entries (dict): A dictionary of tkinter Entry widgets for crack detection configuration parameters.
        check_vars (list of tk.IntVar): A list of IntVar objects associated with checkbuttons for crack plotting options.
        status_var (tk.StringVar): A StringVar object for updating the status bar.

    Methods:
        init_styles(): Configures the styles for various tkinter widgets used in the GUI.
        create_widgets(): Creates and places all the necessary widgets for the GUI, including buttons, labels, and tabs.
        select_directory(): Opens a dialog for selecting the working directory and updates the GUI accordingly.
        select_file(file_type): Opens a dialog for selecting the pre-test or post-test point cloud file.
        start_processing_task(task_type): Starts the selected processing task (registration, crack detection, or crack plotting).
        generate_config(template_path, config_type): Generates configuration files based on user inputs for registration and crack detection.
        checkbox_changed(): Handles the state changes of checkboxes in the GUI.
        update_status(message): Updates the status bar with the provided message.
    """
    
    def __init__(self):
        super().__init__()
        self.title("Point Cloud Crack Detection GUI")
        self.geometry("820x410")
        self.configure(bg="#f0f0f0")
        self.resizable(False, False)
        self.init_styles()
        self.create_widgets()
        self.work_dir_path = ""
        self.raw_cloud_files = {"pre_test": '', "post_test": ''}

    def init_styles(self):
        style = ttk.Style()
        style.configure('TLabel', background='#f0f0f0', foreground='black', font=('Helvetica', 12))
        style.configure('TEntry', font=('Helvetica', 12), foreground='blue')
        style.configure('TButton', font=('Helvetica', 12), background='#f0f0f0', foreground='green')
        style.configure('TCheckbutton', font=('Helvetica', 10), background='#f0f0f0', foreground='blue')

        style.map('TCheckbutton',
                  background=[('active', '#f0f0f0'), ('selected', '#c0c0c0')],
                  foreground=[('pressed', 'red'), ('active', 'blue')],
                  indicatorcolor=[('selected', 'green'), ('pressed', 'red')])

        style.map('TButton',
                  foreground=[('pressed', 'red'), ('active', 'green')],
                  background=[('pressed', '!disabled', 'black'), ('active', 'white')])

    def create_widgets(self):
        notebook = ttk.Notebook(self)
        notebook.pack(padx=10, pady=10, fill="both", expand=True)

        main_tab = ttk.Frame(notebook)
        notebook.add(main_tab, text="Main Processes")

        title_label = ttk.Label(main_tab, text="PC-Cr Main Processes", font=("Helvetica", 14, "bold"))
        title_label.pack(pady=(0, 20))

        button_frame = ttk.Frame(main_tab)
        button_frame.pack(pady=(0, 20))

        buttons_info = [
            ("Select Working Directory", self.select_directory),
            ("Select Pre-test Cloud", lambda: self.select_file("pre")),
            ("Select Post-test Cloud", lambda: self.select_file("post")),
            ("Start Registration",  lambda: self.start_processing_task('registration')),
            ("Start Crack Detection", lambda: self.start_processing_task('crack_detection')),
            ("Start Crack Plotting", lambda: self.start_processing_task('crack_plot'))
        ]

        max_text_length = max(len(text) for text, _ in buttons_info)
        button_width = max_text_length + 6

        for i, (text, command) in enumerate(buttons_info):
            button = ttk.Button(button_frame, text=text, width=button_width, command=command, style='TButton')
            button.grid(row=i, column=0, padx=10, pady=5, sticky="ew")

        message_frame = ttk.Frame(main_tab)
        message_frame.pack(side="bottom", fill="both", expand=True)

        self.message_label = ttk.Label(message_frame, text="PC-Cr (pronounced like 'Pika' in Pikachu)",
                                       wraplength=450, justify="left", style='TLabel')
        self.message_label.pack(side="bottom", fill="x", expand=True, padx=10, pady=(0, 10))

        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        status_bar = ttk.Label(self, textvariable=self.status_var, relief="sunken", anchor="w", style='TLabel')
        status_bar.pack(side="bottom", fill="x")

        tab_names = ["Registration Configuration", "Crack Detection Configuration", "Crack Plotting Configuration"]
        config_tabs = []
        for name in tab_names:
            tab = ttk.Frame(notebook)
            notebook.add(tab, text=name)
            config_tabs.append(tab)

        input_frame_1 = ttk.Frame(config_tabs[0])
        input_frame_1.pack(pady=20)

        reg_labels = ["Structural Component Name", "Ceiling Multiplier",
                      "Local Region Search Radius", "Radius Normal Multiplier",
                      "Radius Feature Multiplier"]
        self.registration_entries = {}
        for i, label_text in enumerate(reg_labels):
            label_text_colon = label_text + ":"
            label = ttk.Label(input_frame_1, text=label_text_colon, style='TLabel')
            label.grid(row=i, column=0, padx=10, pady=5, sticky="w")
            entry = ttk.Entry(input_frame_1, style='TEntry')
            entry.grid(row=i, column=1, padx=10, pady=5, sticky="ew")
            self.registration_entries[label_text] = entry

        generate_button = ttk.Button(config_tabs[0], text="Generate Registration Config",
                                     command=lambda: self.generate_config('feature_registration_config_temp.py', 'registration'),
                                     style='TButton')
        generate_button.pack(pady=(20, 0))

        entries_frame = ttk.Frame(config_tabs[1])
        entries_frame.pack(side="left", fill="both", expand=True)

        canvas = tk.Canvas(entries_frame, bg="#ffffff")
        canvas.pack(side="left", fill="both", expand=True)

        scrollbar = ttk.Scrollbar(entries_frame, orient="vertical", command=canvas.yview)
        scrollbar.pack(side="right", fill="y")

        canvas.configure(yscrollcommand=scrollbar.set)

        entries_container = ttk.Frame(canvas)
        canvas.create_window((0, 0), window=entries_container, anchor="nw")

        crack_labels = [
            "X Coordinate of Point 1", "Y Coordinate of Point 1",
            "X Coordinate of Point 2", "Y Coordinate of Point 2",
            "X Coordinate of Point 3", "Y Coordinate of Point 3",
            "X Coordinate of Point 4", "Y Coordinate of Point 4",
            "X Max Offset", "X Min Offset", "Y Max Offset", "Y Min Offset",
            "Number of Rolling Window Horizontal", "Number of Rolling Window Vertical",
            "KL Alpha", "KL Scale", "Distance Filter", "Normal Filter",
            "Out-of-plane Filter", "Minimum Centroid Distance", "Minimum Differential Displacement Multiplier"
        ]

        self.crack_entries = {}
        for i, label_text in enumerate(crack_labels):
            label_text_colon = label_text + ":"
            label = ttk.Label(entries_container, text=label_text_colon, style='TLabel')
            label.grid(row=i, column=0, padx=10, pady=5, sticky="w")

            if label_text == 'KL Scale':
                kl_scale_options = ["log", "normal"]
                kl_scale_var = tk.StringVar(value="log")
                entry = ttk.Combobox(entries_container, textvariable=kl_scale_var, values=kl_scale_options, state="readonly", style='TEntry')
            else:
                entry = ttk.Entry(entries_container, style='TEntry')

            entry.grid(row=i, column=1, padx=10, pady=5, sticky="ew")
            self.crack_entries[label_text] = entry

        generate_button = ttk.Button(entries_container, text="Generate Crack Detection Config",
                                     command=lambda: self.generate_config("crack_detection_process_config_temp.py", 'crack_detection'),
                                     style='TButton')
        generate_button.grid(row=len(crack_labels) + 1, column=0, columnspan=2, pady=(20, 0))

        entries_container.bind("<Configure>", lambda event: canvas.configure(scrollregion=canvas.bbox("all")))

        options = ["Draw Feature Correspondence", "Draw Identified Crack", "Draw Crack Clustering"]
        self.check_vars = [tk.IntVar() for _ in options]
        
        print(self.check_vars)

        for i, (option, var) in enumerate(zip(options, self.check_vars)):
            label = ttk.Label(config_tabs[2], text=option, style='TLabel')
            label.grid(row=i, column=0, padx=(10, 2), pady=10, sticky='w')

            checkbox = ttk.Checkbutton(config_tabs[2], variable=var, style='TCheckbutton')
            checkbox.grid(row=i, column=1, padx=(2, 10), pady=10, sticky='ew')

    def select_directory(self):
        directory = filedialog.askdirectory()
        if directory:
            self.message_label.config(text=f"Selected Directory: {directory}")
            self.update_status(f"Selected directory: {directory}")
            self.work_dir_path = directory
        else:
            self.message_label.config(text="No Directory Selected")
            self.update_status("No directory selected.")

    def select_file(self, file_type):
        file_path = filedialog.askopenfilename(title=f"Select {'Pre-test' if file_type == 'pre' else 'Post-test'} Cloud")
        if file_path:
            if file_type == 'pre':
                self.message_label.config(text=f"Selected Pre-test Cloud: {file_path}")
                self.update_status(f"Selected Pre-test Cloud: {file_path}")
                self.raw_cloud_files['pre_test'] = file_path
            elif file_type == 'post':
                self.message_label.config(text=f"Selected Post-test Cloud: {file_path}")
                self.update_status(f"Selected Post-test Cloud: {file_path}")
                self.raw_cloud_files['post_test'] = file_path
        else:
            print("No file selected")

    def start_processing_task(self, task_type):
        self.message_label.config(text="Task started. Processing...")
        self.update_status("Task started.")
    
        if task_type == "registration":
            feature_registration_process.feature_registration_process(self.work_dir_path, self.raw_cloud_files)
        elif task_type == "crack_detection":
            crack_detect_measure_process.crack_detection_measure_process(self.work_dir_path)
        elif task_type == "crack_plot":
            values = [var.get() for var in self.check_vars]
            crack_plot_process.crack_plot_process(self.work_dir_path, values)
        self.message_label.config(text="Task Completed")
        self.update_status("Task Completed.")

    def generate_config(self, template_path, config_type):
        
        if config_type == 'registration':
            params = {label.lower().replace(' ', '_').strip(): entry.get()
                      for label, entry in self.registration_entries.items()}
        
        elif config_type == 'crack_detection':
            params = {label.lower().replace(' ', '_').strip(): entry.get()
                      for label, entry in self.crack_entries.items()}
            params['structural_component_name'] = self.registration_entries['Structural Component Name'].get()
            params['base_dir'] = self.work_dir_path
            
            coords_array = np.asarray([
                (self.crack_entries["X Coordinate of Point 1"].get(),
                 self.crack_entries["Y Coordinate of Point 1"].get()),
                (self.crack_entries["X Coordinate of Point 2"].get(),
                 self.crack_entries["Y Coordinate of Point 2"].get()),
                (self.crack_entries["X Coordinate of Point 3"].get(),
                 self.crack_entries["Y Coordinate of Point 3"].get()),
                (self.crack_entries["X Coordinate of Point 4"].get(),
                 self.crack_entries["Y Coordinate of Point 4"].get())
            ])
                        
            plot_coords = pc_utilities.construct_coords_frame(coords_array)
            plot_element_dict = {params['structural_component_name']:[0,1,2,3] }
            
            analysis_config_data_path = os.path.join(params['base_dir'], "analysis_config_data")
            
            if not os.path.exists(analysis_config_data_path):
                os.makedirs(analysis_config_data_path)
                
            with open(os.path.join(analysis_config_data_path, 'plot_coords.pickle'), 'wb') as file:
                    pickle.dump(plot_coords, file)
            
            with open(os.path.join(analysis_config_data_path, 'plot_elements_dict.pickle'), 'wb') as file:
                            pickle.dump(plot_element_dict, file)
            
        file_path = os.path.join(self.work_dir_path,template_path[:-8]+'.py')
        
        template_dir = os.path.dirname(os.path.abspath(__file__))
        template_path = os.path.join(template_dir, template_path)
        
        with open(template_path, 'r') as template_file:
            template_content = template_file.read()
            
        if all(params.values()):
            
            formatted_content = template_content
            try:
                for key, value in params.items():
                    temp_template = f"{{{key}}}"
                    formatted_content = formatted_content.replace(temp_template, str(value))
            except KeyError as e:
                print(f"Error formatting the key: {key} with error: {e}")
            
            with open(file_path, 'w') as output_file:
                output_file.write(formatted_content)
    
            self.message_label.config(text="Configuration file generated successfully!")
            self.update_status("Configuration file generated.")
        
        else:
            messagebox.showwarning("Missing Input", "Please enter values for all parameters.")
            self.update_status("Configuration file generation failed: Missing input.")
                
    def checkbox_changed(self):
        if self.var1.get() == 1:
            self.checkbox3.configure(state=tk.DISABLED)
        else:
            self.checkbox3.configure(state=tk.NORMAL)
            
    def update_status(self, message):
        self.status_var.set(message)

if __name__ == "__main__":
    main()
