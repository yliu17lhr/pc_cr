#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 10:59:26 2024
@author: yiyanliu

Crack plot process module

This module provides functions to process and visualise crack detection results 
from PC-Cr method. The main function, `crack_plot_process`, handles the loading 
of crack analysis data and generates visualisations based on a configurable set of plotting steps.
"""

import open3d as o3d
import sys
import os
import pandas as pd
import pickle
import importlib

from pc_cr.func_collections import pc_draw, pc_utilities

def crack_plot_process(directory_path, plot_lists):
    """
    Process and plot crack-related data based on configuration and specified plotting steps.

    Args:
        directory_path (str): The directory path where configuration and data files are located.
        plot_lists (list of int): A list of integers (0 or 1) indicating which steps to execute:
            - plot_lists[0] == 1: Plot the correspondence between pre-test and post-test point clouds.
            - plot_lists[1] == 1: Plot the crack path on the post-test point cloud.
            - plot_lists[2] == 1: Plot the feature point clustering results in rolling windows where a crack is identified.

    Raises:
        SystemExit: If the configuration module is not found in the specified directory.
    """

    sys.path.insert(0, directory_path)
    try:
        import crack_detection_process_config as config
        importlib.reload(config)
    except ModuleNotFoundError:
        print("Configuration module not found in the specified directory.")
        sys.exit(1)
    finally:
        sys.path.pop(0)

    general_data = {}
    for name in config.gen_data_names:
        path_name = os.path.join(config.data_file_dirs["gen_data"], f"{name}.pickle")
        with open(path_name, 'rb') as file:
            general_data[name] = pd.read_pickle(file)
            
    sc_name = list(config.sc_offsets.keys())[0]

    with open(os.path.join(config.data_file_dirs["analysis_results"], "crack_results.pickle"), 'rb') as handle:
        crack_res = pickle.load(handle)
        
    with open(os.path.join(config.data_file_dirs["analysis_results"], "analysis_results.pickle"), 'rb') as handle:
        analysis_res = pickle.load(handle)
    
    b_frame_full = pc_utilities.construct_b_frame(general_data["plot_elements_dict"], general_data["plot_coords"])
    
    pretest_cloud_file = f"Pretest_{sc_name}_cloud_down.ply"
    posttest_cloud_file = f"Posttest_{sc_name}_cloud_down.ply"
    pretest_cloud_path = os.path.join(directory_path, 'corr_results/downsized_clouds', pretest_cloud_file)
    posttest_cloud_path = os.path.join(directory_path, 'corr_results/downsized_clouds', posttest_cloud_file)
    corrs_res_path = os.path.join(directory_path, 'feature_registration_results.pickle')

    with open(corrs_res_path, 'rb') as file:
        corrs = pd.read_pickle(file)

    pc_pretest_down = o3d.io.read_point_cloud(pretest_cloud_path)
    pc_posttest_down = o3d.io.read_point_cloud(posttest_cloud_path)
    
    if plot_lists[0] == 1:
        print("Executing step 1: Drawing correspondence between point clouds")
        pc_draw.draw_correspondence(pc_pretest_down, pc_posttest_down, corrs[sc_name][:, 0], corrs[sc_name][:, 1])
    
    if plot_lists[1] == 1:
        print("Executing step 2: Drawing crack path")
        pc_draw.draw_crack(
            pc_posttest_down, b_frame_full, sc_name, crack_res[sc_name]["crack_points"], crack_res[sc_name]["crack_path"],
            crack_measurements=crack_res[sc_name]["crack_measurements"], figsize=(8, 6), show_crack_width=True,
            coloured_cloud=True, axis_on=False, savefig=True, file_name=os.path.join(directory_path,f"{sc_name}_crack.png"))

    if plot_lists[2] == 1:
        print("Executing step 3: Drawing crack clustering")
        for i in crack_res[sc_name]["cracked_rec"]:
            pc_draw.draw_crack_clustering(
                pc_posttest_down, analysis_res, b_frame_full, sc_name, rec_num=[i], figsize=(8, 6),
                draw_clustering=True, draw_rec=True, coloured_cloud=False, savefig=True, axis_on=False,
                file_name=os.path.join(directory_path,f"{sc_name}_window{i}_crack_clustering.png"))

    