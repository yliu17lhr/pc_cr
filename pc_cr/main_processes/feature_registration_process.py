#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 13 18:00:04 2024
@author: yiyanliu

feature registration process module

This module implements the feature calculation and matching process as part of PC-Cr method
as described in Section 6.2.3 of Yiyan Liu’s DPhil thesis. The module processes point cloud data 
from masonry buildings affected by ground movement. It includes alignment to XY plane, downsampling, 
feature extraction (FPFH), and feature matching phases, storing outputs for subsequent crack detection and measurement analysis.
"""

import os
import sys
import pickle
import json
import importlib

import numpy as np
import open3d as o3d

from pc_cr.func_collections import pc_feature_reg, pc_preprocess

def feature_registration_process(directory_path, cloud_paths):
    """
    Processes point clouds to calculate features and match them across different tests.
    
    This function changes the current working directory to `directory_path`, imports the 
    configuration settings from `feature_registration_config.py`, processes each structural component's 
    point cloud data, and computes their Fast Point Feature Histograms (FPFH). It performs feature matching 
    using a local region feature matching algorithm. Results and metadata are saved in specified directories.
    *************
    Please note the the function assumes that the x-coordinate of both pre-test and post-test point clouds is aligned with the minor axis. 
    This minor axis represents the direction of least variance in the dataset, as determined by Principal Component Analysis (PCA).
    *************
    
    Parameters:
    directory_path (str): The path to the directory containing the `feature_registration_config.py` file 
                          and raw point cloud files.
    
    Outputs:
    - Pairing indices for pre and post-test point clouds saved in `.pickle` format.
    - Downsampled and aligned point clouds saved in `.ply` format.
    - FPFH signatures saved in `.pcd` format.
    - Metadata containing average point densities and general parameters used in the process saved in `.json`.
    """
    os.chdir(directory_path)
    print(f"Changed working directory to {os.getcwd()}")
    sys.path.insert(0, directory_path)

    try:
        import feature_registration_config as config
        importlib.reload(config)  
        sys.path.pop(0)

    except ModuleNotFoundError:
        print("Configuration module not found in the specified directory.")
        sys.exit(1)

    folder_names = config.folder_names
    sc_key = config.sc_name
    gen_paras = config.general_parameters
    pretest_path = cloud_paths["pre_test"]
    posttest_path = cloud_paths["post_test"]

    ave_point_dist = {}
    reg_results = {}

    print(f"Registering ==> {sc_key}")

    print(pretest_path)
    pc_pretest = pc_preprocess.load_pointcloud_fromtxt(pretest_path, rowstoskip=1, align_angle=gen_paras["rotation_angle"], visualisation=False)
    pc_posttest = pc_preprocess.load_pointcloud_fromtxt(posttest_path, rowstoskip=1, align_angle=gen_paras["rotation_angle"], visualisation=False)

    pretest_avg_dist = np.mean(pc_pretest.compute_nearest_neighbor_distance())
    posttest_avg_dist = np.mean(pc_posttest.compute_nearest_neighbor_distance())
    ave_point_dist[sc_key] = {"pretest_avg_dist": pretest_avg_dist, "posttest_avg_dist": posttest_avg_dist}

    try:
        ceiling_size = round(gen_paras["ceiling_multiplier"] * max(pretest_avg_dist, posttest_avg_dist), 4)
        ave_point_dist[sc_key]["voxel_size"] = ceiling_size

        pc_pretest_down, pc_pretest_fpfh = pc_preprocess.preprocess_point_cloud(pc_pretest, ceiling_size, gen_paras["radius_normal_multiplier"], gen_paras["radius_feature_multiplier"])
        pc_posttest_down, pc_posttest_fpfh = pc_preprocess.preprocess_point_cloud(pc_posttest, ceiling_size, gen_paras["radius_normal_multiplier"], gen_paras["radius_feature_multiplier"])

        cloud_folder, feature_folder, corr_folder = [os.path.join(directory_path, folder) for folder in (folder_names["processed_cloud"], folder_names["FPFH"], folder_names["corrs"])]
        for folder in [cloud_folder, feature_folder, corr_folder]:
            os.makedirs(folder, exist_ok=True)

        o3d.io.write_point_cloud(os.path.join(cloud_folder, f"Pretest_{sc_key}_cloud_down.ply"), pc_pretest_down)
        o3d.io.write_point_cloud(os.path.join(cloud_folder, f"Posttest_{sc_key}_cloud_down.ply"), pc_posttest_down)
        o3d.io.write_feature(os.path.join(feature_folder, f"Pretest_{sc_key}_FPFH.pcd"), pc_pretest_fpfh)
        o3d.io.write_feature(os.path.join(feature_folder, f"Posttest_{sc_key}_FPFH.pcd"), pc_posttest_fpfh)

        corr_set = pc_feature_reg.find_local_correspondences(pc_pretest_down, pc_posttest_down, pc_pretest_fpfh, pc_posttest_fpfh, r=gen_paras["local_region_search_radius"], mutual_filter=True)
        reg_results[sc_key] = corr_set
        with open(os.path.join(corr_folder, f"{sc_key}_corrs_set.pickle"), 'wb') as handle:
            pickle.dump(corr_set, handle, protocol=3)

    except ValueError as e:
        print(f"Error processing {sc_key}: {e}")

    with open(os.path.join(directory_path, 'feature_registration_metadata.json'), 'w') as file:
        json.dump({"general_parameters": gen_paras, "average_point_distance": ave_point_dist}, file)
    with open(os.path.join(directory_path, 'feature_registration_results.pickle'), 'wb') as file:
        pickle.dump(reg_results, file)

    for key in reg_results:
        print(f"{key}: ===> {len(reg_results[key])} correspondences found")

if __name__ == "__main__":
    directory_path = input("Enter the directory path where your config file and data are located: ")
    feature_registration_process(directory_path)

