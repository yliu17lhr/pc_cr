#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 08:35:12 2024
@author: yiyanliu

crack detection and measurement process module

This module performs crack detection and measurements using PC-Cr method as described in Chapter 8 of Liu's DPhil thesis.
It includes functionality to load point cloud data and feature matching results,
executes the crack detection and measurement algorithm, and outputs analysis and crack measurement results.
"""

import os
import copy
import sys
import pickle
import json
import numpy as np
import pandas as pd
import open3d as o3d
import importlib

from pc_cr.func_collections import pc_utilities, pc_crack_detection


def crack_detection_measure_process(directory_path):
    """
    Conducts crack detection and measurements for structural components based on
    point cloud data as described in Chapter 8 of Liu's DPhil thesis. This process
    involves loading data, executing crack detection algorithms, and storing the outputs
    in specified directories.

    Parameters:
        directory_path (str): The path containing the configuration file and data.

    Outputs:
        - Analysis results as a Python dictionary in '.pickle' format for each rolling window
          in each structural component containing:
            "corrs_initial": Initial correspondence set;
            "in_bound": Correspondence as a 2D numpy array for each rolling window;
            "KL_filter": Correspondence post-KL divergence filtering;
            "main_filter": Correspondence post-conventional filtering;
            "division": Details of the current rolling window's boundary;
            "crack_clustering": Results from clustering based on kinematic properties;
            "crack_detection": Detected intersections with the rolling window;
            "crack_slope": Slope of the detected crack in the global coordinate system.

        - Crack results as a Python dictionary in '.pickle' format for each structural component:
            'cracked_rec': IDs of rolling windows where cracks are detected;
            "crack_points": Collection of intersection points;
            'crack_measurements': Results of crack measurements;
            "crack_path": Paths connecting detected crack segments.

        - Analysis metadata in '.json' format, containing parameters used for crack analysis.
    """
    os.chdir(directory_path)
    print(f"Changed working directory to {os.getcwd()}")

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

    with open(config.data_file_dirs["registration_metadata"], 'r') as file:
        reg_metadata = json.load(file)
    
    with open(config.data_file_dirs["registration_results"], 'rb') as file:
        feature_reg_results = pd.read_pickle(file)
    
    b_frame_full = pc_utilities.construct_b_frame(general_data["plot_elements_dict"], general_data["plot_coords"])
    b_frame = b_frame_full.loc[list(config.sc_offsets.keys())]

    analysis_results, crack_res, analysis_metadata = {}, {}, {}
    
    structural_component = list(config.sc_offsets.keys())[0]
    sc_key = structural_component
    print(f"Processing {structural_component}")
    
    analysis_results[structural_component] = {}
    crack_res[structural_component] ={}
    analysis_metadata[structural_component] = {}

    voxel_size = reg_metadata["average_point_distance"][structural_component]["voxel_size"]
    
    pretest_cloud_path = os.path.join(config.data_file_dirs["downsized_cloud"], f"Pretest_{sc_key}_cloud_down.ply")
    posttest_cloud_path = os.path.join(config.data_file_dirs["downsized_cloud"], f"Posttest_{sc_key}_cloud_down.ply")
    pretest_fpfh_path = os.path.join(config.data_file_dirs["FPFH_feature"], f"Pretest_{sc_key}_FPFH.pcd")
    
    pc_pretest_down = o3d.io.read_point_cloud(pretest_cloud_path)
    pc_posttest_down = o3d.io.read_point_cloud(posttest_cloud_path)
    pc_pretest_fpfh = o3d.io.read_feature(pretest_fpfh_path)

    rec_divisions = pc_crack_detection.rectangle_division(b_frame, structural_component, config.sc_divisions[structural_component]["m"], config.sc_divisions[structural_component]["n"], offset=config.sc_offsets[structural_component])
    
    intersection_collection, crack_stats, cracked_rec = [], [], []
    

    for div_num, rec_division in enumerate(rec_divisions):
        
        print(f"Executing rectangle division number {div_num}")
        
        analysis_results[structural_component][div_num] = {}
        
        source_corrs, target_corrs = pc_utilities.feature_corrs_in_boundary(pc_pretest_down, feature_reg_results[structural_component][:,0], feature_reg_results[structural_component][:,1],  None,
        None,boundary=rec_division,  returnmask = False)

        if len(source_corrs) <= 20:
            
            analysis_results[structural_component][div_num] = {
                'corrs_inital': np.empty((0, 2), dtype=np.int32),
                'in_bound': np.empty((0, 2), dtype=np.int32),
                'KL_filter': np.empty((0, 2), dtype=np.int32),
                'main_filter': np.empty((0, 2), dtype=np.int32),
                'division': rec_division,
                'crack_clustering': {
                    'cluster_0': {
                        'filter': {
                            'inliers': np.empty((0,), dtype=np.int32),
                            'model_para': np.asarray([0,0,0]),
                            'current_quality': None,
                            'labels': None
                        },
                        'main': {
                            'inliers': np.empty((0,), dtype=np.int32),
                            'model_para': np.asarray([0,0,0]),
                            'current_quality': None,
                            'labels': None,
                            'centroid': np.asarray([ 0, (rec_division[0]+rec_division[1])/2 , (rec_division[2]+rec_division[3])/2]),
                            'radius': None
                        }
                    }
                },
                'crack_detection': None,
                'crack_slope': None
            }
            
            continue

        source_corrs_KL, target_corrs_KL, _ = pc_utilities.KL_filter(pc_pretest_fpfh, source_corrs, target_corrs, alpha=config.global_analysis_paras["KL_alpha"], scale=config.global_analysis_paras["KL_scale"])

        source_corrs_b, target_corrs_b = pc_utilities.correspondence_filter(pc_pretest_down, pc_posttest_down, source_corrs_KL, target_corrs_KL, distance_threshold=config.global_analysis_paras["dis_filter"], normal_threshold=config.global_analysis_paras["normal_filter"], outplane_threshold=config.global_analysis_paras["outplane_filter"])
        
        assessed_ran_paras = copy.deepcopy(config.ran_crack_paras)
        assessed_ran_paras["f_ransac_threshold"] = config.ran_crack_paras["f_ransac_threshold"](voxel_size)
        assessed_ran_paras["m_ransac_threshold"] = config.ran_crack_paras["m_ransac_threshold"](voxel_size)
        
        clustering_results = pc_crack_detection.ransac_crack_clustering(pc_pretest_down, pc_posttest_down, source_corrs_b, target_corrs_b, **assessed_ran_paras)
        
        intersection, slope = pc_crack_detection.crack_detection(clustering_results, rec_division)
        if intersection:
            intersection_collection.extend(intersection)
            crack_stat = pc_crack_detection.crack_measurement(clustering_results, intersection, slope, rec_division)
            cracked_rec.append(div_num)
            crack_stats.extend(crack_stat)

        analysis_results[structural_component][div_num] = {
            "corrs_initial": feature_reg_results[structural_component],
            "in_bound": np.asarray([source_corrs, target_corrs]).T,
            "KL_filter": np.asarray([source_corrs_KL, target_corrs_KL]).T,
            "main_filter": np.asarray([source_corrs_b, target_corrs_b]).T,
            "division": rec_division,
            "crack_clustering": clustering_results,
            "crack_detection": intersection,
            "crack_slope": slope
        }

    if intersection_collection:
        crack_res[structural_component] = {
            'cracked_rec': cracked_rec,
            "crack_points": intersection_collection,
            'crack_measurements': crack_stats,
            "crack_path": pc_crack_detection.optimal_tsp(intersection_collection)[0]  # Storing only the path
        }
 
    analysis_metadata[structural_component]["ran_paras"] = assessed_ran_paras

    if not os.path.exists(config.data_file_dirs["analysis_results"]):
        os.makedirs(config.data_file_dirs["analysis_results"])

    with open(os.path.join(config.data_file_dirs["analysis_results"], "analysis_results.pickle"), 'wb') as file:
        pickle.dump(analysis_results, file)
    with open(os.path.join(config.data_file_dirs["analysis_results"], "crack_results.pickle"), 'wb') as file:
        pickle.dump(crack_res, file)
    with open(os.path.join(config.data_file_dirs["analysis_results"], "analysis_metadata.json"), 'w') as file:
        json.dump(analysis_metadata, file)

if __name__ == "__main__":
    
    directory_path = input("Enter the directory path where your configuration file and data are located: ")
    crack_detection_measure_process(directory_path)

