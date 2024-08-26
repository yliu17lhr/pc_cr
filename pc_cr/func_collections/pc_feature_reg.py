# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 07:52:13 2024

@author: yiyanliu

Point cloud feature registration module

This module provides functionality to find local correspondences between two point clouds
using their Fast Point Feature Histogram (FPFH) signitures. The module leverages spatial and feature
space proximity to establish these correspondences, optimised for scenarios involving
masonry buildings affected by underground construction. See Section 6.2.3.3 of Yiyan Liu’s
DPhil thesis for further details.
"""

from scipy.spatial import cKDTree
import numpy as np
from tqdm import tqdm

from pc_cr.func_collections.pc_utilities import find_knn_cpu

def find_local_correspondences(pc_pretest, pc_posttest,
                               pretest_fpfh, posttest_fpfh,
                               r=0.1, mutual_filter=True):
    """
    Finds local correspondences between two point clouds based on FPFH signatures.
    
    This function computes correspondences by matching every point in pc_pretest with points in
    pc_posttest within a specified radius 'r' using their FPFH signatures for matching. Optionally,
    mutual correspondences can be filtered to retain only bijective matches.

    Parameters:
    - pc_pretest (open3d.geometry.PointCloud): The pre-test point cloud.
    - pc_posttest (open3d.geometry.PointCloud): The post-test point cloud.
    - pretest_fpfh (open3d.pipelines.registration.Feature): FPFH features for the pre-test point cloud.
    - posttest_fpfh (open3d.pipelines.registration.Feature): FPFH features for the post-test point cloud.
    - r (float): The radius within which to search for point correspondences.
    - mutual_filter (bool): If True, filters correspondences to mutual (bijective) ones only.

    Returns:
    - numpy.ndarray: An array of shape (N, 2) where each row contains indices of corresponding
      points in pc_pretest and pc_posttest.
    """
    pc_pretest_points = np.asarray(pc_pretest.points)
    pc_posttest_points = np.asarray(pc_posttest.points)
    pretest_fpfh_sig = pretest_fpfh.data.T
    posttest_fpfh_sig = posttest_fpfh.data.T
    
    corres01_idx0, corres01_idx1 = find_local_feature(pc_pretest_points, 
                                                      pc_posttest_points,
                                                      pretest_fpfh_sig,
                                                      posttest_fpfh_sig, r=r)
    print("50% complete")
    
    if not mutual_filter:
        return np.asarray([corres01_idx0, corres01_idx1]).T
    
    corres10_idx1, corres10_idx0 = find_local_feature(pc_posttest_points, 
                                                      pc_pretest_points,
                                                      posttest_fpfh_sig,
                                                      pretest_fpfh_sig, r=r)
    mutual_filter = (corres10_idx0[corres01_idx1] == corres01_idx0)
    corres_idx0 = corres01_idx0[mutual_filter]
    corres_idx1 = corres01_idx1[mutual_filter]

    return np.asarray([corres_idx0, corres_idx1]).T

def find_local_feature(points_query, points_tree, feature_query, feature_tree, r=0.1):
    """
    Helper function to find correspondences based on feature proximity within a given radius.
    
    Parameters:
    - points_query (numpy.ndarray): Query points array.
    - points_tree (numpy.ndarray): Tree points array (will be used to create a spatial KD-tree).
    - feature_query (numpy.ndarray): Query feature array corresponding to points_query.
    - feature_tree (numpy.ndarray): Tree feature array corresponding to points_tree.
    - r (float): Radius within which to search for neighbours.

    Returns:
    - tuple: Two arrays (idx0, idx1) where idx0 is indices of query points and idx1 is indices of their corresponding points in tree.
    """
    tree = cKDTree(points_tree)
    corres_idx0 = np.arange(len(points_query))  
    corres_idx1 = [] 
    
    progress_bar = tqdm(total=len(points_query), unit="iteration", position=0, leave=False, dynamic_ncols=True)

    for i in range(len(points_query)):
        dis_nns = tree.query_ball_point(points_query[i, :], r, return_sorted=None, return_length=False)
        
        if not dis_nns:
            # Append index 0 if no correspondence found within the radius (this mismatch will be filtered out in the subsequent process)
            corres_idx1.append(0)  
        else:
            local_region_features = feature_tree[dis_nns, :]
            local_nns = find_knn_cpu(feature_query[i, :], local_region_features, knn=1, return_distance=False)
            corres_idx1.append(dis_nns[local_nns])
        
        progress_bar.update(1)
    progress_bar.close()
    
    return corres_idx0, np.asarray(corres_idx1)

