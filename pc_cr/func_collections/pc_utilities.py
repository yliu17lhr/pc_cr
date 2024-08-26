#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 16:46:53 2024
@author: yiyanliu

point cloud utilities module

This module provides utility functions for point cloud processing, including 
boundary finding, imetrum like DataFrame construction for the use of legacy DIC functions, and correspondence filtering
using conventional and KL filters.
"""

import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.spatial import cKDTree

def find_boundary(target_IDs, alignedcoords_df, offset=None):
    """
    Calculate the boundary of a structural component defined by its DIC target coordinates,
    adjusted by the given offsets.
    
    Parameters:
    - target_IDs (array-like): Indices or identifiers for target points.
    - alignedcoords_df (DataFrame): DataFrame containing coordinates with 'Target_ID', 'X', and 'Y' columns.
    - offset (dict, optional): Dictionary specifying offsets to apply to the boundary in each direction.
      Defaults to zero offsets for all directions if not provided.
    
    Returns:
    - tuple: (x_max, x_min, y_max, y_min) boundary coordinates adjusted by the given offsets.
    """
    if offset is None:
        offset = {"x_max_offset": 0, "x_min_offset": 0, "y_max_offset": 0, "y_min_offset": 0}

    ele_coords = alignedcoords_df.set_index("Target_ID").loc[target_IDs][["X", "Y"]].values
    x_max, x_min = ele_coords[:, 0].max() + offset["x_max_offset"], ele_coords[:, 0].min() + offset["x_min_offset"]
    y_max, y_min = ele_coords[:, 1].max() + offset["y_max_offset"], ele_coords[:, 1].min() + offset["y_min_offset"]

    return x_max, x_min, y_max, y_min

def construct_coords_frame(point_coords, target_ids=None):
    """
    Construct a Imetrum DIC-like DataFrame suitable to store coordinates of targets. 
    
    Parameters:
    - point_coords (numpy.ndarray): Coordinates array of shape (n_points, 2).
    - target_ids (array-like, optional): Identifiers for the points. Defaults to sequential numbers if None.
    
    Returns:
    - DataFrame: DataFrame with columns ['Target_ID', '', '', '', 'X', 'Y'].
    """
    data_array = np.zeros((point_coords.shape[0], 6))
    data_array[:, 0] = np.arange(point_coords.shape[0]) if target_ids is None else target_ids
    data_array[:, 4:6] = point_coords
    point_frame = pd.DataFrame(data_array, columns=["Target_ID", '', '', '', 'X', 'Y'])
    point_frame['Target_ID'] = point_frame['Target_ID'].astype(int)

    return point_frame

def construct_dis_frame(u_array, target_ids=None):
    """
    Construct a Imetrum DIC like DataFrame for dsiplacement time history in a structured format.
    
    Parameters:
    - u_array (numpy.ndarray): Displacement array.
    - target_ids (array-like, optional): Identifiers for the points. Defaults to sequential numbers if None.
    
    Returns:
    - DataFrame: MultiIndex DataFrame with displacement vectors categorised by point ID and direction ('X', 'Y').
    """
    if target_ids is None:
        target_ids = np.arange(int(u_array.shape[0]/2))

    num_points = len(target_ids)
    iterables = [["Displacement " + str(i) for i in range(num_points)], ["X", "Y"]]
    multi_index = pd.MultiIndex.from_product(iterables)

    dis_frame = pd.DataFrame(u_array.reshape(1, -1), columns=multi_index)
    
    return dis_frame

def feature_corrs_in_boundary(source_cloud, source_corrs, target_corrs, alignedcoords_df, target_IDs, 
                              offset={"x_max_offset": 0, "x_min_offset": 0, 
                                      "y_max_offset": 0, "y_min_offset": 0}, 
                              boundary=None, returnmask=False):
    """
    Filters feature correspondences to those within a specified boundary.

    This function checks which of the feature correspondences between a source and target cloud 
    lie within a specified boundary based on source cloud. It applies optional offsets to the boundary and returns 
    either the filtered correspondences or a boolean mask indicating the correspondences within the boundary.

    Parameters:
        source_cloud (open3d.geometry.PointCloud): The source point cloud.
        source_corrs (np.ndarray): Indices of corresponding points in the source cloud.
        target_corrs (np.ndarray): Indices of corresponding points in the target cloud.
        alignedcoords_df (pandas.DataFrame): DataFrame containing the aligned coordinates.
        target_IDs (list or np.ndarray): List of target IDs corresponding to the aligned coordinates.
        offset (dict): Dictionary specifying offsets for the boundary in x and y directions.
                       Contains keys: "x_max_offset", "x_min_offset", "y_max_offset", "y_min_offset".
        boundary (tuple, optional): Tuple specifying the boundary in the form (x_min, x_max, y_min, y_max).
                                    If None, the boundary will be calculated using the target IDs.
        returnmask (bool, optional): If True, returns a boolean mask instead of the filtered correspondences.
                                     Default is False.

    Returns:
        If returnmask is True:
            np.ndarray: A boolean mask indicating the points within the boundary.
        Otherwise:
            np.ndarray: Indices of corresponding points in the source cloud within the boundary.
            np.ndarray: Indices of corresponding points in the target cloud within the boundary.
    """
    
    source_points = np.asarray(source_cloud.points)
    
    if boundary is None:
        x_max, x_min, y_max, y_min = find_boundary(target_IDs, alignedcoords_df, offset=offset)
    else:
        x_min, x_max, y_min, y_max = boundary

    boundary_mask = np.ones(source_corrs.shape[0], dtype=bool)

    for max_val, axis in zip([x_max, y_max], [1, 2]):
        local_mask = source_points[source_corrs, axis] < max_val
        boundary_mask = np.logical_and(boundary_mask, local_mask)

    for min_val, axis in zip([x_min, y_min], [1, 2]):
        local_mask = source_points[source_corrs, axis] > min_val
        boundary_mask = np.logical_and(boundary_mask, local_mask)

    source_corrs_b = source_corrs[boundary_mask.flatten()]
    target_corrs_b = target_corrs[boundary_mask.flatten()]
    
    if returnmask:
        return boundary_mask.flatten()
    else:
        return source_corrs_b, target_corrs_b


def KL_filter(feature, source_corrs, target_corrs, alpha=1, scale='normal'):
    """
    Filter correspondences using Kullback-Leibler (KL) divergence.

    Parameters:
        feature (numpy.ndarray): Input feature data.
        source_corrs (numpy.ndarray): Source correspondences.
        target_corrs (numpy.ndarray): Target correspondences.
        alpha (float, optional): Threshold scaling factor. Default is 1.
        scale (str, optional): Scaling method for KL divergence. Options are 'normal' or 'log'. Default is 'normal'.

    Returns:
        tuple: Filtered source and target correspondences along with their associated KL divergences.
    """
    selected_feature_data = feature.data[:, source_corrs]

    mean_hist = selected_feature_data.mean(axis=1)

    if scale == 'normal':
        KL = stats.entropy(selected_feature_data, mean_hist.reshape(-1, 1))
    elif scale == 'log':
        KL = np.log(stats.entropy(selected_feature_data, mean_hist.reshape(-1, 1)))

    KL = np.nan_to_num(KL)

    up_threshold = KL.mean() + np.std(KL) * alpha

    mask = (KL > up_threshold)

    return source_corrs[mask], target_corrs[mask], KL

def correspondence_filter(source_cloud, target_cloud, source_corrs, target_corrs,
                           distance_threshold=0.1, normal_threshold=0.9, outplane_threshold=0.002):
    """
    Filter correspondences based on distance, normal, and outplane thresholds (conventional filter).

    Parameters:
        source_cloud (PointCloud): Source point cloud.
        target_cloud (PointCloud): Target point cloud.
        source_corrs (numpy.ndarray): Source correspondences.
        target_corrs (numpy.ndarray): Target correspondences.
        distance_threshold (float, optional): Maximum distance threshold. Default is 0.1.
        normal_threshold (float, optional): Minimum normal threshold. Default is 0.9.
        outplane_threshold (float, optional): Outplane threshold. Default is 0.002.

    Returns:
        tuple: Filtered source and target correspondences.
    """
    source_points = np.asarray(source_cloud.points)
    target_points = np.asarray(target_cloud.points)

    source_normals = np.asarray(source_cloud.normals)
    target_normals = np.asarray(target_cloud.normals)

    index_array = np.zeros((source_points.shape[0], 2), dtype=int)
    index_array[source_corrs, 0] = source_corrs
    index_array[source_corrs, 1] = target_corrs

    current_mask = np.asarray([True]*source_corrs.shape[0])

    displacement_array = source_points[source_corrs, :] - target_points[target_corrs, :]
    total_disp = np.linalg.norm(displacement_array, 2, axis=1)
    dis_mask = total_disp < distance_threshold
    current_mask = np.logical_and(current_mask, dis_mask)

    outplane_mask = abs(displacement_array[:, 0]) < outplane_threshold
    current_mask = np.logical_and(current_mask, outplane_mask)

    normal_mask = np.einsum('ij,ij->i', source_normals[source_corrs], target_normals[target_corrs]) > normal_threshold
    current_mask = np.logical_and(current_mask, normal_mask)

    return source_corrs[current_mask], target_corrs[current_mask]

def find_r_cpu(feat0, feat1, r, num_batches=1):
    """
    Find points within a given radius in the target feature space.

    Parameters:
        feat0 (numpy.ndarray): Query points.
        feat1 (numpy.ndarray): Target feature points.
        r (float): Radius threshold.
        num_batches (int, optional): Number of batches. Default is 1.

    Returns:
        list: Indices of points within the radius.
    """
    feat1tree = cKDTree(feat1)
    nn_inds = feat1tree.query_ball_point(feat0, r, return_sorted=None, return_length=False)
    return nn_inds

def find_knn_cpu(feat0, feat1, knn=1, return_distance=False):
    """
    Find k nearest neighbors in the target feature space.

    Parameters:
        feat0 (numpy.ndarray): Query points.
        feat1 (numpy.ndarray): Target feature points.
        knn (int, optional): Number of nearest neighbors. Default is 1.
        return_distance (bool, optional): Whether to return distances. Default is False.

    Returns:
        tuple: Indices of the k nearest neighbors, and distances if return_distance is True.
    """
    feat1tree = cKDTree(feat1)
    dists, nn_inds = feat1tree.query(feat0, k=knn)
    if return_distance:
        return nn_inds, dists
    else:
        return nn_inds
    
def construct_pc_uarray(source_cloud, target_cloud, source_corrs, target_corrs, model='rigid_rec'):
    """
    Constructs a displacement array (`u_array`) representing the displacements between corresponding points 
    in the source and target point clouds.

    Args:
        source_cloud (open3d.geometry.PointCloud): The source point cloud.
        target_cloud (open3d.geometry.PointCloud): The target point cloud.
        source_corrs (numpy.ndarray): Indices of corresponding points in the source cloud.
        target_corrs (numpy.ndarray): Indices of corresponding points in the target cloud.
        model (str, optional): The model type used to process the displacement array. Defaults to 'rigid_rec'.

    Returns:
        numpy.ndarray: The processed displacement array (`u_array`).
    """
    feature_u_array = (np.asarray(target_cloud.points)[target_corrs]
                       - np.asarray(source_cloud.points)[source_corrs])[:, 1:].flatten()
    
    u_array = u_array_wrapper(feature_u_array, model=model)
    
    return u_array

def u_array_wrapper(feature_u_array, model='rigid_rec'):
    """
    Wraps the feature displacement array based on the specified model.

    Args:
        feature_u_array (numpy.ndarray): The raw displacement array.
        model (str, optional): The model type used to process the displacement array. 
                               Defaults to 'rigid_rec'. Special handling is applied for 'v_beam'.

    Returns:
        numpy.ndarray: The processed displacement array (`u_array`).
    """
    if model == "v_beam":
        u_array = np.zeros((feature_u_array.shape[0], 1))
        u_array[0::2] = np.roll(feature_u_array.flatten(), -1)[0::2].reshape(-1, 1)
        u_array[1::2] = np.roll(feature_u_array.flatten(), 1)[1::2].reshape(-1, 1)
    else:
        u_array = feature_u_array
        
    return u_array

def construct_b_frame(strain_elements_dict, coords_df):
    """
    Constructs a boundary frame DataFrame for each structural component based on the 
    given a dictionary of elements (targets) and their coordinates.

    Args:
        strain_elements_dict (dict): A dictionary where keys are structural component identifiers 
                                     and values are lists of element indices.
        coords_df (pandas.DataFrame): A DataFrame containing the coordinates of the elements.

    Returns:
        pandas.DataFrame: A DataFrame containing the boundary frame for each structural component, 
                          with columns ['x_min', 'x_max', 'y_min', 'y_max'].
    """
    num_sc = len(strain_elements_dict)
    
    boundary_frame = pd.DataFrame(np.zeros((num_sc, 4)), columns=["x_min", "x_max", "y_min", "y_max"])
    
    for i, key in enumerate(strain_elements_dict.keys()):
        element_index = strain_elements_dict[key]
        x_max, x_min, y_max, y_min = find_boundary(element_index, coords_df)
        boundary_frame.iloc[i, :] = (x_min, x_max, y_min, y_max)
        
    boundary_frame.index = list(strain_elements_dict.keys())
    
    return boundary_frame

def generate_B(element_index, coords, model='rigid_rec'):
    """    
    This function acts as a wrapper that selects the appropriate B-matrix construction
    function based on the specified model. The B-matrix is a component in 
    strain calculation, relating strains to displacements.

    Parameters:
        element_index (int): The index of the structural element for which the matrix is generated.
        coords (np.ndarray): The coordinates of the nodes or points within the structural element.
        model (str): Specifies the model used to describe the element behaviour. Options include
                     'rec' (rectangular), 'f_rec' (full rectangular), 'rigid_rec' (rigid rectangular),
                     'quad' (quadratic), 'h_beam' (horizontal beam), and 'v_beam' (vertical beam).
                     ***************************************************
                     all other model options removed except rigid_rec as they are not relavent in PC-Cr. they were used
                     for different project on strain analysis.
                     ***************************************************

    Returns:
        np.ndarray: The B-matrix corresponding to the specified model.
    """
    if model == "rigid_rec":
            B_matrix = construct_Bmatrix_rigid_rec(element_index, coords)
    
    else:
        print('invalid model specified !')

    return B_matrix


def construct_Bmatrix_rigid_rec(element_index, coords):
    """
    Constructs a B-matrix for rigid rectangular elements, considering only rigid body motions
    and excluding any deformations or strains.

    Parameters:
        element_index (list): Indices of the nodes or elements.
        coords (pd.DataFrame): DataFrame containing 'X' and 'Y' coordinates of nodes.

    Returns:
        numpy.ndarray: The constructed B-matrix of shape (2*len(element_index), 3).
    """
    coords_data = coords.set_index('Target_ID').loc[element_index, ["X", "Y"]].values
    x, y = coords_data[:, 0], coords_data[:, 1]
    B = np.zeros((2 * len(coords_data), 3))
    
    for i in range(len(coords_data)):
        B[2 * i] = [1, 0, -y[i]]
        B[2 * i + 1] = [0, 1, x[i]]

    return B
