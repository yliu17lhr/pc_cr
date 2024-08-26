#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 22:37:19 2024

@author: yiyanliu

point cloud pre process module

This module provides functionality for processing point clouds: loading from text files,
applying geometric transformations, and computing features for registration purposes.
It utilises PCA for alignment corrections and Open3D for point cloud manipulation.
"""

from sklearn.decomposition import PCA
import numpy as np
import open3d as o3d


def load_pointcloud_fromtxt(file, rowstoskip=1, align_angle=0, visualisation=False):
    """
    Load a point cloud from a text file, apply alignment rotation, and optionally visualise it.

    Parameters:
    - file (str): Path to the text file containing point cloud data.
    - rowstoskip (int): Number of rows to skip at the beginning of the file; default is 1.
    - align_angle (float): Rotation angle in radians to align the point cloud; default is 0.
    - visualisation (bool): If True, visualises the point cloud using Open3D viewer; default is False.

    Returns:
    - o3d.geometry.PointCloud: The loaded and aligned point cloud.
    """
    pc_array = np.loadtxt(file, delimiter=',', skiprows=rowstoskip)

    pcd = o3d.geometry.PointCloud()

    xyz = np.zeros((pc_array.shape[0], 3))
    xyz[:, 0] = pc_array[:, 0]
    xyz[:, 1] = pc_array[:, 1]
    xyz[:, 2] = pc_array[:, 2]

    rotation_matrix = np.array([[np.cos(align_angle), -np.sin(align_angle)],
                                [np.sin(align_angle),  np.cos(align_angle)]])
    xyz[:, 0:2] = np.dot(xyz[:, 0:2], rotation_matrix)

    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(pc_array[:, 3:6]/255)

    if visualisation:
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        vis.add_geometry(pcd)
        vis.get_render_option().show_coordinate_frame = True
        vis.poll_events()
        vis.update_renderer()
        vis.run()

    return pcd


def find_coordframe_angle(reference_pc_array):
    """
    Calculate the angle for the coordinate frame alignment based on PCA.

    Parameters:
    - reference_pc_array (numpy.ndarray): The reference point cloud data array.

    Returns:
    - float: The rotation angle in radians needed to align the point cloud along the Y-axis.
    """
    X = reference_pc_array[:, :2]
    pca = PCA(n_components=2)
    pca.fit(X)

    cross_product = np.linalg.norm(np.cross(pca.components_[0], np.array([0, 1])))
    dot_product = np.dot(pca.components_[0], np.array([0, 1]))

    return -np.arctan2(cross_product, dot_product)


def preprocess_point_cloud(pcd, voxel_size, radius_normal_multiplier=3,
                           radius_feature_multiplier=5, cal_fpfh=True):
    """
    Preprocess a point cloud by downsampling, estimating normals, and optionally computing FPFH features.

    Parameters:
    - pcd (o3d.geometry.PointCloud): The input point cloud.
    - voxel_size (float): The voxel size for downsampling.
    - radius_normal_multiplier (float): Multiplier to define the radius for normal estimation.
    - radius_feature_multiplier (float): Multiplier to define the radius for FPFH feature computation.
    - cal_fpfh (bool): Whether to compute FPFH features; default is True.

    Returns:
    - Tuple: Contains the downsampled point cloud and, if requested, the FPFH features.
    """
    pcd_down = pcd.voxel_down_sample(voxel_size)

    radius_normal = voxel_size * radius_normal_multiplier
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=300))

    if cal_fpfh:
        radius_feature = voxel_size * radius_feature_multiplier
        pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            pcd_down,
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=1000))
        return pcd_down, pcd_fpfh

    return pcd_down
