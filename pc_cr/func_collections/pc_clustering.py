#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 12:03:17 2024
@author: yiyanliu

Point cloud clustering module

This module provides functions for RANSAC-based displacement filtering utilising various kinematic models. 
For the current PC-Cr method, only rigid rectangular (rigid_rec) model implementation is kept for clarity.

It is specifically designed to identify and eliminate outliers that do not adhere to the expected kinematic 
behaviours prescribed by the selected models. The primary implementations within this module derive from 
the methodologies detailed in Chapter 6 of Liu's DPhil thesis.

"""

import numpy as np
from sklearn.cluster import KMeans
from tqdm import tqdm
from pc_cr.func_collections.pc_utilities import construct_coords_frame, construct_pc_uarray, generate_B

def ransac_processing(source_cloud, target_cloud, source_corrs, target_corrs, model='rigid_rec',
                      num_samples=15, ransac_threshold=0.0005, iterations=1000,
                      draw_method="random", score_method="inliers", minimum_inlier_percentage=0.5):
    """
    Performs RANSAC (Random Sample Consensus) processing to fit a model to the displacements of feature points
    derived by matching point features in source and target point clouds, considering various sampling and scoring strategies.
    See Section 6.2.4.2 Rigid model filter of Liu's DPhil thesis for furhter details. This process assumes after align
    
    Parameters:
        source_cloud (open3d.geometry.PointCloud): Point cloud from the source.
        target_cloud (open3d.geometry.PointCloud): Point cloud from the target.
        source_corrs (numpy.array): Indices of correspondence points in the source cloud.
        target_corrs (numpy.array): Indices of correspondence points in the target cloud.
        model (str): The model used for the RANSAC process. Default is 'rigid_rec'.
        num_samples (int): Number of samples to use in each iteration of RANSAC.
        ransac_threshold (float): The threshold distance to consider a point as an inlier.
        iterations (int): Number of iterations to run the RANSAC algorithm.
        draw_method (str): Method to select samples; options include 'random' and 'k_means' (the furthest distance method is removed for clarity).
        score_method (str): Criterion for scoring models; options are 'inliers' and 'quality'.
        minimum_inlier_percentage (float): The minimum percentage of total points that must be inliers to consider a model.

    Returns:
        dict: A dictionary containing the best model parameters, inliers, quality of the fit, and optional labels from clustering.
    """

    progress_bar = tqdm(total=iterations, unit='iterations',
                        position=0, leave=False, dynamic_ncols=True)

    source_points = np.asarray(source_cloud.points)[:, 1:]

    B_matrix, U_array = generate_BU(source_cloud, target_cloud,
                                    source_corrs, target_corrs, model=model)

    current_quality = 100
    inliers = []
    labels = None
    model_para = None  

    n_points = len(source_corrs)
    local_point_ids = np.arange(n_points)

    if draw_method == 'k_means':
        kmeans = KMeans(n_clusters=num_samples, random_state=0, n_init=10).fit(source_points[source_corrs])
        clustering_labels = kmeans.labels_

    i = 1
    while i <= iterations:
        if draw_method == 'random':
            idx_samples = np.random.choice(local_point_ids, num_samples, replace=False)

        elif draw_method == "k_means":
            idx_samples = []
            for j in range(num_samples):
                local_indices = np.where(clustering_labels == j)[0]
                idx_sample = np.random.choice(local_indices)
                idx_samples.append(idx_sample)

        else:
            print("Invalid draw method")
            break

        idx_samples = np.asarray(idx_samples)
        idx_samples_dup = 2 * np.repeat(idx_samples, 2)
        idx_samples_dup[1::2] += 1

        B_selected = B_matrix[idx_samples_dup, :]
        U_selected = U_array[idx_samples_dup]

        model_parameters = np.linalg.lstsq(B_selected, U_selected, rcond=None)
        fitted_displacements = B_matrix @ model_parameters[0]
        distance = np.linalg.norm((fitted_displacements - U_array).reshape(-1, 2), axis=1)

        idx_candidates = np.where(distance <= ransac_threshold)[0]

        if score_method == "quality":
            if len(idx_candidates) > minimum_inlier_percentage * n_points and set(idx_candidates).issuperset(set(idx_samples)):
                quality = np.sum(distance[idx_candidates]) / len(idx_candidates)
                if quality < current_quality:
                    current_quality = quality
                    inliers = idx_candidates
                    model_para = model_parameters[0]
                    if draw_method == "k_means":
                        labels = clustering_labels[idx_candidates]

        elif score_method == "inliers":
            new_inliers_len = len(idx_candidates)
            if new_inliers_len > len(inliers) and set(idx_candidates).issuperset(set(idx_samples)):
                model_para = model_parameters[0]
                inliers = idx_candidates
                current_quality = np.sum(distance[idx_candidates]) / new_inliers_len
            elif new_inliers_len == len(inliers) and set(idx_candidates).issuperset(set(idx_samples)):
                new_quality = np.sum(distance[idx_candidates]) / new_inliers_len
                if new_quality < current_quality:
                    model_para = model_parameters[0]
                    inliers = idx_candidates
                    current_quality = new_quality

        progress_bar.update(1)
        i += 1

    progress_bar.close()

    results = {"inliers": inliers, "model_para": model_para, "current_quality": current_quality, "labels": labels}
    return results

def generate_BU(source_cloud, target_cloud, source_corrs, target_corrs, model='rigid_rec'):
    """
    Generates matrices B and U used in RANSAC (and strain calculations) based on the selected model.
    
    Parameters:
        source_cloud (open3d.geometry.PointCloud): The point cloud from the source.
        target_cloud (open3d.geometry.PointCloud): The point cloud from the target.
        source_corrs (numpy.array): Indices of correspondence points in the source cloud.
        target_corrs (numpy.array): Indices of correspondence points in the target cloud.
        model (str): The model type to use for generating matrices ('rigid_rec').
    
    Returns:
        tuple: A tuple containing:
               - B_matrix (numpy.ndarray): The matrix representing spatial derivatives.
               - u_array (numpy.ndarray): The displacement vector array.
    """
    source_points = np.asarray(source_cloud.points)[:, 1:]
    point_frame = construct_coords_frame(source_points)
    u_array = construct_pc_uarray(source_cloud, target_cloud, source_corrs, target_corrs, model=model)
    B_matrix = generate_B(source_corrs, point_frame, model=model)

    return B_matrix, u_array

