#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon April 15 15:55:25 2024
@author: yiyanliu

Point cloud crack detection module

This module contains the main function and supporting functions that perform RANSAC-based crack detection
and clustering using point cloud data. It facilitates the identification of cracks based
on the methodologies outlined in Chapter 8 of Liu's DPhil thesis. This module allows for kinematical
analysis through iterative filtering and clustering mechanisms to accurately detect and 
measure potential cracks.
"""

import numpy as np
import copy
from shapely.geometry import Polygon
from pc_cr.func_collections import pc_clustering


def ransac_crack_clustering(source_cloud, target_cloud, source_corrs, target_corrs, 
                            f_model='rigid_rec', f_num_samples=2, f_ransac_threshold=0.001, f_iterations=1000, 
                            f_draw_method="random", f_score_method='inliers', m_model='rigid_rec', m_num_samples=2,  
                            m_ransac_threshold=0.001, m_iterations=1000, m_draw_method="random", 
                            m_score_method='quality', maximum_cluster_num=2, min_centroid_dist=0.1, 
                            min_differential_disp_multiplier=0.8, min_inlier_percentage=0.3):
    """
    Implementation of the PC-Cr feature clustering and crack identification as described in Section 8.2 of Liu's thesis.

    The function iteratively detects clusters (potential cracks) based on inferred displacements between the source and target point clouds using RANSAC. 
    It stops when the maximum number of clusters is reached or the clustering criteria are not met.

    Parameters:
        source_cloud (open3d.geometry.PointCloud): The source point cloud.
        target_cloud (open3d.geometry.PointCloud): The target point cloud.
        source_corrs (np.ndarray): Indices of corresponding points in the source cloud.
        target_corrs (np.ndarray): Indices of corresponding points in the target cloud.
        f_model (str): Model to use for the filtering step in RANSAC. Default is 'rigid_rec'.
        f_num_samples (int): Number of samples to use in each draw in the filtering step. Default is 2.
        f_ransac_threshold (float): RANSAC threshold for the filtering step. Default is 0.001. This is usually set to 1.2* re-sampling voxel size.
        f_iterations (int): Number of iterations for the filtering RANSAC step. Default is 1000.
        f_draw_method (str): Method to draw samples in the filtering step. Default is 'random'.
        f_score_method (str): Method to score inliers in the filtering step. Default is 'inliers'.
        m_model (str): Model to use for the main quality based RANSAC step. Default is 'rigid_rec'.
        m_num_samples (int): Number of samples to use in the main main quality based RANSAC step. Default is 2.
        m_ransac_threshold (float): RANSAC threshold for the main step. Default is 0.001. This is usually set to the re-sampling voxel size.
        m_iterations (int): Number of iterations for the main quality based RANSAC step. Default is 1000.
        m_draw_method (str): Method to draw samples in the main quality based RANSAC step. Default is 'random'.
        m_score_method (str): Method to score inliers in the main RANSAC step. Default is 'quality', to return a set of inliers with minimum RMSE within m_ransac_threshold.
        maximum_cluster_num (int): Maximum number of clusters to process. Default is 2.
        min_centroid_dist (float): Minimum distance between centroids of clusters. Default is 0.1. This is set based in relation to the size of rolling window (25% - 50% of the shorted edge)
        min_differential_disp_multiplier (float): Minimum differential displacement multiplier. Default is 0.8. The differential displacement must be greater than 0.8*re-sampling voxel size to be considered as valid crack event.
        min_inlier_percentage (float): Minimum percentage of inliers of the entire set required to consider a valid cluster. Default is 0.3. This is to avoid rare occurance of a very samll subset with best RMSE errors. This parameter only affects the order of cluster if it is set greater than 0.5.

    Returns:
        dict: A dictionary containing the results of the RANSAC clustering for each detected crack.
    """
    
    results = {}
    source_points = np.asarray(source_cloud.points)
    all_indices = np.arange(source_corrs.shape[0])
    used_indices = []
    current_indices = copy.deepcopy(all_indices)
    cluster_centroids = []
    cluster_disps = []
    cluster_count = 0

    while cluster_count < maximum_cluster_num:
        
        print(f"Processing cluster: {cluster_count}")
        
        current_source_corrs = source_corrs[current_indices]
        current_target_corrs = target_corrs[current_indices]
        
        filter_results = pc_clustering.ransac_processing(
            source_cloud, target_cloud, current_source_corrs, current_target_corrs,
            model=f_model, num_samples=f_num_samples, ransac_threshold=f_ransac_threshold, iterations=f_iterations,
            draw_method=f_draw_method, score_method=f_score_method)
        
        current_indices = current_indices[filter_results['inliers']]
        current_source_corrs = source_corrs[current_indices]
        current_target_corrs = target_corrs[current_indices]
        
        if cluster_centroids:
            centroid = source_points[source_corrs][current_indices].mean(axis=0)
            if not all(np.linalg.norm(centroid - x) > min_centroid_dist for x in cluster_centroids):
                break
        
        res_key = f"cluster_{cluster_count}"
        results[res_key] = {}
        results[res_key]["filter"] = filter_results
        results[res_key]["filter"]["inliers"] = current_indices
                
        main_results = pc_clustering.ransac_processing(
            source_cloud, target_cloud, current_source_corrs, current_target_corrs,
            model=m_model, num_samples=m_num_samples, ransac_threshold=m_ransac_threshold, iterations=m_iterations,
            draw_method=m_draw_method, score_method=m_score_method, minimum_inlier_percentage=min_inlier_percentage)
        
        current_indices = current_indices[main_results['inliers']]
        cluster_centroid = source_points[source_corrs][current_indices].mean(axis=0)
        cluster_disp = return_rigid_model_displacements(cluster_centroid, main_results['model_para']) 
        
        if cluster_disps:
            if not all(np.linalg.norm(cluster_disp - x) > m_ransac_threshold * min_differential_disp_multiplier
                       for x in cluster_disps):
                del results[res_key]
                break
            
        radius = np.median(np.linalg.norm((source_points[current_indices] - cluster_centroid.reshape(-1, 3))[:, 1:3], axis=1))
        
        cluster_centroids.append(cluster_centroid)
        cluster_disps.append(cluster_disp)
        
        results[res_key]['main'] = main_results
        results[res_key]['main']['inliers'] = current_indices
        results[res_key]['main']['centroid'] = cluster_centroid
        results[res_key]['main']['radius'] = radius
        
        print_inliers_status(filter_results, main_results)
        
        used_indices.extend(current_indices)
        used_indices.sort()
        
        current_indices = np.setdiff1d(all_indices, used_indices)
        
        cluster_count += 1
        
    
    return results


def return_rigid_model_displacements(point_coords, rigid_model_paras, projection=False):
    """
    Calculates displacement components based on rigid body transformation parameters.
    
    Parameters:
        point_coords (numpy.ndarray): Coordinates of the point (x, y, z) for which displacement is calculated.
        rigid_model_paras (numpy.ndarray): Parameters of the rigid model (u0, v0, omega),
                                           where u0, v0 are translations and omega is the rotation.
        projection (bool): Determines if the displacement calculation should be projected in a different orientation.
                           If False, the standard orientation (original y for x, original z for y) is used. If True, it is assumed the data
                           is already 2D, thereofre original x for x and original y for y.
    
    Returns:
        numpy.ndarray: Displacement vector [u, v] for the given point under the specified model.
    """
    if not projection:
        u = rigid_model_paras[0] - point_coords[2] * rigid_model_paras[2]
        v = rigid_model_paras[1] + point_coords[1] * rigid_model_paras[2]
    else:
        u = rigid_model_paras[0] - point_coords[1] * rigid_model_paras[2]
        v = rigid_model_paras[1] + point_coords[0] * rigid_model_paras[2]
    
    return np.asarray([u, v])

def print_inliers_status(filter_results, main_results):
    """
    Prints the count of inliers identified in the filter phase and main cluestering phase of the RANSAC process.
    
    Parameters:
        filter_results (dict): Results dictionary from the filtering phase containing 'inliers'.
        main_results (dict): Results dictionary from the main RANSAC clustering containing 'inliers'.
    """
    filter_inlier_length = len(filter_results["inliers"])
    main_inlier_length = len(main_results["inliers"])
    
    print(f"Size of inliers (filter) => {filter_inlier_length} (main model) => {main_inlier_length}")

def crack_detection(clustering_results, rec_boundary):
    """
    Implementation of the crack identification method as descibed in Section 8.2.3.2 of Liu's DPhil thesis.
    
    Detects the intersection of the clustering line with a rectangular boundary and calculates the slope of the clustering line.
    
    Parameters:
        clustering_results (dict): A dictionary containing clustering results including centroids and radii of clusters.
        rec_boundary (list): A list of rectangle boundaries [xmin, xmax, ymin, ymax] defining the area of interest.
    
    Returns:
        tuple: (intersection, slope) where 'intersection' is a list of intersection points (if any)
               and 'slope' is the slope of the line connecting cluster centroids, perpendicular to which the crack is detected.
    """
    if "cluster_1" not in clustering_results:
        print('No crack detected')
        return None, None
    
    print('Crack detected')
    centroid_0 = clustering_results["cluster_0"]["main"]["centroid"]
    centroid_1 = clustering_results["cluster_1"]["main"]["centroid"]
    radius_0 = clustering_results["cluster_0"]["main"]["radius"]
    radius_1 = clustering_results["cluster_1"]["main"]["radius"]

    slope, bpoint = perpendicular_clustering_line(centroid_0, centroid_1, radius_0, radius_1)

    rec = [(rec_boundary[0], rec_boundary[2]),
           (rec_boundary[0], rec_boundary[3]),
           (rec_boundary[1], rec_boundary[3]),
           (rec_boundary[1], rec_boundary[2])]

    intersection = line_rect_intersections(bpoint, slope, rec)

    return intersection, slope

def perpendicular_clustering_line(point1, point2, radius_1, radius_2):
    """
    Calculates the midpoint and perpendicular slope between two points representing cluster centroids.
    
    Parameters:
        point1 (tuple): Coordinates of the first centroid.
        point2 (tuple): Coordinates of the second centroid.
        radius_1 (float): Radius around the first centroid.
        radius_2 (float): Radius around the second centroid.
    
    Returns:
        tuple: (perpendicular_slope, midpoint) where 'perpendicular_slope' is the slope of the line
               perpendicular to the line connecting point1 and point2, and 'midpoint' is the calculated midpoint
               between point1 and point2 considering their radii.
    """
    increment_x = (point2[1] - point1[1]) / (radius_1 + radius_2)
    increment_y = (point2[2] - point1[2]) / (radius_1 + radius_2)

    b_x = point1[1] + increment_x * radius_1
    b_y = point1[2] + increment_y * radius_1

    bpoint = (b_x, b_y)

    slope = (point2[2] - point1[2]) / (point2[1] - point1[1])
    perp_slope = -1 / slope 

    return perp_slope, bpoint

def line_rect_intersections(point, gradient, rect_points):
    """
    Calculates the intersection points of a line with a rectangular boundary.

    Parameters:
        point (tuple): A point (x, y) through which the line passes.
        gradient (float): Slope of the line.
        rect_points (list of tuples): Coordinates defining the rectangle [(x1, y1), (x2, y2), (x3, y3), (x4, y4)].

    Returns:
        list: A list of tuples representing the intersection points (x, y) of the line with the rectangle.
    """
    x_coords = [p[0] for p in rect_points]
    y_coords = [p[1] for p in rect_points]
    xmin, xmax = min(x_coords), max(x_coords)
    ymin, ymax = min(y_coords), max(y_coords)

    rect_lines = []
    for i in range(4):
        p1, p2 = rect_points[i], rect_points[(i+1) % 4]
        if p1[0] != p2[0]:
            slope = (p2[1] - p1[1]) / (p2[0] - p1[0])
            y_int = p1[1] - slope * p1[0]
            rect_lines.append((slope, y_int, p1[0], p2[0]))
        else:
            rect_lines.append((None, p1[0], min(p1[1], p2[1]), max(p1[1], p2[1])))

    intersections = []
    for line in rect_lines:
        if line[0] is not None:
            x_int = (line[1] - point[1] + gradient * point[0]) / (gradient - line[0])
            y_int = gradient * x_int + point[1] - gradient * point[0]
        else:
            x_int = line[1]
            y_int = gradient * x_int + point[1] - gradient * point[0]
            
        if xmin - 0.00001 <= x_int <= xmax + 0.00001 and ymin - 0.00001 <= y_int <= ymax + 0.00001:
            intersections.append((x_int, y_int))

    return intersections

def rectangle_division(b_frame, structural_component, m, n, 
                       offset={"x_max_offset": 0, "x_min_offset": 0, 
                               "y_max_offset": 0, "y_min_offset": 0}):
    """
    Divides a structural component into smaller overalapping rectangles for crack detection and measurements. 
    The size of overlapping is fixed to half of the width and height of the rolling window.

    The function divides the specified structural component's bounding frame into a grid of 
    smaller overalapping rectangles, with optional offsets applied to the bounding box dimensions.

    Parameters:
        b_frame (pd.DataFrame): DataFrame containing the boundary of structural components.
        structural_component (str): The name of the structural component to be divided.
        m (int): Number of divisions along the width (horizontal).
        n (int): Number of divisions along the height (vertical).
        offset (dict): Dictionary containing offsets for the bounding box edges:
                       - "x_max_offset": Offset for the maximum x-boundary.
                       - "x_min_offset": Offset for the minimum x-boundary.
                       - "y_max_offset": Offset for the maximum y-boundary.
                       - "y_min_offset": Offset for the minimum y-boundary.
                       suitable offsets are important for cracks along the boundary of structural components

    Returns:
        list: A list of tuples, each representing a rectangle in the form (x_min, x_max, y_min, y_max).
    """
    
    x_min, x_max, y_min, y_max = b_frame.loc[structural_component]

    x_min += offset["x_min_offset"]
    x_max += offset["x_max_offset"]
    y_min += offset["y_min_offset"]
    y_max += offset["y_max_offset"]

    width = x_max - x_min
    height = y_max - y_min
    
    if m == 1:
        rect_width = width
    else:
        rect_width = 2 * width / (m + 1)
        
    if n == 1:
        rect_height = height
    else:
        rect_height = 2 * height / (n + 1)
        
    step_width = rect_width / 2
    step_height = rect_height / 2
    
    rectangles = []
    
    for i in range(m):
        x = x_min + i * step_width
        for j in range(n):
            y = y_min + j * step_height
            rectangles.append((x, x + rect_width, y, y + rect_height))

    return rectangles

def crack_point_cost_matrix(points):
    """
    Calculate the cost matrix for a set of points based on the Euclidean distance
    between each pair of points.

    Args:
        points (list of tuples): A list of (x, y) coordinates representing the points.

    Returns:
        numpy.ndarray: A 2D array where each element at index [i][j] represents the
        distance between point i and point j.
    """
    n = len(points)
    cost_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                x1, y1 = points[i]
                x2, y2 = points[j]
                distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                cost_matrix[i][j] = distance
    return cost_matrix

def find_extreme_points(points):
    """
    Identify the indices of the extreme points in a list of points. Extreme points
    are those with the maximum or minimum x and y coordinates.

    Args:
        points (list of tuples): A list of (x, y) coordinates representing the points.

    Returns:
        list of int: A list of unique indices corresponding to the extreme points
        (maximum and minimum x and y coordinates).
    """
    max_x_idx = np.argmax([p[0] for p in points])
    max_y_idx = np.argmax([p[1] for p in points])
    min_x_idx = np.argmin([p[0] for p in points])
    min_y_idx = np.argmin([p[1] for p in points])
    return list(set([max_x_idx, max_y_idx, min_x_idx, min_y_idx]))

def nearest_neighbor(cost_matrix, start):
    """
    Generate a open path using the nearest neighbour heuristic without returning to start point. Starting from a given
    point, the path progresses by repeatedly visiting the nearest unvisited point.

    Args:
        cost_matrix (numpy.ndarray): A 2D array where each element represents the cost
        (distance) between two points.
        start (int): The starting point index.

    Returns:
        list of int: A list of indices representing the order of points visited in the path.
    """
    n = len(cost_matrix)
    path = [start]
    visited = set(path)
    while len(path) < n:
        last = path[-1]
        next_city = min((cost_matrix[last][j], j) for j in range(n) if j not in visited)[1]
        path.append(next_city)
        visited.add(next_city)
    return path

def calculate_cost(cost_matrix, path):
    """
    Calculate the total cost of a given path based on the provided cost matrix.

    Args:
        cost_matrix (numpy.ndarray): A 2D array where each element represents the cost
        (distance) between two points.
        path (list of int): A list of indices representing the order of points visited.

    Returns:
        float: The total cost of the path.
    """
    return sum(cost_matrix[path[i - 1]][path[i]] for i in range(1, len(path)))

def two_opt(cost_matrix, path):
    """
    Optimise a given path using the 2-opt algorithm, which iteratively swaps pairs
    of edges to reduce the total path cost.

    Args:
        cost_matrix (numpy.ndarray): A 2D array where each element represents the cost
        (distance) between two points.
        path (list of int): A list of indices representing the initial path.

    Returns:
        tuple: A tuple containing the optimised path (list of int) and the corresponding cost (float).
    """
    n = len(path)
    best_cost = calculate_cost(cost_matrix, path)
    improved = True
    while improved:
        improved = False
        for i in range(1, n - 1):
            for j in range(i + 1, n):
                new_path = path[:i] + path[i:j][::-1] + path[j:]
                new_cost = calculate_cost(cost_matrix, new_path)
                if new_cost < best_cost:
                    path = new_path
                    best_cost = new_cost
                    improved = True
    return path, best_cost

def optimal_tsp(points):
    """
    Connecting all crack points by solving the openloop travelling salesman problem (TSP) by using
    a combination of the nearest neighbour heuristic and the 2-opt algorithm. The assumption is a crack 
    always starts (or ends at) from a extreme point - there is no crack starts at a mid point and end at a mid point.
    It probably doesn't matter if we check all points as the start point but as the crack path becomes more
    complex this will be needed.

    Args:
        points (list of tuples): A list of (x, y) coordinates representing the points.

    Returns:
        tuple: A tuple containing the optimal path (list of int) and the corresponding
        minimal cost (float).
    """
    cost_matrix = crack_point_cost_matrix(points)
    extreme_points = find_extreme_points(points)
    best_overall_path = None
    best_overall_cost = float('inf')

    for start in extreme_points:
        initial_path = nearest_neighbor(cost_matrix, start)
        path, cost = two_opt(cost_matrix, initial_path)
        if cost < best_overall_cost:
            best_overall_path, best_overall_cost = path, cost

    return best_overall_path, best_overall_cost


def crack_measurement(clustering_results, intersections, slope, boundary):
    """
    Implementation of crack measurement as described in Section 8.2.4 of Liu's thesis.
    Crack measurements are derived using optimal rigid body paramters in relation to the identified crack path.
    
    Parameters:
        clustering_results (dict): Contains the centroids and model parameters for clusters detected.
        intersections (list of tuples): Points where the clustering line intersects the boundary of rolling windows.
        slope (float): Slope of the line connecting cluster centroids.
        boundary (list): Coordinates defining the boundary within which the measurements are confined. (Legacy placeholder)
    
    Returns:
        list: Measurements of the cracks including width and slide distances between transformed points.
    """
    centroids = []
    model_paras = []
    crack_measurements = []
    
    if slope == 0:
        m = np.inf
    else:
        m = -1 / slope
    
    centroids.append(clustering_results["cluster_0"]["main"]["centroid"][1:])
    centroids.append(clustering_results["cluster_1"]["main"]["centroid"][1:])
    
    model_paras.append(clustering_results["cluster_0"]["main"]["model_para"])
    model_paras.append(clustering_results["cluster_1"]["main"]["model_para"])
    
    u_00, v_00 = return_rigid_model_displacements(intersections[0], model_paras[0], projection=True)
    u_01, v_01 = return_rigid_model_displacements(intersections[0], model_paras[1], projection=True)
    u_10, v_10 = return_rigid_model_displacements(intersections[1], model_paras[0], projection=True)
    u_11, v_11 = return_rigid_model_displacements(intersections[1], model_paras[1], projection=True)
    
    B_0 = [intersections[0][0] + u_00, intersections[0][1] + v_00]
    B_1 = [intersections[0][0] + u_01, intersections[0][1] + v_01]
    C_0 = [intersections[1][0] + u_10, intersections[1][1] + v_10]
    C_1 = [intersections[1][0] + u_11, intersections[1][1] + v_11]
    
    u_00_prime, v_00_prime = rotate_point_by_slope(u_00, v_00, m)
    u_01_prime, v_01_prime = rotate_point_by_slope(u_01, v_01, m)
    u_10_prime, v_10_prime = rotate_point_by_slope(u_10, v_10, m)
    u_11_prime, v_11_prime = rotate_point_by_slope(u_11, v_11, m)
    
    triangle_ABC = [centroids[0], B_0, C_0]
    triangle_BCD = [centroids[1], B_1, C_1]
    
    polygon1 = Polygon(triangle_ABC)
    polygon2 = Polygon(triangle_BCD)
    intersection = polygon1.intersection(polygon2)
    
    if intersection.is_empty:
        print("No overlapping area")
        
        width_0 = abs(u_01_prime - u_00_prime)
        slide_0 = abs(v_01_prime - v_00_prime)
        crack_measurements.append((width_0, slide_0))
        
        width_1 = abs(u_11_prime - u_10_prime)
        slide_1 = abs(v_11_prime - v_10_prime)
        crack_measurements.append((width_1, slide_1))
        
    else:
        print("Overlap area:", intersection.area)
        centroid = intersection.centroid
        distance_to_B = np.linalg.norm([centroid.x - intersections[0][0], centroid.y - intersections[0][1]])
        distance_to_C = np.linalg.norm([centroid.x - intersections[1][0], centroid.y - intersections[1][1]])
        
        if distance_to_B <= distance_to_C:
            width_0 = 0
            slide_0 = abs(v_01_prime - v_00_prime)
            crack_measurements.append((width_0, slide_0))
            width_1 = abs(u_11_prime - u_10_prime)
            slide_1 = abs(v_11_prime - v_10_prime)
            crack_measurements.append((width_1, slide_1))
        else:
            width_0 = abs(u_01_prime - u_00_prime)
            slide_0 = abs(v_01_prime - v_00_prime)
            crack_measurements.append((width_0, slide_0))
            width_1 = 0
            slide_1 = abs(v_11_prime - v_10_prime)
            crack_measurements.append((width_1, slide_1))
    
    return crack_measurements


def rotate_point_by_slope(x, y, m):
    """
    Rotates a point (x, y) by an angle derived from a given slope 'm'.

    Parameters:
        x (float): The x-coordinate of the point.
        y (float): The y-coordinate of the point.
        m (float): The slope from which the angle of rotation is derived.
    
    Returns:
        tuple: The new coordinates (x', y') of the point after rotation.
    """
    if m == 0:
        return x, y
    elif np.isinf(m):
        return -y, x
    
    theta_radians = np.arctan(m)
    x_prime = x * np.cos(theta_radians) - y * np.sin(theta_radians)
    y_prime = x * np.sin(theta_radians) + y * np.cos(theta_radians)
    return x_prime, y_prime

