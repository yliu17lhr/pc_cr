#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 11:53:53 2024
@author: yiyanliu

This module provides functions for visualising crack detection and measurement results of PC-Cr, 
showing identified feature point clusters based on kinematic properties as well as visualising identified cracks
and their measurements.

"""
import numpy as np
import open3d as o3d
import matplotlib
from matplotlib import pyplot as plt
import matplotlib.patches as patches
from matplotlib import cm
import matplotlib.colors as mcolors

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = 'Times New Roman'

def draw_crack_clustering(cloud, analysis_res, b_frame, sc_name, rec_num=None, figsize=(8, 6),
                          draw_clustering=True, draw_rec=False, coloured_cloud=False,
                          axis_on=True, savefig=False, file_name=""):
    """
    Visualises crack clustering on a point cloud.

    Args:
        cloud (open3d.geometry.PointCloud): The point cloud to visualise.
        analysis_res (dict): Analysis results containing crack and clustering information.
        b_frame (pandas.DataFrame): The boundary frame for each structural component.
        sc_name (str): The name of the structural component.
        rec_num (list of int, optional): Specific records to visualise. Defaults to None, meaning all records.
        figsize (tuple, optional): Figure size for the plot. Defaults to (8, 6).
        draw_clustering (bool, optional): Whether to draw clustering information. Defaults to True.
        draw_rec (bool, optional): Whether to draw the rectangular boundary. Defaults to False.
        coloured_cloud (bool, optional): Whether to use colours from the point cloud. Defaults to False.
        axis_on (bool, optional): Whether to display axis. Defaults to True.
        savefig (bool, optional): Whether to save the figure. Defaults to False.
        file_name (str, optional): File name to save the figure. Defaults to "".

    Returns:
        None
    """
    matplotlib.use('TkAgg')
    fig, ax = plt.subplots(figsize=figsize, dpi=100)
    coords = np.asarray(cloud.points)[:, 1:]

    return_colored_rectangle_box(ax, None, b_frame.loc[sc_name], linewidth=1.5, color=None, edgecolor='grey')

    if coloured_cloud:
        points = np.asarray(cloud.points)
        colours = np.asarray(cloud.colors)
        plt.scatter(points[:, 1], points[:, 2], c=colours, s=2)

    rec_nums = list(analysis_res[sc_name].keys()) if rec_num is None else rec_num

    for rec_num in rec_nums:
        if analysis_res[sc_name][rec_num]["crack_detection"] is not None:
            intersection = analysis_res[sc_name][rec_num]["crack_detection"]
            plt.plot([intersection[0][0], intersection[1][0]], [intersection[0][1], intersection[1][1]], color='k', linewidth=5)

            if draw_clustering:
                corrs = analysis_res[sc_name][rec_num]["main_filter"][:, 1]
                for i, (cluster, color) in enumerate(zip(["cluster_0", "cluster_1"], ["r", 'b'])):
                    inliers = analysis_res[sc_name][rec_num]["crack_clustering"][cluster]['main']["inliers"]
                    plt.scatter(coords[corrs, 0][inliers], coords[corrs, 1][inliers], s=20, c=color, label="Cluster " + str(i))

            if draw_rec:
                x_min, x_max, y_min, y_max = analysis_res[sc_name][rec_num]["division"]
                rec_patch = patches.Rectangle((x_min, y_min), x_max-x_min, y_max-y_min, linewidth=2, edgecolor='k', facecolor='none')
                ax.add_patch(rec_patch)
        else:
            if draw_clustering:
                corrs = analysis_res[sc_name][rec_num]["main_filter"][:, 1]
                for i, cluster in enumerate(["cluster_0"]):
                    inliers = analysis_res[sc_name][rec_num]["crack_clustering"][cluster]['main']["inliers"]
                    plt.scatter(coords[corrs, 0][inliers], coords[corrs, 1][inliers], s=10, label="Cluster " + str(i))

            if draw_rec:
                x_min, x_max, y_min, y_max = analysis_res[sc_name][rec_num]["division"]
                rec_patch = patches.Rectangle((x_min, y_min), x_max-x_min, y_max-y_min, linewidth=2, edgecolor='k', facecolor='none')
                ax.add_patch(rec_patch)

    plt.xlabel('X (m)', fontsize=26)
    plt.ylabel('Y (m)', fontsize=26)
    plt.xticks(fontsize=22, rotation=0)
    plt.yticks(fontsize=22, rotation=0)
    plt.axis('equal')

    if draw_rec:
        legend_rect = patches.Rectangle((0, 0), 1, 1, facecolor="none", edgecolor="k")
        legend_scatter1 = plt.Line2D([0], [0], marker='o', color='w', label='Cluster 0', markersize=10, markerfacecolor='red')
        legend_scatter2 = plt.Line2D([0], [0], marker='o', color='w', label='Cluster 1', markersize=10, markerfacecolor='blue')
        ax.legend(handles=[legend_rect, legend_scatter1, legend_scatter2], labels=['Rolling Window', "Cluster 0", "Cluster 1"],
                  loc="upper left", bbox_to_anchor=(0.5, 1.05), prop={'size': 26})

    if not axis_on:
        plt.axis("off")

    plt.tight_layout()

    if savefig:
        plt.savefig(file_name)
        plt.show()
        plt.close()
    else:
        plt.show()
        plt.close()

def draw_crack(cloud, b_frame, sc_name, crack_intersection, crack_path, crack_measurements=None,
               figsize=(8, 6), coloured_cloud=False, max_intersection_dis=0.5,
               show_crack_width=False, show_crack_slide=False, axis_on=True, savefig=False, file_name=""):
    """
    Visualises the identified crack path on a point cloud.

    Args:
        cloud (open3d.geometry.PointCloud): The point cloud to visualise.
        b_frame (pandas.DataFrame): The boundary frame for the structural component.
        sc_name (str): The name of the structural component.
        crack_intersection (list): List of points where the crack intersects.
        crack_path (list): The path of the crack.
        crack_measurements (list, optional): Measurements of crack width or slide. Defaults to None.
        figsize (tuple, optional): Figure size for the plot. Defaults to (8, 6).
        coloured_cloud (bool, optional): Whether to use colours from the point cloud. Defaults to False.
        max_intersection_dis (float, optional): Maximum distance for plotting intersections. Defaults to 0.5.
        show_crack_width (bool, optional): Whether to display crack width. Defaults to False.
        show_crack_slide (bool, optional): Whether to display crack slide. Defaults to False.
        axis_on (bool, optional): Whether to display axis. Defaults to True.
        savefig (bool, optional): Whether to save the figure. Defaults to False.
        file_name (str, optional): File name to save the figure. Defaults to "".

    Returns:
        None
    """
    matplotlib.use('TkAgg')
    fig, ax = plt.subplots(figsize=figsize, dpi=100)

    if coloured_cloud:
        points = np.asarray(cloud.points)
        colours = np.asarray(cloud.colors)
        plt.scatter(points[:, 1], points[:, 2], c=colours, s=2)
    else:
        return_colored_rectangle_box(ax, None, b_frame.loc[sc_name], linewidth=1.5, color=None, edgecolor='grey')

    crack_points = np.asarray(crack_intersection)[crack_path, :]

    if not show_crack_width and not show_crack_slide:
        plt.plot(crack_points[:, 0], crack_points[:, 1], color='r', linewidth=5, label="Identified crack")
    else:
        crack_bounds = [0, 0.5, 2.5, 7.5, 12.5, 25]
        amber_colour = (1.0, 0.75, 0.0)
        green_colour = (0, 1, 0)
        colours = ["white", 'blue', green_colour, amber_colour, 'red']

        cmap = mcolors.ListedColormap(colours)
        norm = mcolors.BoundaryNorm(crack_bounds, cmap.N)
        crack_records = np.asarray(crack_measurements)[crack_path, :]

        for i in range(crack_points.shape[0] - 1):
            x_array = np.linspace(crack_points[i, 0], crack_points[i + 1, 0], 100)
            y_array = np.linspace(crack_points[i, 1], crack_points[i + 1, 1], 100)

            if x_array.max() - x_array.min() < max_intersection_dis and y_array.max() - y_array.min() < max_intersection_dis:
                crack_array = np.linspace(crack_records[i, 0], crack_records[i + 1, 0], 100) if show_crack_width else np.linspace(crack_records[i, 0], crack_records[i + 1, 1], 100)
                plt.scatter(x_array, y_array, color=cmap(norm(crack_array * 1000)), s=50, label="Identified crack")
                line, = plt.plot([], [], linewidth=5, linestyle='-', color=cmap(norm(max(crack_array * 1000))))
                plt.legend(handles=[line], labels=["Identified crack"], loc="upper left", bbox_to_anchor=(0.5, 1.05), prop={'size': 26})

        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        cbar = plt.colorbar(sm, ax=ax, fraction=0.046, orientation="horizontal", pad=0.1)
        cbar.set_label("crack size (mm)", fontsize=26)
        cbar.ax.tick_params(labelsize=24)

    plt.xlabel('X (m)', fontsize=26)
    plt.ylabel('Y (m)', fontsize=26)
    plt.xticks(fontsize=22, rotation=0)
    plt.yticks(fontsize=22, rotation=0)
    plt.axis('equal')

    if not axis_on:
        plt.axis("off")

    plt.tight_layout()

    if savefig:
        plt.savefig(file_name)
        plt.show()
        plt.close()
    else:
        plt.show()
        plt.close()
        
def draw_correspondence(source_cloud, target_cloud, source_corrs, target_corrs):
    """
    Visualises the correspondences between two point clouds by drawing lines between corresponding points.

    Parameters:
        source_cloud (open3d.geometry.PointCloud): The source point cloud.
        target_cloud (open3d.geometry.PointCloud): The target point cloud.
        source_corrs (numpy.array): Indices of corresponding points in the source cloud.
        target_corrs (numpy.array): Indices of corresponding points in the target cloud.

    Returns:
       
        None: This function visualises the correspondences.
    """
    
    source_cloud_offset = o3d.geometry.PointCloud()
    
    source_points = np.asarray(source_cloud.points)
    target_points = np.asarray(target_cloud.points)
    
    offset_source_corr_points = source_points[source_corrs] + np.array([1, 0, 0])
    target_corr_points = target_points[target_corrs]
    
    num_correspondences = len(source_corrs)
    print(f'FPFH generates {num_correspondences} putative correspondences.')
    
    combined_points = np.concatenate((offset_source_corr_points, target_corr_points), axis=0)
    correspondence_lines = [[i, i + num_correspondences] for i in range(num_correspondences)]
    line_colours = [[0, 1, 0] for _ in range(len(correspondence_lines))]
    
    correspondence_line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(combined_points),
        lines=o3d.utility.Vector2iVector(correspondence_lines))
    correspondence_line_set.colors = o3d.utility.Vector3dVector(line_colours)
    
    source_cloud_offset.points = o3d.utility.Vector3dVector(source_points + np.array([1, 0, 0]))
    source_cloud_offset.colors = o3d.utility.Vector3dVector(np.asarray(source_cloud.colors))
    
    o3d.visualization.draw_geometries([source_cloud_offset, target_cloud, correspondence_line_set])


def return_colored_rectangle_box(ax, index, row, linewidth=1.5, color=None, edgecolor='grey'):
    """
    Draws a coloured rectangle on the given axes.

    Args:
        ax (matplotlib.axes.Axes): The axes on which to draw the rectangle.
        index (int or None): Index (Legacy placeholder).
        row (list or tuple): Contains [x_min, x_max, y_min, y_max] defining the rectangle bounds.
        linewidth (float, optional): Width of the rectangle edges. Defaults to 1.5.
        colour (str or None, optional): Fill colour of the rectangle. Defaults to None.
        edgecolour (str, optional): Edge colour of the rectangle. Defaults to 'grey'.

    Returns:
        matplotlib.patches.Rectangle: The rectangle patch added to the axes.
    """
    x_min, x_max, y_min, y_max = row
    height = y_max - y_min
    width = x_max - x_min
    rectangle = patches.Rectangle((x_min, y_min), width, height, color=color, linewidth=linewidth, facecolor='none', edgecolor=edgecolor)
    return ax.add_patch(rectangle)

 