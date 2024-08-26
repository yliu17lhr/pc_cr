import os

data_file_dirs = {"gen_data": os.path.join("{base_dir}", "analysis_config_data"),
                  "downsized_cloud": os.path.join("{base_dir}", "corr_results/downsized_clouds"),
                  "FPFH_feature": os.path.join("{base_dir}", "corr_results/FPFH_features"),
                  "registration_results": os.path.join("{base_dir}", "feature_registration_results.pickle"),
                  "registration_metadata": os.path.join("{base_dir}", "feature_registration_metadata.json"),
                  "analysis_results": os.path.join("{base_dir}", "Crack_results")}

gen_data_names = ["plot_coords","plot_elements_dict"]


sc_offsets = {"{structural_component_name}": {"x_max_offset": {x_max_offset}, "x_min_offset": {x_min_offset},
                                              "y_max_offset": {y_max_offset}, "y_min_offset": {y_min_offset}}}

sc_divisions = {"{structural_component_name}": {"m": {number_of_rolling_window_horizontal},
                                                "n": {number_of_rolling_window_vertical}}}


global_analysis_paras = {"KL_alpha": {kl_alpha}, "KL_scale": '{kl_scale}', "dis_filter": {distance_filter},
                         "normal_filter": {normal_filter}, "outplane_filter": {out-of-plane_filter}}


def f_ran_threshold(voxel_size):

    return voxel_size * 1.2


def m_ran_threshold(voxel_size):

    return voxel_size * 1


ran_crack_paras = {"f_model": 'rigid_rec', "f_num_samples": 2, "f_ransac_threshold": f_ran_threshold,
                   "f_iterations": 20000, "f_draw_method": "random", "f_score_method": "inliers",
                   "m_model": 'rigid_rec', "m_num_samples": 2, "m_ransac_threshold": m_ran_threshold,
                   "m_iterations": 20000, "m_draw_method": "random", "m_score_method": "quality",
                   "maximum_cluster_num": 2, "min_centroid_dist": {minimum_centroid_distance},
                   "min_differential_disp_multiplier": {minimum_differential_displacement_multiplier}}
