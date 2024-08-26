import os

general_parameters = {
    'ceiling_multiplier': {ceiling_multiplier},
    'local_region_search_radius': {local_region_search_radius},
    'radius_normal_multiplier': {radius_normal_multiplier},
    'radius_feature_multiplier': {radius_feature_multiplier},
    'rotation_angle': 0
}

folder_names = {
    "processed_cloud": os.path.join("corr_results", "downsized_clouds"),
    "FPFH": os.path.join("corr_results", "FPFH_features"),
    "corrs": os.path.join("corr_results", "correspondence_set")
}

sc_name = '{structural_component_name}'
