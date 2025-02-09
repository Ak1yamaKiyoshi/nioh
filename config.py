import os

class Config:
    prefix = "dataset"
    path_sensors_dataset = f"{prefix}/transition_3_sensors"
    path_images_dataset = f"{prefix}/transition_3_nav_cam"

    col_t = "t"

    path_image_timestamps_csv = os.path.join(path_images_dataset, "nav_cam_timestamps.csv")

    path_odometry_csv = os.path.join(path_sensors_dataset, "rs_odom.csv")
    col_odom_z = " p_z"

    path_lrf_range_csv = os.path.join(path_sensors_dataset, "lrf_range.csv")
    col_lrf_range = " range"