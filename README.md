run `preproc.py`


roslaunch realsense2_camera rs_camera.launch enable_pointcloud:=true depth_width:=848 depth_height:=480 color_width:=848 color_height:=480



elenoide@elepc:~rosrun tf tf_echo base_footprint camera_depth_optical_frame
At time 1664480625.072
- Translation: [-0.220, -0.650, 1.080]
- Rotation: in Quaternion [-0.191, 0.657, 0.203, 0.700]
            in RPY (radian) [-0.002, 1.508, 0.564]
            in RPY (degree) [-0.088, 86.400, 32.312]
elenoide@elepc:~$ rosrun tf tf_echo base_footprint camera_color_optical_frame
At time 1664480641.672
- Translation: [-0.221, -0.651, 1.095]
- Rotation: in Quaternion [-0.190, 0.654, 0.204, 0.703]
            in RPY (radian) [-0.010, 1.498, 0.555]
            in RPY (degree) [-0.550, 85.830, 31.782]
