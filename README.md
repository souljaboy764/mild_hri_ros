# mild_hri_ros

ROS Package to run the [MILD-HRI](https://sites.google.com/view/mild-hri) models on the Pepper Robot

## Dependencies

### Python Libraries

- torch
- matplotlib
- numpy
<!-- - qi (libqi ; libqi-python) [Installation Instructions](https://gist.github.com/souljaboy764/beadea15ecaf9a2cde18420e1da04871) -->
- nuitrack (sdk installed and licensed ; python wrapper installed)
- ikpy ([modified](https://github.com/souljaboy764/ikpy))
- [pbdlib_pytorch](https://git.ias.informatik.tu-darmstadt.de/prasad/pbdlib-torch)
- [mild_hri](https://git.ias.informatik.tu-darmstadt.de/prasad/mild_hri)

### ROS Packages

- [naoqi_dcm_driver](https://github.com/souljaboy764/naoqi_dcm_driver) (along with the rest of the Pepper robot ROS stack)
- [check_selfcollision](https://github.com/souljaboy764/check_selfcollision)
- [tf_dynreconf](https://github.com/souljaboy764/tf_dynreconf)
- [pepper_controller_server](https://github.com/souljaboy764/pepper_controller_server)

## Setup

1. Before running any ROS nodes, make sure that the library path is set for Nuitrack.

    ```bash
    export LD_LIBRARY_PATH=/usr/local/lib/nuitrack
    source /path/to/current_ws/devel/setup.bash
    ```

2. Run `roslaunch mild_hri_ros prepare_hri.launch` after setting the IP of the Pepper robot and the network interface accordingly to get the setup ready. This launches the robot nodes, the transformation between the camera and the robot, collision avoidance etc.

3. For the external calibration, after starting up the robot with [`naoqi_dcm_driver`](https://github.com/souljaboy764/naoqi_dcm_driver), and `nuitrack_node.py` to start nuitrack, run `rosrun rqt_reconfigure rqt_reconfigure gui:=true` and change the values of the transofrmation until the external calibration is satisfactory. Save these values from the dynamic reconfigrue GUI in [`config/nuitrack_pepper_tf.yaml`](config/nuitrack_pepper_tf.yaml).

## Testing

Run steps 1 and 2 from above to setup the experiment.

For running mild:
```bash
python src/mild_hr_node.py  --ckpt models/nuisiv2pepper_3joints_xvel_v3_2_z5h6_trial1_400.pth --action handshake --ik
```

after --action use either handshake or rocket for the interaction to be performed. For running only MILD without ik, remove the --ik flag at the end. for running the IK baseline, change the flag at the end to --ik-only
Ask the user to stand still initially. Once it shows that the calibration is ready, in another terminal, run `rostopic pub /is_still std_msgs/Empty "{}" -1` to start the interaction.