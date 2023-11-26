# mild_hri_ros

ROS Package to run the [MILD-HRI](https://git.ias.informatik.tu-darmstadt.de/phri-learning/mild_hri) models on the Pepper Robot

## Dependencies

### Python Libraries

The following python libraries need to be installed with pip

- torch
- matplotlib
- numpy
- ikpy ([modified](https://github.com/souljaboy764/ikpy))
- [pbdlib_pytorch](https://git.ias.informatik.tu-darmstadt.de/prasad/pbdlib-torch)

For the skeleton tracking, Nuitrack needs to be installed. Follow the [instructions](https://github.com/3DiVi/nuitrack-sdk/blob/master/doc/Install.md) to install nuitrack. Once Nuitrack is installed and the license is activated, download the suitable [python wheel file for Nuitrack](https://github.com/3DiVi/nuitrack-sdk/tree/master/PythonNuitrack-beta/pip_packages/dist) and install it with `pip install /path/to/wheel.whl`

### ROS Packages

The following packages need to be installed to your catkin workspace:

- [naoqi_dcm_driver](https://github.com/souljaboy764/naoqi_dcm_driver) (along with the rest of the Pepper robot ROS stack)
- [tf_dynreconf](https://github.com/souljaboy764/tf_dynreconf)
- [pepper_controller_server](https://github.com/souljaboy764/pepper_controller_server)

## Installation

Once the prerequisites are installed, clone this repository to your catkin workspace and build it.

```bash
cd /path/to/catkin_ws/src
git clone https://github.com/souljaboy764/mild_hri_ros
cd ..
catkin_make
```

The pretrained model `mild_v3_2_pepper_nuisi.pth` is also made available with this repository.

## Setup

1. Before running any ROS nodes, make sure that the library path is set for Nuitrack.

    ```bash
    export LD_LIBRARY_PATH=/usr/local/lib/nuitrack
    source /path/to/current_ws/devel/setup.bash
    ```

2. Run `roslaunch mild_hri_ros prepare_hri.launch` after setting the IP of the Pepper robot and the network interface accordingly to get the setup ready. This launches the robot nodes, the transformation between the camera and the robot, collision avoidance etc.

3. For the external calibration, after starting up the robot with [`naoqi_dcm_driver`](https://github.com/souljaboy764/naoqi_dcm_driver), and `nuitrack_node.py` to start nuitrack, run `rosrun rqt_reconfigure rqt_reconfigure gui:=true` and change the values of the transofrmation until the external calibration is satisfactory. Save these values from the dynamic reconfigrue GUI in [`config/nuitrack_pepper_tf.yaml`](config/nuitrack_pepper_tf.yaml).

## Experimental run

Run steps 1 and 2 from above to setup the experiment to start up the robot.

First, reset the robot by running

`rosrun mild_hri_ros reset_pepper.py`

For each user, get them accustomed to the robot using:

`rosrun mild_hri_ros training_run.py --action (handshake|rocket)`

where the `--action` option needs to be either `handshaking` or `rocket` depending on the action that is being performed.

For running MILD:

`rosrun mild_hri_ros --ckpt /path/to/mild_v3_2_pepper_nuisi.pth --action handshake --ik`

after `--action` use either handshake or rocket for the interaction to be performed. For running only MILD without Inverse Kinematics, remove the `--ik` flag at the end. for running the IK baseline, change the flag at the end to `--ik-only`
Ask the user to stand still initially. Once it shows that the calibration is ready, in another terminal, run `rostopic pub /is_still std_msgs/Empty "{}" -1` to start the interaction.
