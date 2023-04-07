# segmint-ik

SegmInt-IK: Segmenting Interactions for Inverse Kinematic Adaptation of Physically Interactive Gestures

## Dependencies

### Python Libraries

- torch (only for preprocessing)
- matplotlib
- numpy
- qi (libqi ; libqi-python)
- qibullet (for visualising training results)
- nuitrack (sdk installed and licensed ; python wrapper installed)
- ikpy ([modified](https://github.com/souljaboy764/ikpy))
- pbdlib (after applying [this patch for python3 issues](https://gist.github.com/souljaboy764/5d551c432d4a4ebf1433615595cfd87d))

### ROS Packages

- [naoqi_dcm_driver](https://github.com/souljaboy764/naoqi_dcm_driver) (along with the rest of the Pepper robot ROS stack)
- [check_selfcollision](https://github.com/souljaboy764/check_selfcollision)
- [tf_dynreconf](https://github.com/souljaboy764/tf_dynreconf)
- [pepper_controller_server](https://github.com/souljaboy764/pepper_controller_server)

## Data Preprocessing

The data used for training is from the handshaking and rocket fistbump interactions from the [NuiSI Dataset](https://github.com/souljaboy764/nuisi-dataset).

## Training

(only qibullet, matplotlib and numpy needed)
For training a basic HSMM on only human hand data and visualising example interactions, run `train_hands.py`
For training the HSMMs for HRI and to see how the robot motions look like in simulation, run `train_hri.py`

## Testing

For the external calibration, after starting up the robot with [`naoqi_dcm_driver`](https://github.com/souljaboy764/naoqi_dcm_driver), and `nuitrack_node.py` to start nuitrack, run `roslaunch tf_dynreconf node.launch reconfigure reconfigure:=true` and change the values of the transofrmation until the external calibration is satisfactory. Save these values from the dynamic reconfigrue GUI in [`config/nuitrack_pepper_tf.yaml`](config/nuitrack_pepper_tf.yaml).

Run `roslaunch segmint-ik prepare_hri.launch` after setting the IP of the Pepper robot and the network interface accordingly.
For running the codes, there
