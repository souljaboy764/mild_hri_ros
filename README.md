# SegmInt

# Dependencies:
- torch (only for preprocessing)
- matplotlib
- numpy
- ros
- qi (libqi and libqi-python)
- qibullet
- nuitrack
check_selfcollision package
tf_dynreconf package


# Data Preprocessing: 
Run `preproc.py` to format the data in the directory "`data/`" in a suitable manner. This would generate the file `labelled_sequences_prolonged.npz` (which currently exisits in the `data/` directory)

# Training
(only qibullet, matplotlib and numpy needed)
For training a basic HSMM on only human hand data and visualising example interactions, run `train_hands.py`
For training the HSMMs for HRI and to see how the robot motions look like in simulation, run `train_hri.py`

# Testing
For the external calibration, after starting up the robot with [`pepper_moveit_config`](https://github.com/ros-naoqi/pepper_moveit_config), and `nuitrack_node.py` to start nuitrack, run `roslaunch tf_dynreconf node.launch reconfigure reconfigure:=true` and change the values of the transofrmation until the external calibration is satisfactory. Save these values and also change the values in the matrix `base2cam` in the file `nuitrack_node.py` accordingly.


Run `roslaunch segmint prepare_hri.launch` after setting the IP of the Pepper robot and the network interface accordingly. 
For running the codes, there 