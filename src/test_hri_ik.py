import matplotlib.pyplot as plt
import numpy as np

import rospy
import tf2_ros
from tf.transformations import *
from moveit_msgs.msg import DisplayRobotState
from geometry_msgs.msg import *

import qi

import sys
import time

from utils import *
from nuitrack_node import NuitrackROS, NuitrackWrapper

######################
### Starting up Pepper
######################
rospy.init_node("segment_ik_test")
broadcaster = tf2_ros.StaticTransformBroadcaster()
ik_target = TransformStamped()
ik_target.header.frame_id = 'base_footprint'
ik_target.child_frame_id = 'ik_result'
ik_target.transform.rotation.w = 1

hand_tf = TransformStamped()
hand_tf.header.frame_id = 'base_footprint'
hand_tf.child_frame_id = 'hand_ik_request'
hand_tf.transform.rotation.w = 1

session = qi.Session()
try:
	session.connect("tcp://192.168.100.122:9559")
except RuntimeError:
	print ("Can't connect to Naoqi.")
	sys.exit(1)

motion_service  = session.service("ALMotion")
motion_service.setBreathEnabled('Body', False)

if not motion_service.robotIsWakeUp():
	motion_service.wakeUp()


pub = rospy.Publisher("display_robot_state", DisplayRobotState, queue_size=5)
msg = DisplayRobotState()
msg.state.joint_state.name = ['RShoulderPitch', 'RShoulderRoll', 'RElbowYaw', 'RElbowRoll', 'LShoulderPitch', 'LShoulderRoll', 'LElbowYaw', 'LElbowRoll']
default_arm_joints = np.deg2rad([90,-1,90,1,90,1,-90,-1]).tolist() # default standing angle values
msg.state.joint_state.position = default_arm_joints[:]

rarm_joints = np.array(default_arm_joints[:4].copy()) # default standing angle values
joint_names = ['RShoulderPitch', 'RShoulderRoll', 'RElbowYaw', 'RElbowRoll', 'RHand']
motion_service.setStiffnesses(joint_names, 1.0)
motion_service.setAngles(joint_names, rarm_joints.tolist()+[0.0], 0.5)
fwd_kin = PepperKinematics(lambda_x = 0.7, lambda_theta=0.2)
# target_pos, _ = fwd_kin.end_effector(np.deg2rad([6,-38,26,40]))

rate = rospy.Rate(100)

handshake = False

if handshake:
	model = np.load('models/handshake_hri.npy', allow_pickle=True).item()
	offset = np.array([0,-0.,0])
	r_hand = 0.5
else:
	model = np.load('models/rocket_hri.npy', allow_pickle=True).item()
	offset = np.array([0.05,0,0])
	r_hand = 0.0


nuitrack = NuitrackROS(horizontal=False)
nuitrack.update() # IDK Why but needed for the first time
fig = plt.figure(figsize=(5,5))
ax = fig.add_subplot(projection='3d')
plt.ion()

count = 0
trajectory = []
alpha_hsmm = None
# input('')
crossed = False
last_segment = []
while not rospy.is_shutdown():
	img, nui_skeleton, stamp = nuitrack.update()
	if img is None:
		break
	prepare_axes(ax)
	plot_pbd(ax, model, alpha_hsmm)
	
	if len(nui_skeleton) == 0:
		pub.publish(msg)
		rate.sleep()
		plt.pause(0.001)
		continue
	
	
	# Origin at right shoulder, like in the training data
	skeleton = nui_skeleton.copy()
	skeleton -= skeleton[joints_idx["right_shoulder"]-1:joints_idx["right_shoulder"], :]
	skeleton = rotation_normalization(skeleton).dot(skeleton.T).T
	skeleton[:, 0] *= -1
	plot_skeleton(ax, skeleton)
	
	hand_pose = nui_skeleton[-1, :]
	hand_pose[1] *= -1
	hand_pose = nuitrack.base2cam[:3,:3].dot(hand_pose) + nuitrack.base2cam[:3,3]
	target_pose = hand_pose - offset
	hand_tf.transform.translation.x = target_pose[0]
	hand_tf.transform.translation.y = target_pose[1]
	hand_tf.transform.translation.z = target_pose[2]
	hand_tf.header.stamp = stamp

	# Giving ~10s ot the user to adjust pose for better visibility
	count += 1
	if count < 20:
		print(count)
		broadcaster.sendTransform([hand_tf])
		pub.publish(msg)
		rate.sleep()
		plt.pause(0.001)
		continue

	if len(trajectory) == 0:
		trajectory = [np.concatenate([skeleton[-1,:],np.zeros_like(skeleton[-2,:])])]
		broadcaster.sendTransform([hand_tf])
		pub.publish(msg)
		rate.sleep()
		plt.pause(0.001)
		continue
	
	trajectory.append(np.concatenate([skeleton[-1,:], skeleton[-2,:] - trajectory[-1][:3]]))
	cond_traj = np.array(trajectory)
	alpha_hsmm = forward_variable(model, len(cond_traj), cond_traj, slice(0, 6))
	mu_est_hsmm, sigma_est_hsmm = model.condition(cond_traj, dim_in=slice(0, 6), dim_out=slice(6, 10))
	active_segment = alpha_hsmm[:,-1].argmax()
	last_segment.append(active_segment)
	if sum(last_segment[:5])>20 or (len(trajectory) > 50 and active_segment==alpha_hsmm.shape[0] - 1): # if it's the last segment, then that means you're going back to the final position
		break
	if active_segment==0 or active_segment==alpha_hsmm.shape[0] - 1: # Handshake
	# if active_segment==0 or active_segment==1 or active_segment==alpha_hsmm.shape[0] - 1:  # Rocket
		mu_q = mu_est_hsmm[-1]
		print('NO IK')
	else:
		print('IK')
		mu_q, _ = fwd_kin.inv_kin(mu_theta=mu_est_hsmm[-1], sig_theta=np.eye(4),#sigma_est_hsmm[-1]*10,
										mu_x = target_pose, sig_x = np.eye(3)*0.01, 
										method='L-BFGS-B', jac=None, bounds=bounds,
										options={'disp': False}
								)
	
	rarm_joints = rarm_joints*0.9 + 0.1*mu_q
	msg.state.joint_state.position[:4] = rarm_joints
	pos, _ = fwd_kin.end_effector(mu_q)
	ik_target.transform.translation.x = pos[0]
	ik_target.transform.translation.y = pos[1]
	ik_target.transform.translation.z = pos[2]
	ik_target.header.stamp = stamp
	
	motion_service.setAngles(joint_names, np.clip(rarm_joints, lower_bounds, upper_bounds).tolist()+[r_hand], 0.5)
	# motion_service.angleInterpolation(joint_names, np.clip(rarm_joints, lower_bounds, upper_bounds).tolist()+[0.5], 0.4, True)
	broadcaster.sendTransform([ik_target, hand_tf])
	pub.publish(msg)
	rate.sleep()
	plt.pause(0.001)

plt.close()
motion_service.setAngles(joint_names, default_arm_joints[:4]+[0.5], 0.1)
time.sleep(3)
motion_service.setStiffnesses(joint_names, 0.0)
session.close()