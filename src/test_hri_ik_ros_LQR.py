import cv2
import matplotlib.pyplot as plt
import numpy as np
import pbdlib as pbd

import rospy
import tf2_ros
from tf.transformations import *
from moveit_msgs.msg import DisplayRobotState
import moveit_commander
from geometry_msgs.msg import Point, Quaternion, TransformStamped
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from sensor_msgs.msg import JointState

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
hand_tf.child_frame_id = 'hand'
hand_tf.transform.rotation.w = 1

robot_traj_publisher = rospy.Publisher("/pepper_dcm/RightArm_controller/command", JointTrajectory, queue_size=1)
pub = rospy.Publisher("display_robot_state", DisplayRobotState, queue_size=5)
msg = DisplayRobotState()
msg.state.joint_state.name = ['RShoulderPitch', 'RShoulderRoll', 'RElbowYaw', 'RElbowRoll', 'LShoulderPitch', 'LShoulderRoll', 'LElbowYaw', 'LElbowRoll']
default_arm_joints = np.deg2rad([90,-1,90,1,90,1,-90,-1]).tolist() # default standing angle values
msg.state.joint_state.position = default_arm_joints[:]

rarm_joints = np.array(default_arm_joints[:4].copy()) # default standing angle values
joint_names = ['RShoulderPitch', 'RShoulderRoll', 'RElbowYaw', 'RElbowRoll', 'RWristYaw']
fwd_kin = PepperKinematics(lambda_x = 0.5, lambda_theta=0.5)

joint_trajectory = JointTrajectory()
joint_trajectory.header.frame_id = "odom"
joint_trajectory.joint_names = joint_names
for i in range(5):
	joint_trajectory.points.append(JointTrajectoryPoint())
	joint_trajectory.points[-1].time_from_start = rospy.Duration.from_sec(0.05*(i+1))
	joint_trajectory.points[-1].positions = rarm_joints+ [0.0]
joint_trajectory.header.stamp = rospy.Time.now()
robot_traj_publisher.publish(joint_trajectory)
rospy.Rate(1).sleep()
rate = rospy.Rate(100)

A, b = pbd.utils.get_canonical(2, 2, 0.01)
lqr = pbd.LQR(A, b, nb_dim=7, horizon=6)
handshake = True

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
last_q = rarm_joints

robot_model = pbd.GMM(nb_stats=model.nb_states, nb_dim=8, mu=model.mu[:,6:], sigma=model.mu[:,6:, 6:])

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

	if len(trajectory) < 6:
		trajectory.append(np.concatenate([skeleton[-1,:],np.zeros_like(skeleton[-2,:])]))
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
	if sum(last_segment[-6:])>=30 or (len(trajectory) > 50 and active_segment==alpha_hsmm.shape[0] - 1): # if it's the last segment, then that means you're going back to the final position
		print(last_segment[-6:], sum(last_segment[-6:]))
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

	sq = np.argmax(alpha_hsmm, axis=0)
	lqr.gmm_xi = model, sq
	lqr.gmm_u = -4.
	lqr.ricatti()
	q_lqr, _ = lqr.get_seq(np.concatenate([cond_traj[-1], mu_q, mu_q-last_q]))
	q_lqr = q_lqr[:, 6:10]

	msg.state.joint_state.position[:4] = q_lqr[0]
	rarm_joints = rarm_joints*0.3 + 0.7*q_lqr[0]
	
	
	for i in range(5):
		joint_trajectory.points[i].positions = q_lqr[i].tolist() + [0.0]
	pos, _ = fwd_kin.end_effector(q_lqr[0])
	ik_target.transform.translation.x = pos[0]
	ik_target.transform.translation.y = pos[1]
	ik_target.transform.translation.z = pos[2]
	ik_target.header.stamp = hand_tf.header.stamp = rospy.Time.now()
	
	joint_trajectory.header.stamp = rospy.Time.now()
	robot_traj_publisher.publish(joint_trajectory)
	# joint_trajectory.header.stamp = rospy.Time.now()
	# robot_traj_publisher.publish(joint_trajectory)
	# joint_trajectory.header.stamp = rospy.Time.now()
	# robot_traj_publisher.publish(joint_trajectory)
	rate.sleep()
	broadcaster.sendTransform([ik_target, hand_tf])
	pub.publish(msg)
	plt.pause(0.001)
	last_q = rarm_joints

plt.close()
for i in range(5):
	joint_trajectory.points[i].positions = default_arm_joints[:4]+ [0.0]
joint_trajectory.header.stamp = rospy.Time.now()
robot_traj_publisher.publish(joint_trajectory)
rospy.Rate(1).sleep()
# joint_trajectory.header.stamp = rospy.Time.now()
# robot_traj_publisher.publish(joint_trajectory)
# rospy.Rate(1).sleep()
# joint_trajectory.header.stamp = rospy.Time.now()
# robot_traj_publisher.publish(joint_trajectory)
# rospy.Rate(1).sleep()