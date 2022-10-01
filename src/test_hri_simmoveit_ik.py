import cv2
import matplotlib.pyplot as plt
import numpy as np
import pbdlib as pbd

import rospy
import tf2_ros
import moveit_commander
from tf.transformations import *
from geometry_msgs.msg import Point, Quaternion, TransformStamped
from moveit_msgs.msg import DisplayRobotState
from moveit_msgs.srv import GetPositionIK, GetPositionIKRequest, GetPositionIKResponse
from std_msgs.msg import Header
from sensor_msgs.msg import Image, PointCloud2, PointField, CameraInfo
from visualization_msgs.msg import Marker, MarkerArray


import sys
import time

from utils import *
from nuitrack_node import NuitrackROS, NuitrackWrapper

######################
### Starting up Pepper
######################
rospy.init_node("segment_ik_test")
moveit_commander.roscpp_initialize(sys.argv)
broadcaster = tf2_ros.StaticTransformBroadcaster()
ik_target = TransformStamped()
ik_target.header.frame_id = 'base_footprint'
ik_target.child_frame_id = 'ik_result'
ik_target.transform.rotation.w = 1

hand_tf = TransformStamped()
hand_tf.header.frame_id = 'base_footprint'
hand_tf.child_frame_id = 'hand'
hand_tf.transform.rotation.w = 1

pub = rospy.Publisher("display_robot_state", DisplayRobotState, queue_size=5)
msg = DisplayRobotState()
msg.state.joint_state.name = ['RShoulderPitch', 'RShoulderRoll', 'RElbowYaw', 'RElbowRoll', 'LShoulderPitch', 'LShoulderRoll', 'LElbowYaw', 'LElbowRoll']
default_arm_joints = np.deg2rad([90,-1,90,1,90,1,-90,-1]).tolist() # default standing angle values
msg.state.joint_state.position = default_arm_joints[:]
pub.publish(msg)
limits_max = [2.08567, -0.00872665, 2.08567, 1.56207]
limits_min = [-2.08567, -1.56207, -2.08567, 0.00872665]
bounds = ((limits_min[0], limits_max[0]),(limits_min[1], limits_max[1]),(limits_min[2], limits_max[2]),(limits_min[3], limits_max[3]))

fwd_kin = PepperKinematics(lambda_x = 0.5, lambda_theta=0.5)
# target_pos, _ = fwd_kin.end_effector(np.deg2rad([38,-9,62,40]))
# target_pos, _ = fwd_kin.end_effector(np.deg2rad([6,-38,26,40]))

move_group = moveit_commander.MoveGroupCommander("right_arm")
ik_service_client = rospy.ServiceProxy("compute_ik", GetPositionIK)
move_group.allow_replanning(True)
move_group.go(tuple(default_arm_joints[:4] + [0]), wait=False)

rospy.wait_for_service("compute_ik")
ik_request = GetPositionIKRequest()
ik_request.group_name = "right_arm"


rate = rospy.Rate(10)

model = np.load('models/rocket_hri.npy', allow_pickle=True).item()

nuitrack = NuitrackROS(horizontal=False)
nuitrack.update() # IDK Why but needed for the first time
fig = plt.figure(figsize=(7,7))
ax = fig.add_subplot(projection='3d')
plt.ion()

count = 0
trajectory = []
alpha_hsmm = None
# input('')
while not rospy.is_shutdown():
	img, nui_skeleton = nuitrack.update()
	if img is None:
		break
	prepare_axes(ax)
	plot_pbd(ax, model, alpha_hsmm)
	
	if len(nui_skeleton) == 0:
		plt.pause(0.001)
		continue
	
	# Giving ~10s ot the user to adjust pose for better visibility
	count += 1
	if count < 50:
		print(count)
		plt.pause(0.001)
		continue
	
	# Origin at right shoulder, like in the 
	skeleton = nui_skeleton.copy()
	skeleton -= skeleton[joints_idx["right_shoulder"]-1:joints_idx["right_shoulder"], :]
	skeleton = rotation_normalization(skeleton).dot(skeleton.T).T
	skeleton[:, 0] *= -1

	if len(trajectory) == 0:
		trajectory = [np.concatenate([skeleton[-2,:],np.zeros_like(skeleton[-2,:])])]
		plt.pause(0.001)
		continue
	trajectory.append(np.concatenate([skeleton[-2,:], skeleton[-2,:] - trajectory[-1][:3]]))
	cond_traj = np.array(trajectory)
	alpha_hsmm = forward_variable(model, len(cond_traj), cond_traj, slice(0, 6))
	mu_est_hsmm, sigma_est_hsmm = model.condition(cond_traj, dim_in=slice(0, 6), dim_out=slice(6, 10))
	active_segment = alpha_hsmm[:,-1].argmax()
	if len(trajectory) > 80 and alpha_hsmm[:,-1].max()>0.7 and active_segment == alpha_hsmm.shape[0] - 1:
		break
	
	print(alpha_hsmm[:,-1])
	hand_pose = nui_skeleton[-1, :]
	hand_pose[1] *= -1
	hand_pose = nuitrack.base2cam[:3,:3].dot(hand_pose) + nuitrack.base2cam[:3,3]
	if active_segment==0 or active_segment==alpha_hsmm.shape[0] - 1: # Handshake
	# if active_segment==0 or active_segment==1 or active_segment==alpha_hsmm.shape[0] - 1: # Rocket
		mu_q = mu_est_hsmm[-1]
	else:
		# mu_q, _ = fwd_kin.inv_kin(mu_theta=mu_est_hsmm[-1], sig_theta=np.eye(4),
		# 								mu_x = hand_pose, sig_x = np.eye(3)*0.01, 
		# 								method='L-BFGS-B', jac=None, bounds=bounds, options={'disp': False, 'iprint':-1})
		ik_request.robot_state = msg.state
		ik_request.pose_stamped.pose.position = Point(x=hand_pose[0], y=hand_pose[1], z=hand_pose[2])
		ik_request.pose_stamped.header = 'base_footprint'
		ik_request.pose_stamped.header.stamp = rospy.Time.now()
		ik_response = ik_service_client(ik_request)

		rospy.loginfo("FK Result: " + str((ik_response.error_code.val == ik_response.error_code.SUCCESS)) +' '+ str(ik_response.error_code.val))
		if ik_response.error_code.val == ik_response.error_code.SUCCESS:
			mu_q = np.array(ik_response.solution.joint_state.position[:4])
		
	msg.state.joint_state.position[:4] = np.array(msg.state.joint_state.position[:4])*0.7 + 0.3*mu_q
	pos, _ = fwd_kin.end_effector(mu_q)
	ik_target.transform.translation.x = pos[0]
	ik_target.transform.translation.y = pos[1]
	ik_target.transform.translation.z = pos[2]

	hand_tf.transform.translation.x = hand_pose[0]
	hand_tf.transform.translation.y = hand_pose[1]
	hand_tf.transform.translation.z = hand_pose[2]
	hand_tf.header.stamp = ik_target.header.stamp = rospy.Time.now()
	broadcaster.sendTransform([ik_target, hand_tf])
	pub.publish(msg)

	plot_skeleton(ax, skeleton)
	plt.tight_layout()
	plt.pause(0.001)

	rate.sleep()

plt.close()
move_group.go(tuple(default_arm_joints), wait=False)
# motion_service.setAngles(joint_names, arm_joints, 0.1)
# motion_service.setStiffnesses(joint_names, 0.0)
# cv2.destroyAllWindows()
# session.close()