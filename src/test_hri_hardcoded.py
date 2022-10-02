import cv2
import matplotlib.pyplot as plt
import numpy as np

import rospy
import tf2_ros
from tf.transformations import *
from moveit_msgs.msg import DisplayRobotState
from geometry_msgs.msg import TransformStamped
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from utils import *
from nuitrack_node import NuitrackROS

######################
### Starting up Pepper
######################
rospy.init_node("segment_ik_test")
broadcaster = tf2_ros.StaticTransformBroadcaster()
ik_target = TransformStamped()
ik_target.header.frame_id = 'base_footprint'
ik_target.child_frame_id = 'robot_goal'
ik_target.transform.rotation.w = 1

hand_tf = TransformStamped()
hand_tf.header.frame_id = 'base_footprint'
hand_tf.child_frame_id = 'hand'
hand_tf.transform.rotation.w = 1

handshake = False

if handshake:
	model = np.load('models/handshake_hri.npy', allow_pickle=True).item()
	offset = np.array([0,-0.,0])
	r_hand = 0.5
else:
	model = np.load('models/rocket_hri.npy', allow_pickle=True).item()
	offset = np.array([0.05,0,0])
	r_hand = 0.0

robot_traj_publisher = rospy.Publisher("/pepper_dcm/RightArm_controller/command", JointTrajectory, queue_size=1)
robot_hand_publisher = rospy.Publisher("/pepper_dcm/RightHand_controller/command", JointTrajectory, queue_size=1)
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
	joint_trajectory.points[-1].time_from_start = rospy.Duration.from_sec(0.1*(i+1))
	joint_trajectory.points[-1].positions = default_arm_joints[:4]+ [0.0]
joint_trajectory.header.stamp = rospy.Time.now()
robot_traj_publisher.publish(joint_trajectory)

rospy.Rate(1).sleep()

hand_trajectory = JointTrajectory()
hand_trajectory.header.frame_id = "odom"
hand_trajectory.joint_names = ['RHand']
for i in range(5):
	hand_trajectory.points.append(JointTrajectoryPoint())
	hand_trajectory.points[-1].time_from_start = rospy.Duration.from_sec(0.1*(i+1))
	hand_trajectory.points[-1].positions = [r_hand]
hand_trajectory.header.stamp = rospy.Time.now()
robot_hand_publisher.publish(hand_trajectory)
rospy.Rate(1).sleep()
rate = rospy.Rate(100)



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
started = False
last_segment = []
while not rospy.is_shutdown():
	if not plt.fignum_exists(fig.number):
		break
	img, nui_skeleton, stamp = nuitrack.update()
	if img is None:
		continue
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
		trajectory = [np.concatenate([skeleton[-1,:],np.zeros_like(skeleton[-1,:])])]
		broadcaster.sendTransform([hand_tf])
		pub.publish(msg)
		rate.sleep()
		plt.pause(0.001)
		continue

	trajectory.append(np.concatenate([skeleton[-1,:], skeleton[-1,:] - trajectory[-1][:3]]))
	if not started and (trajectory[-1][3:]**2).sum() < 0.001:
		print((trajectory[-1][3:]**2).sum())
		broadcaster.sendTransform([hand_tf])
		pub.publish(msg)
		rate.sleep()
		plt.pause(0.001)
		continue
	else:
		started = True
	if len(trajectory) < 40:
		mu_q = np.deg2rad([51, -1, 74, 52])
	elif len(trajectory) < 80:
		mu_q = np.deg2rad([-35, -1, 74, 19])
	else:
		break
	
	
	rarm_joints = rarm_joints*0.3 + 0.7*mu_q
	msg.state.joint_state.position[:4] = rarm_joints
	for i in range(5):
		joint_trajectory.points[i].positions = rarm_joints.tolist() + [0.0]
		# hand_trajectory.points[-1].positions = [r_hand]
	pos, _ = fwd_kin.end_effector(mu_q)
	ik_target.transform.translation.x = pos[0]
	ik_target.transform.translation.y = pos[1]
	ik_target.transform.translation.z = pos[2]
	ik_target.header.stamp = hand_tf.header.stamp = rospy.Time.now()
	
	joint_trajectory.header.stamp = rospy.Time.now()
	robot_traj_publisher.publish(joint_trajectory)
	# hand_trajectory.header.stamp = rospy.Time.now()
	# robot_hand_publisher.publish(hand_trajectory)
	# joint_trajectory.header.stamp = rospy.Time.now()
	# robot_traj_publisher.publish(joint_trajectory)
	# joint_trajectory.header.stamp = rospy.Time.now()
	# robot_traj_publisher.publish(joint_trajectory)
	rate.sleep()
	broadcaster.sendTransform([ik_target, hand_tf])
	pub.publish(msg)
	plt.pause(0.001)


print(np.array(trajectory)[:5])
for i in range(5):
	joint_trajectory.points[i].positions = default_arm_joints[:4]+ [0.5]
	hand_trajectory.points[-1].positions = [0.5]
joint_trajectory.header.stamp = rospy.Time.now()
robot_traj_publisher.publish(joint_trajectory)
hand_trajectory.header.stamp = rospy.Time.now()
robot_hand_publisher.publish(hand_trajectory)
rospy.Rate(1).sleep()
plt.close()
# joint_trajectory.header.stamp = rospy.Time.now()
# robot_traj_publisher.publish(joint_trajectory)
# rospy.Rate(1).sleep()
# joint_trajectory.header.stamp = rospy.Time.now()
# robot_traj_publisher.publish(joint_trajectory)
# rospy.Rate(1).sleep()