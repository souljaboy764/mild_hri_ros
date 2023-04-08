import numpy as np
import os
from ikpy.chain import Chain

from utils.visualization import *
from nuitrack_node import NuitrackROS

import rospy
import tf2_ros
from tf.transformations import *
from moveit_msgs.msg import DisplayRobotState
from geometry_msgs.msg import TransformStamped
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from pepper_controller_server.srv import JointTarget
import rospkg
rospack = rospkg.RosPack()

rospy.init_node("ik_hri_node")

rospy.wait_for_service('/pepper_dcm/RightArm_controller/goal')
send_target = rospy.ServiceProxy('/pepper_dcm/RightArm_controller/goal', JointTarget)
state_pub = rospy.Publisher("display_robot_state", DisplayRobotState, queue_size=5)
broadcaster = tf2_ros.StaticTransformBroadcaster()
tfBuffer = tf2_ros.Buffer()
listener = tf2_ros.TransformListener(tfBuffer)

ik_target = TransformStamped()
ik_target.header.frame_id = 'base_footprint'
ik_target.child_frame_id = 'ik_result'
ik_target.transform.rotation.w = 1

hand_tf = TransformStamped()
hand_tf.header.frame_id = 'base_footprint'
hand_tf.child_frame_id = 'hand'
hand_tf.transform.rotation.w = 1

# To visualise the expected state. Using both left and right arms otherwise the left arm is always raised
msg = DisplayRobotState()
msg.state.joint_state.name = ['RShoulderPitch', 'RShoulderRoll', 'RElbowYaw', 'RElbowRoll', "RWristYaw", 'LShoulderPitch', 'LShoulderRoll', 'LElbowYaw', 'LElbowRoll']
default_arm_joints = [1.5708, -0.109, 0.7854, 0.009]+np.deg2rad([90, 90,1,-90,-1]).tolist() # default standing angle values
msg.state.joint_state.position = default_arm_joints[:]

joint_names = ["RShoulderPitch", "RShoulderRoll", "RElbowYaw", "RElbowRoll", "RWristYaw"]
num_joints = len(joint_names)
joint_trajectory = JointTrajectory()
joint_trajectory.header.frame_id = "base_footprint"
joint_trajectory.joint_names = joint_names
joint_trajectory.points.append(JointTrajectoryPoint())
joint_trajectory.points[0].effort = np.ones(num_joints).tolist()
joint_trajectory.points[0].positions = default_arm_joints[:5]

pepper_right_arm_chain = Chain.from_json_file(os.path.join(rospack.get_path('segmint-ik'), "resources", "pepper", "pepper_right_arm.json"))

rate = rospy.Rate(100)
count = 0.
offset = np.array([0,0.1,0])
nuitrack = NuitrackROS(height=480, width=848, horizontal=False)
frame_target = np.eye(4)
ik_result = [0.0, 1.5708, -0.109, 0.7854, 0.009, 0.7854, 0.0 , 0.0, 0.0]
while not rospy.is_shutdown():
	img, nui_skeleton, stamp = nuitrack.update()
	if img is None or len(nui_skeleton)==0:
		continue
	
	hand_pose = nui_skeleton[-1, :]
	print(hand_pose)
	hand_pose[1] *= -1
	hand_pose = nuitrack.base2cam[:3,:3].dot(hand_pose) + nuitrack.base2cam[:3,3]
	hand_tf.transform.translation.x = hand_pose[0]
	hand_tf.transform.translation.y = hand_pose[1]
	hand_tf.transform.translation.z = hand_pose[2]
	
	target_pose = hand_pose - offset
	ik_target.transform.translation.x = target_pose[0]
	ik_target.transform.translation.y = target_pose[1]
	ik_target.transform.translation.z = target_pose[2]

	# Need pose in base_link frame for IK
	link_TF = tfBuffer.lookup_transform('base_link', 'base_footprint', rospy.Time())
	link_transform = quaternion_matrix([link_TF.transform.rotation.x, link_TF.transform.rotation.y, link_TF.transform.rotation.z, link_TF.transform.rotation.w])
	link_transform[:3,3] = [link_TF.transform.translation.x, link_TF.transform.translation.y, link_TF.transform.translation.z]
	hand_pose = link_transform[:3,:3].dot(hand_pose) + link_transform[:3,3]
	target_pose = hand_pose - offset

	count += 1
	if count < 20:
		print(count)
		# send_target(joint_trajectory)
		broadcaster.sendTransform([ik_target, hand_tf])
		state_pub.publish(msg)
		rate.sleep()
		continue
	elif count >60:
		joint_trajectory.points[0].effort[0] = 0.2

	frame_target[:3, 3] = target_pose
	ik_result = pepper_right_arm_chain.inverse_kinematics_frame(frame_target, initial_position=ik_result)
	# rarm_joints = rarm_joints*0.3 + 0.7*ik[2:6]
	rarm_joints = ik_result[2:6].tolist()
	msg.state.joint_state.position[:4] = rarm_joints
	joint_trajectory.points[0].positions = 0.3*np.array(joint_trajectory.points[0].positions) + 0.7*np.array(rarm_joints + [0.0])
	
	ik_target.header.stamp = hand_tf.header.stamp = joint_trajectory.header.stamp = stamp
	
	send_target(joint_trajectory)
	broadcaster.sendTransform([ik_target, hand_tf])
	state_pub.publish(msg)
	rate.sleep()
	
joint_trajectory.points[0].positions = default_arm_joints[:4]+ [0.5]
joint_trajectory.points[0].effort[0] = 0.7
joint_trajectory.header.stamp = rospy.Time.now()
send_target(joint_trajectory)
rospy.Rate(1).sleep()