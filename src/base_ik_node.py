#!/usr/bin/python

import numpy as np
import os
from ikpy.chain import Chain

# from utils import *
from utils import *
from nuitrack_node import NuitrackROS

import rospy
import tf2_ros
from tf.transformations import *
from moveit_msgs.msg import DisplayRobotState
from geometry_msgs.msg import TransformStamped
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from pepper_controller_server.srv import JointTarget
from sensor_msgs.msg import JointState
import rospkg
rospack = rospkg.RosPack()
default_arm_joints = [1.5708, -0.109, 0.7854, 0.009, 1.8239, 0.75] # default standing angle values

class BaseIKController:
	def __init__(self):
		rospy.wait_for_service('/pepper_dcm/RightArm_controller/goal')
		self.send_target = rospy.ServiceProxy('/pepper_dcm/RightArm_controller/goal', JointTarget)
		self.state_pub = rospy.Publisher("display_robot_state", DisplayRobotState, queue_size=5)
		self.broadcaster = tf2_ros.StaticTransformBroadcaster()
		self.tfBuffer = tf2_ros.Buffer()
		self.listener = tf2_ros.TransformListener(self.tfBuffer)
		self.robot_hand_joint = 0.

		self.target_tf = TransformStamped()
		self.target_tf.header.frame_id = 'base_footprint'
		self.target_tf.child_frame_id = 'target'
		self.target_tf.transform.rotation.w = 1.

		self.hand_tf = TransformStamped()
		self.hand_tf.header.frame_id = 'base_footprint'
		self.hand_tf.child_frame_id = 'human_hand'
		self.hand_tf.transform.rotation.w = 1.

		self.endeff_tf = TransformStamped()
		self.endeff_tf.header.frame_id = 'base_link'
		self.endeff_tf.child_frame_id = 'endeff'
		self.endeff_tf.transform.rotation.w = 1.

		self.state_msg = DisplayRobotState()
		self.state_msg.state.joint_state.header.frame_id = "base_footprint"

		self.joint_names = ["RShoulderPitch", "RShoulderRoll", "RElbowYaw", "RElbowRoll", "RWristYaw", 'RHand']
		self.joint_trajectory = JointTrajectory()
		self.joint_trajectory.header.frame_id = "base_footprint"
		self.joint_trajectory.joint_names = self.joint_names
		self.joint_trajectory.points.append(JointTrajectoryPoint())
		self.joint_trajectory.points[0].effort = np.ones(len(self.joint_names)).tolist()
		self.joint_trajectory.points[0].effort[0] = 1.0
		self.joint_trajectory.points[0].positions = default_arm_joints
		self.joint_trajectory.header.stamp = rospy.Time.now()

		self.pepper_chain = Chain.from_json_file(os.path.join(rospack.get_path('mild_hri_ros'), "resources", "pepper", "pepper_right_arm.json"))

		self.offset = np.array([0.0,0.0,0])
		self.nuitrack = NuitrackROS(height=480, width=848, horizontal=False)
		self.ik_result = np.array([0.0, 0.0, 1.5708, -0.109, 0.7854, 0.009, 1.8239, 0.0 , 0.0])

		self.joint_state_sub = rospy.Subscriber('/joint_states', JointState, self.joint_state_cb)

	def joint_state_cb(self, msg:JointState):
		if len(msg.name)<=6:
			return
		self.joint_readings = np.array(msg.position[11:17])
		self.endeff = self.pepper_chain.forward_kinematics(self.pepper_chain.active_to_full(self.joint_readings[:4], [0] * len(self.pepper_chain.links)))
		self.endeff_tf.transform = mat2TF(self.endeff)

		self.state_msg.state.joint_state = msg
		self.state_msg.state.joint_state.header.frame_id = "base_footprint"
		self.state_msg.state.joint_state.position = list(self.state_msg.state.joint_state.position)
		self.state_msg.state.joint_state.position[11:17] = list(self.joint_trajectory.points[0].positions)

	def observe_human(self):
		img, nui_skeleton, stamp = self.nuitrack.update()
		if img is None or len(nui_skeleton)==0:
			return [], None, stamp
		
		hand_pose = self.nuitrack.base2cam[:3,:3].dot(nui_skeleton[-1, :]) + self.nuitrack.base2cam[:3,3]

		self.hand_tf.transform = mat2TF(hand_pose)
		
		return nui_skeleton, hand_pose, stamp
	
	def publish(self, stamp):
		self.state_msg.state.joint_state.header.stamp = self.target_tf.header.stamp = self.endeff_tf.header.stamp = self.hand_tf.header.stamp = self.joint_trajectory.header.stamp = stamp
		self.send_target(self.joint_trajectory)
		self.broadcaster.sendTransform([self.target_tf, self.hand_tf, self.endeff_tf])
		self.state_pub.publish(self.state_msg)

	def in_baselink(self, hand_pose):
		# Need pose in base_link frame for IK
		link_TF = self.tfBuffer.lookup_transform('base_link', 'base_footprint', rospy.Time())
		link_transform = ROS2mat(link_TF.transform)
		hand_pose = link_transform[:3,:3].dot(hand_pose) + link_transform[:3,3]
		return hand_pose - self.offset
	
	def step(self, nui_skeleton, hand_pose, **kwargs):
		if len(nui_skeleton)==0 or hand_pose is None:
			return
		self.target_tf.transform = mat2TF(hand_pose)
		target_pose = self.in_baselink(hand_pose)
		frame_target = np.eye(4)
		frame_target[:3, 3] = target_pose
		self.ik_result = self.pepper_chain.inverse_kinematics_frame(frame_target, initial_position=self.ik_result, **kwargs)
		rarm_joints = self.ik_result[2:6].tolist()
		self.joint_trajectory.points[0].positions = 0.2*np.array(self.joint_trajectory.points[0].positions) + 0.8*np.array(rarm_joints + [1., self.robot_hand_joint])
	
if __name__=='__main__':
	rospy.init_node('base_ik_node')
	rate = rospy.Rate(100)
	controller = BaseIKController()
	controller.observe_human()
	count = 0
	hand_pos_init = []
	prev_z = None
	stop_counter = 0
	rospy.Rate(0.5).sleep()
	controller.joint_trajectory.points[0].effort[0] = 1.0
	started = False
	while not rospy.is_shutdown():
		nui_skeleton, hand_pose, stamp = controller.observe_human()
		if len(nui_skeleton)!=0:
			count += 1
		if count < 20:
			if hand_pose is not None:
				hand_pos_init.append(hand_pose)
			controller.publish(stamp)
			rate.sleep()
			continue
		elif count == 20:
			hand_pos_init = np.mean(hand_pos_init, 0)
			print('Calibration ready')
		if not started and ((hand_pose - hand_pos_init)**2).sum() < 0.001:
			print('Not yet started. Current displacement:', ((hand_pose - hand_pos_init)**2).sum())
			continue
		else:
			started = True
		# if started and controller.joint_readings[0] < 0.2:
		# 	if controller.joint_trajectory.points[0].effort[0] > 0.2:
		# 		controller.joint_trajectory.points[0].effort[0] -= 0.1
		# print(controller.joint_trajectory.points[0].effort[0], controller.joint_readings[0])
		controller.step(nui_skeleton, hand_pose, optimizer = "least_squares")
		controller.publish(stamp)
		rate.sleep()
		if started and count>100 and ((hand_pose - hand_pos_init)**2).sum() < 0.005: # hand_pose[2] < 0.63 and hand_pose[2] - prev_z < -0.005:
			break
			

	controller.joint_trajectory.points[0].effort[0] = 1.0
	controller.joint_trajectory.points[0].positions = default_arm_joints
	controller.publish(rospy.Time.now())
	rospy.Rate(0.5).sleep()
	rospy.signal_shutdown('done')