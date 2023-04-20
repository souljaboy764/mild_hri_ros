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

		self.target_tf = TransformStamped()
		self.target_tf.header.frame_id = 'base_footprint'
		self.target_tf.child_frame_id = 'target'
		self.target_tf.transform.rotation.w = 1.

		self.hand_tf = TransformStamped()
		self.hand_tf.header.frame_id = 'base_footprint'
		self.hand_tf.child_frame_id = 'hand'
		self.hand_tf.transform.rotation.w = 1.

		# To visualise the expected state. Using both left and right arms otherwise the left arm is always raised
		self.state_msg = DisplayRobotState()
		self.state_msg.state.joint_state.header.frame_id = "base_footprint"

		self.joint_names = ["RShoulderPitch", "RShoulderRoll", "RElbowYaw", "RElbowRoll", "RWristYaw", 'RHand']
		self.joint_trajectory = JointTrajectory()
		self.joint_trajectory.header.frame_id = "base_footprint"
		self.joint_trajectory.joint_names = self.joint_names
		self.joint_trajectory.points.append(JointTrajectoryPoint())
		self.joint_trajectory.points[0].effort = np.ones(len(self.joint_names)).tolist()
		self.joint_trajectory.points[0].effort[0] = 0.5
		self.joint_trajectory.points[0].positions = default_arm_joints
		self.joint_trajectory.header.stamp = rospy.Time.now()

		self.pepper_chain = Chain.from_json_file(os.path.join(rospack.get_path('segmint-ik'), "resources", "pepper", "pepper_right_arm.json"))

		self.offset = np.array([0.0,0.0,0])
		self.nuitrack = NuitrackROS(height=480, width=848, horizontal=False)
		self.ik_result = [0.0, 1.5708, -0.109, 0.7854, 0.009, 1.8239, 0.0 , 0.0, 0.0]

		self.joint_state_sub = rospy.Subscriber('/joint_states', JointState, self.joint_state_cb)

	def joint_state_cb(self, msg:JointState):
		if len(msg.name)<=6:
			return
		self.joint_readings = msg.position[11:17]
		self.state_msg.state.joint_state = msg
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
		self.state_msg.state.joint_state.header.stamp = self.target_tf.header.stamp = self.hand_tf.header.stamp = self.joint_trajectory.header.stamp = stamp
		# self.send_target(self.joint_trajectory)
		self.broadcaster.sendTransform([self.target_tf, self.hand_tf])
		self.state_pub.publish(self.state_msg)

	def in_baselink(self, hand_pose):
		# Need pose in base_link frame for IK
		link_TF = self.tfBuffer.lookup_transform('base_link', 'base_footprint', rospy.Time())
		link_transform = ROS2mat(link_TF.transform)
		hand_pose = link_transform[:3,:3].dot(hand_pose) + link_transform[:3,3]
		return hand_pose - self.offset
	
	def step(self, nui_skeleton, hand_pose):
		if len(nui_skeleton)==0 or hand_pose is None:
			return
		self.target_tf.transform = mat2TF(hand_pose)
		target_pose = self.in_baselink(hand_pose)
		frame_target = np.eye(4)
		frame_target[:3, 3] = target_pose
		self.ik_result = self.pepper_chain.inverse_kinematics_frame(frame_target, initial_position=self.ik_result, optimizer = "scalar")
		rarm_joints = self.ik_result[2:6].tolist()
		self.joint_trajectory.points[0].positions = 0.3*np.array(self.joint_trajectory.points[0].positions) + 0.7*np.array(rarm_joints + [1.8239, 0.75])
	
if __name__=='__main__':
	rospy.init_node('hri_ik_node')
	rate = rospy.Rate(100)
	controller = BaseIKController()
	controller.observe_human()
	count = 0
	prev_z = None
	stop_counter = 0
	rate.sleep()
	while not rospy.is_shutdown():
		nui_skeleton, hand_pose, stamp = controller.observe_human()
		controller.joint_trajectory.points[0].effort[0] = 0.2
		count += 1
		if count < 20:
			controller.publish(stamp)
			rate.sleep()
			continue
		elif count >100:
			controller.joint_trajectory.points[0].effort[0] = 0.2
		controller.step(nui_skeleton, hand_pose)
		controller.publish(stamp)
		rate.sleep()
		if hand_pose is not None:
			if prev_z is not None:
				if hand_pose[2] < 0.65 and hand_pose[2] - prev_z < -0.001:
					controller.joint_trajectory.points[0].positions = default_arm_joints
					controller.publish(stamp)
					rate.sleep()
					rospy.signal_shutdown('done')
			prev_z = hand_pose[2]


		# if hand_pose is not None and hand_pose[2] < 0.8:
		# 	stop_counter+= 1
		# else:
		# 	stop_counter = 0		
		# if stop_counter>10:
		# 	controller.joint_trajectory.points[0].positions = default_arm_joints
		# 	controller.publish(stamp)
		# 	rate.sleep()
		# 	rospy.signal_shutdown('Done')
