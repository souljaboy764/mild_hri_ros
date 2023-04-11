import matplotlib.pyplot as plt
import numpy as np
import pbdlib as pbd

import rospy
import tf2_ros
from tf.transformations import *
from moveit_msgs.msg import DisplayRobotState
from geometry_msgs.msg import TransformStamped
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from utils import *
from nuitrack_node import NuitrackROS

from base_ik_controller import *

class HSMMIKController(BaseIKController):
	def __init__(self):
		super().__init__()
		self.__model = np.load('models/handshake_hri.npy', allow_pickle=True).item()
		self.alpha_hsmm = None
		self.crossed = False
		self.started = False
		self.history = []
		self.predicted_segment = []

		self.rotation_normalization_matrix = None
		
		self.fig = plt.figure(figsize=(5,5))
		self.ax = self.fig.add_subplot(projection='3d')
		plt.ion()

	def step(self, nui_skeleton, hand_pose):
		if len(nui_skeleton)==0 or hand_pose is None:
			return
		# Origin at right shoulder, like in the training data
		# if self.rotation_normalization_matrix is None:
		self.rotation_normalization_matrix = np.eye(4)
		self.rotation_normalization_matrix[:3, 3] = nui_skeleton[joints_idx["right_shoulder"]-1:joints_idx["right_shoulder"], :]
		nui_skeleton -= nui_skeleton[joints_idx["right_shoulder"]-1:joints_idx["right_shoulder"], :]
		self.rotation_normalization_matrix[:3, :3] = rotation_normalization(nui_skeleton)
		nui_skeleton = self.rotation_normalization_matrix[:3,:3].dot(nui_skeleton.T).T
		nui_skeleton[:, 1] *= -1
		nui_skeleton[:, 0] *= -1
		# else:
		# 	nui_skeleton = self.rotation_normalization_matrix[:3,:3].dot(nui_skeleton.T).T
		# 	nui_skeleton -= self.rotation_normalization_matrix[:3, 3]


		if len(self.history) == 0:
			self.history = [np.concatenate([nui_skeleton[-1,:],np.zeros_like(nui_skeleton[-1,:])])]
			return

		self.history.append(np.concatenate([nui_skeleton[-1,:], nui_skeleton[-1,:] - self.history[-1][:3]]))
		if not self.started and (self.history[-1][3:]**2).sum() < 0.001:
			print((self.history[-1][3:]**2).sum())
		else:
			self.started = True

		alpha_hsmm = forward_variable(self.__model, len(self.history), np.array(self.history), slice(0, 6))
		active_segment = alpha_hsmm[:,-1].argmax()
		self.predicted_segment.append(alpha_hsmm)
			
		prepare_axes(self.ax)
		if self.predicted_segment ==[]:
			plot_pbd(self.ax, self.__model)
		else:
			plot_pbd(self.ax, self.__model, self.predicted_segment[-1])
		plot_skeleton(self.ax, nui_skeleton)
	
	def publish(self, stamp):
		super().publish(stamp)

if __name__=='__main__':
	rospy.init_node('hri_hsmmik_node')
	controller = HSMMIKController()
	controller.observe_human()
	rate = rospy.Rate(100)
	count = 0
	while not rospy.is_shutdown():
		if not plt.fignum_exists(controller.fig.number):
			break
		nui_skeleton, hand_pose, stamp = controller.observe_human()
		count += 1
		if count < 20:
			controller.publish(stamp)
			rate.sleep()
			plt.pause(0.001)
			continue
		elif count >100:
			controller.joint_trajectory.points[0].effort[0] = 0.2
		
		controller.step(nui_skeleton, hand_pose)
		controller.publish(stamp)
		rate.sleep()
		plt.pause(0.001)
		# if hand_pose is not None:
		# 	if prev_z is not None:
		# 		print(hand_pose[2] - prev_z)
		# 		if hand_pose[2] < 0.65 and hand_pose[2] - prev_z < -0.001:
		# 			controller.joint_trajectory.points[0].positions = default_arm_joints
		# 			controller.publish(stamp)
		# 			rate.sleep()
		# 			rospy.signal_shutdown('done')
		# 	prev_z = hand_pose[2]


		# if hand_pose is not None and hand_pose[2] < 0.8:
		# 	stop_counter+= 1
		# else:
		# 	stop_counter = 0		
		# if stop_counter>10:
		# 	controller.joint_trajectory.points[0].positions = default_arm_joints
		# 	controller.publish(stamp)
		# 	rate.sleep()
		# 	rospy.signal_shutdown('Done')

	# 	cond_traj = np.array(trajectory)
	# 	alpha_hsmm = forward_variable(model, len(cond_traj), cond_traj, slice(0, 6))
	# 	mu_est_hsmm, sigma_est_hsmm = model.condition(cond_traj, dim_in=slice(0, 6), dim_out=slice(6, 10))
	# 	active_segment = alpha_hsmm[:,-1].argmax()
	# 	last_segment.append(active_segment)
	# 	if sum(last_segment[-6:])>=30 or (len(trajectory) > 50 and active_segment==alpha_hsmm.shape[0] - 1): # if it's the last segment, then that means you're going back to the final position
	# 		print(last_segment[-6:], sum(last_segment[-6:]))
	# 		break
	# 	if active_segment==0 or active_segment==alpha_hsmm.shape[0] - 1: # Handshake
	# 	# if active_segment==0 or active_segment==1 or active_segment==alpha_hsmm.shape[0] - 1:  # Rocket
	# 		mu_q = mu_est_hsmm[-1]
	# 		print('NO IK')
	# 	else:
	# 		print('IK')
	# 		mu_q, _ = fwd_kin.inv_kin(mu_theta=mu_est_hsmm[-1], sig_theta=np.eye(4),#sigma_est_hsmm[-1]*10,
	# 										mu_x = target_pose, sig_x = np.eye(3)*0.01, 
	# 										method='L-BFGS-B', jac=None, bounds=bounds,
	# 										options={'disp': False}
	# 								)
		
	# 	rarm_joints = rarm_joints*0.3 + 0.7*mu_q
	# 	msg.state.joint_state.position[:4] = rarm_joints
	# 	for i in range(5):
	# 		joint_trajectory.points[i].positions = rarm_joints.tolist() + [0.0]
	# 		# hand_trajectory.points[-1].positions = [r_hand]
	# 	pos, _ = fwd_kin.end_effector(mu_q)
	# 	ik_target.transform.translation.x = pos[0]
	# 	ik_target.transform.translation.y = pos[1]
	# 	ik_target.transform.translation.z = pos[2]
	# 	ik_target.header.stamp = hand_tf.header.stamp = rospy.Time.now()
		
	# 	joint_trajectory.header.stamp = rospy.Time.now()
	# 	robot_traj_publisher.publish(joint_trajectory)
	# 	# hand_trajectory.header.stamp = rospy.Time.now()
	# 	# robot_hand_publisher.publish(hand_trajectory)
	# 	# joint_trajectory.header.stamp = rospy.Time.now()
	# 	# robot_traj_publisher.publish(joint_trajectory)
	# 	# joint_trajectory.header.stamp = rospy.Time.now()
	# 	# robot_traj_publisher.publish(joint_trajectory)
	# 	rate.sleep()
	# 	broadcaster.sendTransform([ik_target, hand_tf])
	# 	pub.publish(msg)
	# 	plt.pause(0.001)


	# print(np.array(trajectory)[:5])
	# for i in range(5):
	# 	joint_trajectory.points[i].positions = default_arm_joints[:4]+ [0.5]
	# 	hand_trajectory.points[-1].positions = [0.5]
	# joint_trajectory.header.stamp = rospy.Time.now()
	# robot_traj_publisher.publish(joint_trajectory)
	# hand_trajectory.header.stamp = rospy.Time.now()
	# robot_hand_publisher.publish(hand_trajectory)
	# rospy.Rate(1).sleep()
	# plt.close()
	# # joint_trajectory.header.stamp = rospy.Time.now()
	# # robot_traj_publisher.publish(joint_trajectory)
	# # rospy.Rate(1).sleep()
	# # joint_trajectory.header.stamp = rospy.Time.now()
	# # robot_traj_publisher.publish(joint_trajectory)
	# # rospy.Rate(1).sleep()
plt.show()