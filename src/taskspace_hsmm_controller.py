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
	def __init__(self, nbD=10):
		super().__init__()
		self.model = np.load('models/handshake_hands.npy', allow_pickle=True).item()
		self.model.Pd = np.zeros((self.model.nb_states, nbD))
		# Precomputation of duration probabilities
		for i in range(self.model.nb_states):
			self.model.Pd[i, :] = pbd.multi_variate_normal(np.arange(nbD), self.model.Mu_Pd[i], self.model.Sigma_Pd[i], log=False)
			self.model.Pd[i, :] = self.model.Pd[i, :] / (np.sum(self.model.Pd[i, :])+pbd.realmin)
		self.alpha_hsmm = None
		self.crossed = False
		self.started = False
		self.history = []
		self.predicted_segment = []
		self.nbD = nbD

		self.shoulder_tf = TransformStamped()
		self.shoulder_tf.header.frame_id = 'camera_color_optical_frame'
		self.shoulder_tf.child_frame_id = 'body_frame'

		self.viz_pub = rospy.Publisher('visualization_marker_array', MarkerArray, queue_size=10)
	
	def skeleton_transformation(self, nui_skeleton):
		# Origin at right shoulder, like in the training data
		rotation_normalization_matrix = euler_matrix(0,0,np.pi).T
		rotation_normalization_matrix[:3, 3] = nui_skeleton[joints_idx["right_shoulder"]-1:joints_idx["right_shoulder"], :]
		nui_skeleton -= rotation_normalization_matrix[:3, 3]
		rotation_normalization_matrix[:3, :3] = rotation_normalization_matrix[:3, :3].dot(rotation_normalization(nui_skeleton))
		nui_skeleton = rotation_normalization_matrix[:3,:3].dot(nui_skeleton.T).T

		return nui_skeleton, rotation_normalization_matrix

	def hsmm_init(self):
		p_obs, _ = self.model.obs_likelihood(self.history, None, slice(0, 6), 1)
		self.model._B = p_obs
		bmx, ALPHA, S, h = fwd_init(self.model, self.nbD, p_obs[:, 0])
		self.hsmm_forward = [bmx, ALPHA, S]
		return h/np.sum(h)

	def hsmm_step(self):
		bmx, ALPHA, S = self.hsmm_forward
		p_obs, _ = self.model.obs_likelihood(self.history[-1:], None, slice(0, 6), 1)
		self.model._B = np.vstack([self.model._B, p_obs])
		bmx, ALPHA, S, h = fwd_step(self.model, bmx, ALPHA, S, self.nbD, p_obs[:, 0])
		self.hsmm_forward = [bmx, ALPHA, S]
		return h/np.sum(h)
	
	def transform_hsmm(self, hsmm_transform):
		R = hsmm_transform[:3, :3]
		t = hsmm_transform[:3, 3]
		D = np.diag(np.array([1.0, 0.8, 2]))
		# D = np.eye(3)
		T = R.dot(D)
		T_ = D.T.dot(R.T)
		self.model.mu[:, 6] += 0.2
		self.model.mu[:, 7] += -1.
		self.model.mu[:, 8] += 0.45
		self.model.mu[:, 6:9] = D.dot(self.model.mu[:, 6:9].T).T
		if self.model.mu.shape[-1] > 9:
			self.model.mu[:, 9:] = D.dot(self.model.mu[:, 9:].T).T
		self.model.mu[:, 6:9] = R.dot(self.model.mu[:, 6:9].T).T
		if self.model.mu.shape[-1] > 9:
			self.model.mu[:, 9:] = R.dot(self.model.mu[:, 9:].T).T
		self.model.mu[:, 6] += t[0]
		self.model.mu[:, 7] += t[1]
		self.model.mu[:, 8] += t[2]
		for i in range(self.model.nb_states):
			self.model.sigma[i, :, 6:9] = self.model.sigma[i, :, 6:9].dot(T_)
			if self.model.mu.shape[-1] > 9:
				self.model.sigma[i, :, 9:] = self.model.sigma[i, :, 9:].dot(T_)
			self.model.sigma[i, 6:9, :] = T.dot(self.model.sigma[i, 6:9, :])
			if self.model.mu.shape[-1] > 9:
				self.model.sigma[i, 9:, :] = T.dot(self.model.sigma[i, 9:, :])

	def step(self, nui_skeleton, hand_pose):
		if len(nui_skeleton)==0 or hand_pose is None:
			return
		
		nui_skeleton, rotation_normalization_matrix = self.skeleton_transformation(nui_skeleton)
		
		if len(self.history) == 0:
			rotation_normalization_matrix[:3,:3] = rotation_normalization_matrix[:3,:3].T
			print(rotation_normalization_matrix)
			self.shoulder_tf.transform = mat2TF(rotation_normalization_matrix)
			self.history = np.array([np.concatenate([nui_skeleton[-1,:],np.zeros_like(nui_skeleton[-1,:])])])
			self.predicted_segment.append(self.hsmm_init().argmax())
			self.broadcaster.sendTransform([self.shoulder_tf])
			rospy.Rate(100).sleep()
			self.transform_hsmm(ROS2mat(self.tfBuffer.lookup_transform('body_frame', 'base_link', rospy.Time()).transform).dot(euler_matrix(0,0,np.pi/2+np.deg2rad(10))))
			self.markerarray_msg = rviz_gmm3d(self.model, dims = slice(0,3), rgb = [0,0,1], frame_id=self.shoulder_tf.child_frame_id)
			markerarray_msg = rviz_gmm3d(self.model, dims = slice(6,9), rgb = [1,0,0], frame_id=self.shoulder_tf.child_frame_id)
			for i in range(len(markerarray_msg.markers)):
				markerarray_msg.markers[i].id += self.model.nb_states
			self.markerarray_msg.markers += markerarray_msg.markers

			return

		if not self.started and ((nui_skeleton[-1,:] - self.history[-1, :3])**2).sum() < 0.001:
			print('Not yet started. Current displacement:', ((nui_skeleton[-1,:] - self.history[-1, :3])**2).sum())
			return
		else:
			self.started = True
		self.history = np.vstack([self.history, np.concatenate([nui_skeleton[-1,:], nui_skeleton[-1,:] - self.history[-1, :3]])])
		
		alpha_hsmm = self.hsmm_step()
		print(alpha_hsmm)
		if np.allclose(alpha_hsmm, alpha_hsmm[0], atol=0.0001):
			active_segment = self.predicted_segment[-1]
		else:
			active_segment = alpha_hsmm.argmax()
		self.predicted_segment.append(active_segment)
		for i in range(len(alpha_hsmm)):
			self.markerarray_msg.markers[i].color.a = max(0.2, alpha_hsmm[i])
			self.markerarray_msg.markers[i + len(alpha_hsmm)].color.a = max(0.2, alpha_hsmm[i])
		
		mu_est_hsmm, sigma_est_hsmm = pbd.Model.condition(self.model, self.history[-1:], dim_in=slice(0, 6), dim_out=slice(6, 9), h=alpha_hsmm)
		target_pose = rotation_normalization_matrix[:3,:3].T.dot(mu_est_hsmm[0]) + rotation_normalization_matrix[:3,3]
		target_pose = self.nuitrack.base2cam[:3,:3].dot(target_pose) + self.nuitrack.base2cam[:3,3]
		self.ik_result[2:6] = self.joint_readings[:4]
		super().step(nui_skeleton, target_pose)
	
	def publish(self, stamp):
		super().publish(stamp)
		if self.started:
			self.shoulder_tf.header.stamp = stamp
			self.broadcaster.sendTransform([self.shoulder_tf])
			self.viz_pub.publish(self.markerarray_msg)

if __name__=='__main__':
	rospy.init_node('hri_hsmmik_node')
	controller = HSMMIKController()
	controller.observe_human()
	rate = rospy.Rate(100)
	rate.sleep()
	count = 0
	def spin(publish=True):
		if publish:
			controller.publish(stamp)
		rate.sleep()

	while not rospy.is_shutdown():
		nui_skeleton, hand_pose, stamp = controller.observe_human()
		if len(nui_skeleton)==0 or hand_pose is None:
			spin(False)
			continue
		count += 1
		if count < 20:
			print(count)
			spin()
			continue
		controller.step(nui_skeleton, hand_pose)
		if controller.predicted_segment[-1] in [0, controller.model.nb_states - 1]:
			controller.joint_trajectory.points[0].effort[0] = 0.7
		# elif np.linalg.norm(np.array(controller.joint_readings[:4]) - ncontroller.joint_trajectory.points[0].effort[0]p.array(controller.joint_trajectory.points[0].positions[:4])) > 0.25 :
		elif len(controller.history) < 60:
			controller.joint_trajectory.points[0].effort[0] = 0.7
		else:
			controller.joint_trajectory.points[0].effort[0] = 0.1
		print(controller.joint_trajectory.points[0].effort[0])
		if sum(controller.predicted_segment[-6:])>=5*controller.model.nb_states or (len(controller.history) > 50 and controller.predicted_segment[-1]==controller.model.nb_states - 1): # if it's the last segment, then that means you're going back to the final position
			controller.state_msg.state.joint_state.position[11:17] = controller.joint_trajectory.points[0].positions = default_arm_joints
			spin()
			break
		
		spin()
