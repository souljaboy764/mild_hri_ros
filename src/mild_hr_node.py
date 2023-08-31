#!/usr/bin/python

import matplotlib.pyplot as plt
import torch
import cv2
import argparse

import datetime

import mild_hri
import pbdlib_torch as pbd_torch

from utils import *
from nuitrack_node import NuitrackWrapper

from base_ik_node import BaseIKController, default_arm_joints

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MILDHRIController(BaseIKController):
	def __init__(self, args):
		super().__init__()
		ckpt = torch.load(args.ckpt)
		# NuiSI ['clapfist2', 'fistbump2', 'handshake2', 'highfive1', 'rocket1', 'wave1']
		# Buetepage ['waving', 'handshake', 'rocket', 'parachute']
		self.args = args
		if args.action == 'handshake':
			self.ssm = ckpt['ssm'][1]
			self.robot_hand_joint = 0.7
			self.q_var_inv = np.array([ 12.63, 204.53,  40.24,  11.73])
		if args.action == 'rocket':
			self.ssm = ckpt['ssm'][2]
			self.robot_hand_joint = 0.
			self.q_var_inv = np.array([ 5.4 , 60.67, 12.76,  5.72])
		self.mu = self.ssm.mu.cpu().numpy()
		self.sigma = self.ssm.sigma.cpu().numpy()
		self.model_h = mild_hri.vae.VAE(**(ckpt['args_h'].__dict__)).to(device)
		self.model_h.load_state_dict(ckpt['model_h'])
		self.model_h.eval()
		self.model_r = mild_hri.vae.VAE(**{**(ckpt['args_h'].__dict__), **(ckpt['args_r'].__dict__)}).to(device)
		self.model_r.load_state_dict(ckpt['model_r'])
		self.model_r.eval()
		self.z_dim = self.model_h.latent_dim

		# Model Warmup https://medium.com/log-loss/model-warmup-8e9681ef4d41
		self.model_h(torch.zeros((100,self.model_h.input_dim),device=device))
		self.model_r(torch.zeros((100,self.model_r.input_dim),device=device))
		if self.model_h.cov_cond:
			data_Sigma_in = torch.eye(self.z_dim, device=device)[None].repeat(100,1,1)
		else:
			data_Sigma_in = None
		self.ssm.condition(torch.ones((100,self.z_dim), device=device), dim_in=slice(0, self.z_dim), dim_out=slice(self.z_dim, 2*self.z_dim),
											return_cov=False, data_Sigma_in=data_Sigma_in)

		self.alpha_t = None
		self.crossed = False
		self.started = False
		self.history = []
		self.still_reaching = True
		self.predicted_segment = []

		self.joint_trajectory.points[0].effort[0] = 1.0

	def step(self, nui_skeleton, hand_pose):
		if self.history == []:
			self.preproc_transformation = rotation_normalization(nui_skeleton)

		nui_skeleton = torch.Tensor(self.preproc_transformation[:3,:3].dot(nui_skeleton.T).T + self.preproc_transformation[:3,3]).to(device)
		if self.history == []:
			x_pos = nui_skeleton[[-4,-3,-2], :].flatten()[None]
			self.history = torch.cat([x_pos, torch.zeros_like(x_pos)], dim=-1)
		if not self.started and ((nui_skeleton[-2,:] - self.history[-1, 6:9])**2).sum() < 0.0005:
			print('Not yet started. Current displacement:', ((nui_skeleton[-2,:] - self.history[-1, 6:9])**2).sum())
			return
		
		x_pos = nui_skeleton[[-4,-3,-2], :].flatten()[None]
		x_vel = x_pos - self.history[-1, :9]
		self.history = torch.vstack([self.history, torch.cat([x_pos, x_vel], dim=-1)])
		
		if self.history.shape[0] < 5:
			return
		if not self.started:
			print('Starting',((nui_skeleton[-2,:] - self.history[-1, 6:9])**2).sum())
		self.started = True
		self.history = self.history[-5:]
		
		zh_post = self.model_h(self.history.flatten()[None], dist_only=True)

		B, _ = self.ssm.obs_likelihood(zh_post.mean, marginal=slice(0, self.z_dim))
		if self.alpha_t is None:
			alpha_t = self.ssm.init_priors * B[:, 0]
		else:
			alpha_t = self.alpha_t.matmul(self.ssm.Trans) * B[:, 0]
		alpha_norm = torch.sum(alpha_t)
		if torch.any(torch.isnan(alpha_t)) or torch.allclose(alpha_norm, torch.zeros_like(alpha_norm)):
			alpha_t = self.ssm.init_priors * B[:, 0] + 1e-6
			alpha_norm = torch.sum(alpha_t) + 1e-6
			self.alpha_t = alpha_t / alpha_norm
		else:
			self.alpha_t = alpha_t / (alpha_norm + pbd_torch.realmin)

		if self.model_h.cov_cond:
			data_Sigma_in = zh_post.covariance_matrix
		else:
			data_Sigma_in = None

		zr_cond = self.ssm.condition(zh_post.mean, dim_in=slice(0, self.z_dim), dim_out=slice(self.z_dim, 2*self.z_dim), h=self.alpha_t.reshape((self.ssm.nb_states, 1)),
												return_cov=False, data_Sigma_in=data_Sigma_in)
		xr_cond = self.model_r._output(self.model_r._decoder(zr_cond))
		if torch.allclose(self.alpha_t, self.alpha_t[0], atol=0.0001): # someimtes if the movement is out of distribution it can be nan or all will be similar
			if len(self.predicted_segment)>0:
				active_segment = self.predicted_segment[-1]
			else:
				active_segment = 0
		else:
			active_segment = self.alpha_t.argmax()
		self.predicted_segment.append(active_segment)
		# print(active_segment.item(), self.alpha_t[active_segment].item(), self.alpha_t.tolist(), torch.isnan(B[:, 0]).tolist())
		q = xr_cond.reshape((self.model_r.window_size, self.model_r.num_joints)).mean(0).cpu().numpy()
		# Handshake IK Stiffness control
		self.joint_trajectory.points[0].positions = 0.2*np.array(self.joint_trajectory.points[0].positions) + 0.8*np.array(q.tolist() + [1., self.robot_hand_joint])
		self.joint_trajectory.points[0].positions[0] -= np.deg2rad(15)
		
		if args.ik_only: # Putting the baseline here to simulate the same perception-reaction delay as with the network
			super().step(nui_skeleton, hand_pose, optimizer="least_squares")
			if self.args.action=='handshake' and self.joint_readings[0] < 0.3:
				self.joint_trajectory.points[0].effort[0] = 0.1
		elif (active_segment>0 and active_segment<self.ssm.nb_states-1) or ((active_segment==0 or active_segment==self.ssm.nb_states-1) and self.alpha_t[active_segment]<0.6):
			if self.args.ik:
				self.ik_result = self.pepper_chain.active_to_full(self.joint_trajectory.points[0].positions[0:4], [0] * len(self.pepper_chain.links))
				super().step(nui_skeleton, hand_pose, optimizer="least_squares",  regularization_parameter=0.01)
			if self.args.action=='handshake':
				if self.still_reaching:
					if np.linalg.norm(self.joint_readings[:4] - self.joint_trajectory.points[0].positions[0:4]) < 0.09:
						self.still_reaching = False
				else:
					self.joint_trajectory.points[0].effort[0] = 0.1
		else:# self.args.action=='handshake':
			if self.joint_trajectory.points[0].effort[0] < 0.5:
				self.joint_trajectory.points[0].effort[0] += 0.01


if __name__=='__main__':
	with torch.no_grad():
		parser = argparse.ArgumentParser(description='Nuitrack HR Testing')
		parser.add_argument('--ckpt', type=str, required=True, metavar='CKPT',
								help='Checkpoint to resume training from (default: None)')
		parser.add_argument('--ik', action="store_true",
								help='Flag for setting whether to use IK or not')
		parser.add_argument('--ik-only', action="store_true",
								help='Flag for setting whether to use the Base IK or not')
		parser.add_argument('--action', type=str, required=True, metavar='ACTION', choices=['handshake', 'rocket'],
						help='Action to perform: handshake or rocket).')
		args = parser.parse_args()
		rospy.init_node('mild_hri_node')
		rate = rospy.Rate(100)
		controller = MILDHRIController(args)
		controller.observe_human()
		controller.observe_human()
		count = 0
		hand_pos = []
		rospy.Rate(0.5).sleep()
		while not rospy.is_shutdown():
			nui_skeleton, hand_pose, stamp = controller.observe_human()
			if len(nui_skeleton)==0:
				continue
			count += 1
			
			if count<30:
				hand_pos.append(nui_skeleton[-2])
				continue
			elif count == 30:
				hand_pos = np.mean(hand_pos, 0)
				print('Calibration done')
			start = datetime.datetime.now()
			controller.step(nui_skeleton.copy(), hand_pose)
			controller.publish(stamp)
			# print(controller.started, ((nui_skeleton[-2] - hand_pos)**2).sum(), nui_skeleton[-2], hand_pos)
			if count > 70 and controller.started and ((nui_skeleton[-2] - hand_pos)**2).sum() < 0.005:
				# rospy.signal_shutdown('Done')
				break
	
		controller.joint_trajectory.points[0].effort[0] = 1.0
		controller.joint_trajectory.points[0].positions = default_arm_joints
		controller.publish(rospy.Time.now())
		controller.publish(rospy.Time.now())
		rospy.Rate(0.5).sleep()
		rospy.signal_shutdown('Done')