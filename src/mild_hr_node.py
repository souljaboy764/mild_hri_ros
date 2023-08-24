#!/usr/bin/python

import matplotlib.pyplot as plt
import torch
import cv2
import argparse

import mild_hri
import pbdlib_torch as pbd_torch

from utils import *
from nuitrack_node import NuitrackWrapper

from base_ik_node import BaseIKController

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MILDHRIController(BaseIKController):
	def __init__(self, ckpt_path):
		super().__init__()
		ckpt = torch.load(ckpt_path)
		# NuiSI ['clapfist2', 'fistbump2', 'handshake2', 'highfive1', 'rocket1', 'wave1']
		# Buetepage ['waving', 'handshake2', 'rocket', 'parachute']
		self.ssm = ckpt['ssm'][1]
		self.mu = self.ssm.mu.cpu().numpy()
		self.sigma = self.ssm.sigma.cpu().numpy()
		self.model_h = mild_hri.vae.VAE(**(ckpt['args_h'].__dict__)).to(device)
		self.model_h.load_state_dict(ckpt['model_h'])
		self.model_h.eval()
		self.model_r = mild_hri.vae.VAE(**{**(ckpt['args_h'].__dict__), **(ckpt['args_r'].__dict__)}).to(device)
		self.model_r.load_state_dict(ckpt['model_r'])
		self.model_r.eval()

		self.z_dim = self.model_h.latent_dim

		self.alpha_t = None
		self.crossed = False
		self.started = False
		self.history = []
		self.predicted_segment = []

	def step(self, nui_skeleton, hand_pose, ik=False):
		nui_skeleton, _ = skeleton_transformation(nui_skeleton)
		nui_skeleton = torch.Tensor(nui_skeleton).to(device)
		if self.history == []:
			self.history = nui_skeleton[-5:-1, :].flatten()[None]
		if not self.started and ((nui_skeleton[-1,:] - self.history[-1, -3:])**2).sum() < 0.0004:
			print('Not yet started. Current displacement:', ((nui_skeleton[-1,:] - self.history[-1, -3:])**2).sum())
			self.history = nui_skeleton[-4:, :].flatten()[None]
			self.update_plot(None, None, None)
			return
		
		self.history = torch.vstack([self.history, nui_skeleton[-5:-1, :].flatten()[None]])
		
		if self.history.shape[0] < 5:
			self.update_plot(None, None, None)
			return
		self.started = True
		self.history = self.history[-5:]

		if not self.started:
			return
			
		zh_post = self.model_h(self.history.flatten()[None], dist_only=True)

		B, _ = self.ssm.obs_likelihood(zh_post.mean, marginal=slice(0, self.z_dim))
		if self.alpha_t is None:
			alpha_t = self.ssm.init_priors * B[:, 0]
		else:
			alpha_t = self.alpha_t.matmul(self.ssm.Trans) * B[:, 0]
		self.alpha_t = alpha_t / (torch.sum(alpha_t) + pbd_torch.realmin)
		# print(self.alpha_t)
		if self.model_h.cov_cond:
			data_Sigma_in = zh_post.covariance_matrix
		else:
			data_Sigma_in = None

		zr_cond = self.ssm.condition(zh_post.mean, dim_in=slice(0, self.z_dim), dim_out=slice(self.z_dim, 2*self.z_dim), h=self.alpha_t,
												return_cov=False, data_Sigma_in=data_Sigma_in)
		xr_cond = self.model_r._output(self.model_r._decoder(zr_cond))
		if torch.allclose(self.alpha_t, self.alpha_t[0], atol=0.0001): # someimtes if the movement is out of distribution it can be nan or all will be similar
			active_segment = self.predicted_segment[-1]
		else:
			active_segment = self.alpha_t.argmax()
		self.predicted_segment.append(active_segment)

		q = xr_cond.reshape((self.model_r.window_length, self.model_r.num_joints)).mean(0).cpu().numpy()
		# Handshake IK Stiffness control
		if ik:
			if ((active_segment>0 and active_segment<self.ssm.nb_states-1) or ((active_segment==0 or active_segment==self.ssm.nb_states-1) and self.alpha_t[active_segment]<0.6)):
				self.ik_result = q
				super().step(self, nui_skeleton, hand_pose, regularization_parameter=0.5)
		else:
			self.joint_trajectory.points[0].positions = 0.5*np.array(self.joint_trajectory.points[0].positions) + 0.5*np.array(q.tolist() + [1.8239, 0])


if __name__=='__main__':
	with torch.no_grad():
		parser = argparse.ArgumentParser(description='Nuitrack HR Testing')
		parser.add_argument('--ckpt', type=str, required=True, metavar='CKPT',
								help='Checkpoint to resume training from (default: None)')
		parser.add_argument('--ik', action="store_true", metavar='IK',
								help='Flag for setting whether to use IK or not')
		args = parser.parse_args()
		rospy.init_node('mild_hri_node')
		rate = rospy.Rate(100)
		controller = MILDHRIController(args.ckpt)
		controller.observe_human()
		controller.observe_human()
		count = 0
		hand_pos = []
		while not rospy.is_shutdown():
			nui_skeleton, hand_pose, stamp = controller.observe_human()
			if len(nui_skeleton)==0:
				continue
			count += 1
			
			if count<30:
				hand_pos.append(hand_pose)
				continue
			elif count == 30:
				hand_pos = np.mean(hand_pos, 0)
				print('Calibration done')
			controller.step(nui_skeleton, hand_pose, args.ik)
			if count > 30 and ((nui_skeleton[-2] - hand_pos)**2).sum() < 0.001:
				break
			print(((nui_skeleton[-2] - hand_pos)**2).sum(), nui_skeleton[-2], hand_pos)
			