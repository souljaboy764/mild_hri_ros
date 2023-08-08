#!/usr/bin/python

import matplotlib.pyplot as plt
import torch
import cv2
import argparse

import mild_hri
import pbdlib_torch as pbd_torch

from utils import *
from nuitrack_node import NuitrackWrapper

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MILDHHIController:
	def __init__(self, ckpt_path):
		ckpt = torch.load(ckpt_path)
		# ['clapfist2', 'fistbump2', 'handshake2', 'highfive1', 'rocket1', 'wave1']
		self.ssm = ckpt['ssm'][2]
		self.mu = self.ssm.mu.cpu().numpy()
		self.sigma = self.ssm.sigma.cpu().numpy()
		self.vae = mild_hri.vae.VAE(**(ckpt['args'].__dict__)).to(device)
		self.vae.load_state_dict(ckpt['model'])
		self.vae.eval()

		self.nuitrack = NuitrackWrapper()
		self.z_dim = self.vae.latent_dim

		self.alpha_t = None
		self.crossed = False
		self.started = False
		self.history = []
		self.predicted_segment = []
		
		fig = plt.figure()
		# plt.ion()
		self.ax_skel = fig.add_subplot(2,1,1, projection='3d')
		self.ax_latent = fig.add_subplot(2,1,2, projection='3d')
		plt.show(block=False)

	def update_plot(self, zh, zr_cond, xr_cond):
		if not self.started:
			self.ax_latent.cla()
			for k in range(self.ssm.nb_states):
				pbd_torch.plot_gauss3d(self.ax_latent, self.mu[k, :3], self.sigma[k, :3, :3], color='r', alpha=1./self.ssm.nb_states)
				pbd_torch.plot_gauss3d(self.ax_latent, self.mu[k, self.vae.latent_dim:self.vae.latent_dim+3], self.sigma[k, self.vae.latent_dim:self.vae.latent_dim+3, self.vae.latent_dim:self.vae.latent_dim+3], color='b', alpha=1./self.ssm.nb_states)

			self.ax_skel.cla()
			self.ax_skel.set_xlabel('X')
			self.ax_skel.set_ylabel('Y')
			self.ax_skel.set_zlabel('Z')
			self.ax_skel.set_xlim(-0.05,0.75)
			self.ax_skel.set_ylim(-0.4,0.4)
			self.ax_skel.set_zlim(-0.6,0.2)
			
			if self.history != []:
				x_h = self.history[-1].reshape(self.vae.num_joints, self.vae.joint_dims).cpu().numpy()
				self.ax_skel.plot(x_h[:,0], x_h[:,1], x_h[:,2], 'k-', markerfacecolor='r', marker='o')

			mild_hri.utils.mypause(0.001)
			
			return

		self.ax_skel.cla()
		self.ax_skel.set_xlabel('X')
		self.ax_skel.set_ylabel('Y')
		self.ax_skel.set_zlabel('Z')
		self.ax_skel.set_xlim(-0.05,0.75)
		self.ax_skel.set_ylim(-0.4,0.4)
		self.ax_skel.set_zlim(-0.6,0.2)
		x_h = self.history.reshape(self.vae.window_size, self.vae.num_joints, self.vae.joint_dims).cpu().numpy()
		xr_cond = xr_cond.reshape((self.vae.window_size, self.vae.num_joints, self.vae.joint_dims))
		for w in range(self.vae.window_size):
			self.ax_skel.plot(x_h[w,:,0], x_h[w,:,1], x_h[w,:,2], 'k-', markerfacecolor='r', marker='o', alpha=(w+1)/self.vae.window_size)
			self.ax_skel.plot(0.7 - xr_cond[w,:,0], -xr_cond[w,:,1], xr_cond[w,:,2], 'k--', markerfacecolor='c', marker='o', alpha=(w+1)/self.vae.window_size)

		alpha = self.alpha_t.cpu().numpy()
		print(alpha.argmax(), alpha)
		# self.ax_latent.cla()
		# for k in range(self.ssm.nb_states):
		# 	pbd_torch.plot_gauss3d(self.ax_latent, self.mu[k, :3], self.sigma[k, :3, :3], color='r', alpha=max(0.1, alpha[k])*0.8)
		# 	pbd_torch.plot_gauss3d(self.ax_latent, self.mu[k, self.vae.latent_dim:self.vae.latent_dim+3], self.sigma[k, self.vae.latent_dim:self.vae.latent_dim+3, self.vae.latent_dim:self.vae.latent_dim+3], color='b', alpha=max(0.1, alpha[k])*0.8)
		self.ax_latent.scatter3D(zh[0], zh[1], zh[2], c='r')
		self.ax_latent.scatter3D(zr_cond[0], zr_cond[1], zr_cond[2], c='c')
		mild_hri.utils.mypause(0.001)

	def observe_human(self, nui_skeleton):
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

	def step(self):
		if not self.started:
			return
			
		zh_post = self.vae(self.history.flatten()[None], dist_only=True)

		B, _ = self.ssm.obs_likelihood(zh_post.mean, marginal=slice(0, self.z_dim))
		if self.alpha_t is None:
			alpha_t = self.ssm.init_priors * B[:, 0]
		else:
			alpha_t = self.alpha_t.matmul(self.ssm.Trans) * B[:, 0]
		self.alpha_t = alpha_t / (torch.sum(alpha_t) + pbd_torch.realmin)
		# self.alpha_t = self.ssm.forward_variable(,  alpha_0=self.alpha_t)
		# self.alpha_t = self.alpha_t[:, 0]
		# print(self.alpha_t)
		if self.vae.cov_cond:
			data_Sigma_in = zh_post.covariance_matrix
		else:
			data_Sigma_in = None

		zr_cond = self.ssm.condition(zh_post.mean, dim_in=slice(0, self.z_dim), dim_out=slice(self.z_dim, 2*self.z_dim), h=self.alpha_t,
												return_cov=False, data_Sigma_in=data_Sigma_in)
		xr_cond = self.vae._output(self.vae._decoder(zr_cond))
		if torch.allclose(self.alpha_t, self.alpha_t[0], atol=0.0001): # someimtes if the movement is out of distribution it can be nan or all will be similar
			active_segment = self.predicted_segment[-1]
		else:
			active_segment = self.alpha_t.argmax()
		self.predicted_segment.append(active_segment)

		self.update_plot(zh_post.mean[0].cpu().numpy(), zr_cond[0].cpu().numpy(), xr_cond[0].cpu().numpy())

if __name__=='__main__':
	with torch.no_grad():
		parser = argparse.ArgumentParser(description='Nuitrack HH Testing')
		parser.add_argument('--ckpt', type=str, required=True, metavar='CKPT',
								help='Checkpoint to resume training from (default: None)')
		args = parser.parse_args()
		controller = MILDHHIController(args.ckpt)
		controller.nuitrack.update()
		controller.nuitrack.update()
		window_name = 'Nuitrack'
		cv2.namedWindow(window_name)
		count = 0
		hand_pos = []
		while True:
			img, nui_skeleton = controller.nuitrack.update()
			cv2.imshow(window_name, img)#.transpose((1,0,2)))
			cv2.waitKey(1)
			if len(nui_skeleton)==0:
				continue
			count += 1
			
			if count<30:
				hand_pos.append(nui_skeleton[-2])
				continue
			elif count == 30:
				hand_pos = np.mean(hand_pos, 0)
				print('Calibration done')
			nui_skeleton_scaled = nui_skeleton/hand_pos[2]
			controller.observe_human(nui_skeleton_scaled)
			# controller.step()
			# if count > 30 and ((nui_skeleton[-2] - hand_pos)**2).sum() < 0.001:
				# break
			print(((nui_skeleton[-2] - hand_pos)**2).sum(), nui_skeleton[-2], hand_pos)
		plt.show()
			