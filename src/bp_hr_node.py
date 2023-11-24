#!/usr/bin/python
import rospy
import matplotlib.pyplot as plt
import torch
import cv2
import argparse
import os

import datetime

import mild_hri
import bp_hri
import pbdlib_torch as pbd_torch
from pbdlib_torch.functions import multi_variate_normal

from utils import *
from bp_hri.utils import *
from nuitrack_node import NuitrackWrapper
from base_ik_node import BaseIKController, default_arm_joints

from std_msgs.msg import Empty

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class BPHRIController(BaseIKController):
	def __init__(self, args):
		super().__init__()
		# NuiSI ['clapfist2', 'fistbump2', 'handshake2', 'highfive1', 'rocket1', 'wave1']
		# Buetepage ['waving', 'handshake', 'rocket', 'parachute']
		self.args = args
		if args.action == 'handshake':
			self.label = torch.Tensor(np.array([1,1,1,1.])).to(device)
			self.robot_hand_joint = 0.7
			
		if args.action == 'rocket':
			self.robot_hand_joint = 0.
			self.label = torch.Tensor(np.array([1,1,1,1.])).to(device)

		MODELS_FOLDER = os.path.join('models','bp_hri')
		
		robot_vae_hyperparams = np.load(os.path.join(MODELS_FOLDER,'hyperparams.npz'), allow_pickle=True)
		robot_vae_config = robot_vae_hyperparams['vae_config'].item()
		self.robot_vae = bp_hri.networks.VAE(**(robot_vae_config.__dict__)).to(device)
		ckpt = torch.load(os.path.join(MODELS_FOLDER,'robot_vae.pth'))
		self.robot_vae.load_state_dict(ckpt['model'])
		self.robot_vae.eval()

		human_tdm_hyperparams = np.load(os.path.join(MODELS_FOLDER,'tdm_hyperparams.npz'), allow_pickle=True)
		human_tdm_config = human_tdm_hyperparams['tdm_config'].item()
		self.human_tdm = bp_hri.networks.TDM(**(human_tdm_config.__dict__)).to(device)
		ckpt = torch.load(os.path.join(MODELS_FOLDER,'tdm.pth'))
		self.human_tdm.load_state_dict(ckpt['model_1'])
		self.human_tdm.eval()

		hri_hyperparams = np.load(os.path.join(MODELS_FOLDER,'hri_hyperparams.npz'), allow_pickle=True)
		hri_config = hri_hyperparams['hri_config'].item()
		self.hri = bp_hri.networks.HRIDynamics(**(hri_config.__dict__)).to(device)
		ckpt = torch.load(os.path.join(MODELS_FOLDER,'hri.pth'))
		self.hri.load_state_dict(ckpt['model'])
		self.hri.eval()

		# Model Warmup https://medium.com/log-loss/model-warmup-8e9681ef4d41
		self.robot_vae(torch.zeros((100,self.robot_vae.input_dim),device=device))
		self.human_tdm(torch.zeros((100,self.human_tdm.input_dim),device=device), None)
		self.hri(torch.zeros((100,self.hri.input_dim),device=device), None)
		
		self.started = False
		self.history = []
		self.still_reaching = True
		
		if self.args.action=='handshake':
			self.joint_trajectory.points[0].effort[0] = 0.4
		else:
			self.joint_trajectory.points[0].effort[0] = 0.7

		self.is_still = True
		self.still_sub = rospy.Subscriber('/is_still', Empty, self.stillness)
		self.last_pred = None

		self.dh = []
		self.zr_hri = []

	def stillness(self, msg):
		self.is_still = True

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

		x_p1_tdm = torch.cat([self.history[-1], self.label])[None]
		if self.last_pred is None:
			x_r2_hri = torch.cat([torch.Tensor(self.joint_readings[:4]).to(device), torch.ones_like(self.label)])[None]
		else:
			x_r2_hri = torch.cat([torch.Tensor(self.last_pred).to(device), torch.zeros_like(self.label)])[None]
		# x_r2_hri[0,0] += np.deg2rad(15)
		_, _, d_x1_dist = self.human_tdm(x_p1_tdm, None)
		hri_input = torch.concat([x_r2_hri, d_x1_dist.mean], dim=-1)
		z_r2hri_dist, z_r2hri_samples = self.hri(hri_input, None)
		x_r2_gen = self.robot_vae._output(self.robot_vae._decoder(z_r2hri_dist.mean))

		self.zr_hri.append(z_r2hri_dist.mean.cpu().numpy()[0])
		self.dh.append(d_x1_dist.mean.cpu().numpy()[0])
		
		self.last_pred = x_r2_gen.reshape((self.robot_vae.window_size, self.robot_vae.num_joints))[0].cpu().numpy()#.mean(0).cpu().numpy()

		print(self.history[-1], x_r2_hri[0,:4], d_x1_dist.mean)
		
		self.joint_trajectory.points[0].positions = 0.2*np.array(self.joint_trajectory.points[0].positions) + 0.8*np.array(self.last_pred.tolist() + [1., self.robot_hand_joint])
		self.joint_trajectory.points[0].positions[0] -= np.deg2rad(15)
		
		# if args.ik_only: # Putting the baseline here to simulate the same perception-reaction delay as with the network
		# 	super().step(nui_skeleton, hand_pose, optimizer="least_squares")
		# 	if self.args.action=='handshake' and self.joint_readings[0] < 0.35:
		# 		self.joint_trajectory.points[0].effort[0] = 0.15
		# elif (active_segment>0 and active_segment<self.ssm.nb_states-1) or \
		# 	(self.args.action=='handshake' and self.crossed) or \
		# 	(self.args.action=='rocket' and active_segment==0 and self.alpha_t[active_segment]<0.6) or \
		# 	(active_segment==self.ssm.nb_states-1 and self.alpha_t[active_segment]<0.6):
		# 	if (self.args.action=='handshake' and active_segment==0 and self.alpha_t[active_segment]<transition_state_prob):
		# 		print('Transition state')
		if self.args.ik:
			self.ik_result = self.pepper_chain.active_to_full(self.joint_trajectory.points[0].positions[0:4], [0] * len(self.pepper_chain.links))
			super().step(nui_skeleton, hand_pose, optimizer="scalar",  regularization_parameter=0.01)
		# 	if self.args.action=='handshake': # Handshake Stiffness control
		# 		print(active_segment, self.still_reaching, np.linalg.norm(self.joint_readings[:4] - self.joint_trajectory.points[0].positions[0:4]), np.linalg.norm(self.joint_readings[0] - self.joint_trajectory.points[0].positions[0]))
		# 		if self.still_reaching:
		# 			if np.linalg.norm(self.joint_readings[:4] - self.joint_trajectory.points[0].positions[0:4]) < 0.1:
		# 				self.still_reaching = False
		# 				self.joint_trajectory.points[0].effort[0] = 0.15
		# 		else:
		# 			self.joint_trajectory.points[0].effort[0] = 0.15
		# else:# self.args.action=='handshake':
		# 	if self.joint_trajectory.points[0].effort[0] < 0.5:
		# 		self.joint_trajectory.points[0].effort[0] += 0.01

		# self.predicted_segment.append(active_segment)

if __name__=='__main__':
	with torch.no_grad():
		parser = argparse.ArgumentParser(description='Nuitrack HR Testing')
		parser.add_argument('--action', type=str, required=True, metavar='ACTION', choices=['handshake', 'rocket'],
						help='Action to perform: handshake or rocket).')
		parser.add_argument('--ik', action="store_true",
								help='Flag for setting whether to use IK or not')
		args = parser.parse_args()
		rospy.init_node('mild_hri_node')
		rate = rospy.Rate(100)
		controller = BPHRIController(args)
		controller.observe_human()
		controller.observe_human()
		count = 0
		hand_pos = []
		while not rospy.is_shutdown():
			nui_skeleton, hand_pose, stamp = controller.observe_human()
			if len(nui_skeleton)==0:
				continue

			if controller.is_still:
				count += 1
			else:
				controller.publish(stamp)
				continue
			
			if count<70:
				hand_pos.append(nui_skeleton[-2])
				continue
			elif count == 70:
				hand_pos = np.mean(hand_pos, 0)
				print('Calibration done')
			# start = datetime.datetime.now()
			controller.step(nui_skeleton.copy(), hand_pose)
			controller.publish(stamp)
			# print(controller.started, ((nui_skeleton[-2] - hand_pos)**2).sum(), nui_skeleton[-2], hand_pos)
			if count > 200 and controller.started and ((nui_skeleton[-2] - hand_pos)**2).sum() < 0.005:
				controller.joint_trajectory.points[0].effort[0] = min(0.5, controller.joint_trajectory.points[0].effort[0])
				controller.joint_trajectory.points[0].positions = default_arm_joints
				controller.publish(rospy.Time.now())
				rospy.Rate(1).sleep()
				controller.publish(rospy.Time.now())
				rospy.Rate(1).sleep()
				rospy.signal_shutdown('Done')
	np.savez_compressed('bp_latents_realtime.npz', zr_hri=np.array(controller.zr_hri), dh=np.array(controller.dh))