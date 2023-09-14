#!/usr/bin/python

import matplotlib.pyplot as plt
import torch
import cv2
import argparse

import datetime

import mild_hri
import pbdlib_torch as pbd_torch
from pbdlib_torch.functions import multi_variate_normal

from utils import *
from nuitrack_node import NuitrackWrapper
from base_ik_node import BaseIKController, default_arm_joints

from std_msgs.msg import Empty

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
			self.transition_state = pbd_torch.GMM(1, self.ssm.nb_dim)
			self.transition_state.mu = torch.Tensor(np.array([[-1.9675896062570457, -1.923336698728449, -0.37299678518491625, -1.0331682597889618, 0.18544509568635156, -2.042820166139041, -1.951088730026694, -0.014375370324534528, -0.8148276104646571, 0.4311470345539205]])).to(device)
			self.transition_state.sigma = torch.Tensor(np.array([[[0.00957983774738249, 0.004574332781173601, 0.0024715991341523547, 0.0030348719403180046, -0.0007205742085409716, 0.010752520093476674, -0.001493834343930241, 0.003500224653460497, -0.0017659381669538584, -0.002105599144907377], [0.004574332781173601, 0.004839884960559701, 0.0005968331178778182, 0.0031698625597272974, 0.0006732461409940893, -0.00043778333599439503, -0.002188319795936592, -0.0006709640995806779, -0.0017372348194722765, 0.0019774956729364255], [0.0024715991341523547, 0.0005968331178778184, 0.0025502073445139235, 0.001366884647949878, -0.0007270015693428607, 0.008290561054616539, -0.0010820517704194965, 0.0030944048193008444, 9.163908046253943e-05, -0.002643690264210873], [0.0030348719403180038, 0.0031698625597272974, 0.001366884647949878, 0.006451350105787166, 0.001165569942854926, -0.0013781137961501972, -0.0038496947187874287, 0.00016107772158065973, -0.0013316730308901687, 0.005114580493759031], [-0.0007205742085409718, 0.0006732461409940893, -0.0007270015693428607, 0.001165569942854926, 0.0012220755878306025, -0.004875410611232933, -0.0008380589083944642, -0.001635473288767781, -0.000964531915715876, 0.0024956132592148725], [0.010752520093476674, -0.00043778333599439547, 0.008290561054616539, -0.0013781137961501968, -0.004875410611232933, 0.041787018338494825, 0.00040931666925617864, 0.01545500473353755, 0.0018523518112009793, -0.017776936903791233], [-0.001493834343930241, -0.002188319795936592, -0.0010820517704194965, -0.0038496947187874287, -0.0008380589083944642, 0.0004093166692561782, 0.003274054225896396, 1.756891014996129e-05, 0.0010912516677778887, -0.0027168012749139046], [0.003500224653460498, -0.0006709640995806777, 0.0030944048193008444, 0.00016107772158065973, -0.001635473288767781, 0.01545500473353755, 1.7568910149961316e-05, 0.0070943100125677875, 0.0011623827874170905, -0.006019603317754002], [-0.0017659381669538586, -0.0017372348194722765, 9.16390804625394e-05, -0.0013316730308901687, -0.000964531915715876, 0.0018523518112009791, 0.0010912516677778887, 0.0011623827874170905, 0.0020305419691392067, -0.0009090471561462323], [-0.0021055991449073774, 0.0019774956729364255, -0.002643690264210874, 0.005114580493759031, 0.0024956132592148725, -0.017776936903791233, -0.002716801274913905, -0.006019603317754002, -0.0009090471561462321, 0.012171442316050987]]])).to(device)

		if args.action == 'rocket':
			self.ssm = ckpt['ssm'][2]
			self.robot_hand_joint = 0.
			self.q_var_inv = np.array([ 5.4 , 60.67, 12.76,  5.72])
			self.transition_state = pbd_torch.GMM(1, self.ssm.nb_dim)
			self.transition_state.mu = torch.Tensor(np.array([[-2.1719038262963295, -1.7572695426642895, -0.2571734532248229, -0.5487183891236782, 0.4382002092897892, -2.6759870797395706, -2.016182992607355, -0.1125284016598016, -0.7959552519023418, 0.7181216403841972]])).to(device)
			self.transition_state.sigma = torch.Tensor(np.array([[[0.012632379219419888, 0.00012220196838977349, -0.003803978942719178, 0.0051977695040900385, -8.127316009018393e-05, 0.001497771859861774, 0.0004906212750620254, 0.0032398664278201587, -0.0028868325167692166, -0.00453049388180321], [0.00012220196838977349, 0.003037707334253101, -0.0005811895939313739, 0.0040873590281343235, 0.0011994972338637416, -0.0009580319228041367, -0.0017027730285675141, -0.00048438121675491034, 0.003930149290761062, 0.0029866421743388882], [-0.003803978942719178, -0.0005811895939313739, 0.004007686017490798, -0.005590592067893152, -0.0011833230284382763, 0.0024306999006719664, 0.0011659366023175396, -0.0009878982014802189, -0.0015159506622007075, -0.0021005653466097157], [0.0051977695040900385, 0.0040873590281343235, -0.005590592067893152, 0.01648992754341218, 0.004677018791507431, -0.004087943206574296, -0.004158756214688399, 0.002040353816094129, 0.006799345059538315, 0.005951742887044337], [-8.127316009018393e-05, 0.0011994972338637416, -0.0011833230284382763, 0.004677018791507431, 0.0019990402431167872, -0.001455302163019867, -0.0007993328951538675, 0.000290519381719372, 0.0025085501369138535, 0.0021244258914275005], [0.001497771859861774, -0.0009580319228041367, 0.0024306999006719664, -0.004087943206574296, -0.001455302163019867, 0.029486015921692633, 0.009161249536808913, -0.0033778632445439255, 0.016606244991639818, 0.000927437355404126], [0.0004906212750620254, -0.0017027730285675141, 0.0011659366023175396, -0.004158756214688399, -0.0007993328951538675, 0.009161249536808913, 0.00784912845087312, 0.0010716874448362635, 0.005182904091500648, -0.0010504141507925724], [0.0032398664278201587, -0.00048438121675491034, -0.0009878982014802189, 0.002040353816094129, 0.000290519381719372, -0.0033778632445439255, 0.0010716874448362635, 0.004746312127751783, -0.0022660370097463602, -0.001128739919313278], [-0.0028868325167692166, 0.003930149290761062, -0.0015159506622007075, 0.006799345059538315, 0.0025085501369138535, 0.016606244991639818, 0.005182904091500648, -0.0022660370097463602, 0.026403821963669936, 0.013139737533809086], [-0.00453049388180321, 0.0029866421743388882, -0.0021005653466097157, 0.005951742887044337, 0.0021244258914275005, 0.000927437355404126, -0.0010504141507925724, -0.001128739919313278, 0.013139737533809086, 0.010990330094474521]]])).to(device)

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
		self.predicted_segment = [0]

		if self.args.action=='handshake':
			self.joint_trajectory.points[0].effort[0] = 0.4
		else:
			self.joint_trajectory.points[0].effort[0] = 0.7

		self.is_still = False
		self.still_sub = rospy.Subscriber('/is_still', Empty, self.stillness)

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
		
		zh_post = self.model_h(self.history.flatten()[None], dist_only=True)

		B, _ = self.ssm.obs_likelihood(zh_post.mean, marginal=slice(0, self.z_dim))
		
		if self.alpha_t is None:
			alpha_t = self.ssm.init_priors * B[:, 0]
		else:
			alpha_t = self.alpha_t.matmul(self.ssm.Trans) * B[:, 0]
		alpha_norm = torch.sum(alpha_t)
		self.alpha_t = alpha_t / (alpha_norm + pbd_torch.realmin)
		if torch.any(torch.isnan(self.alpha_t)) or torch.allclose(self.alpha_t, torch.zeros_like(self.alpha_t), atol=1e-15):
			print('Nans or zeros', alpha_t.tolist(), B[:, 0].tolist())
			alpha_t = self.ssm.init_priors * B[:, 0] + 1e-6
			alpha_norm = torch.sum(alpha_t) + 1e-6
			self.alpha_t = alpha_t / alpha_norm

		transition_state_prob = multi_variate_normal(zh_post.mean, self.transition_state.mu[0, :self.z_dim], self.transition_state.sigma[0, :self.z_dim, :self.z_dim], log=False)

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
			active_segment = self.alpha_t.argmax().item()

		if self.args.action=='handshake':
			if not self.crossed:
				if (active_segment==0 and self.alpha_t[active_segment]<transition_state_prob) or (active_segment!=0 and self.predicted_segment[-1]==0) :
					self.crossed = True
					print('crossed')

		q = xr_cond.reshape((self.model_r.window_size, self.model_r.num_joints))[0].cpu().numpy()#.mean(0).cpu().numpy()
		
		self.joint_trajectory.points[0].positions = 0.2*np.array(self.joint_trajectory.points[0].positions) + 0.8*np.array(q.tolist() + [1., self.robot_hand_joint])
		self.joint_trajectory.points[0].positions[0] -= np.deg2rad(15)
		
		if args.ik_only: # Putting the baseline here to simulate the same perception-reaction delay as with the network
			super().step(nui_skeleton, hand_pose, optimizer="least_squares")
			if self.args.action=='handshake' and self.joint_readings[0] < 0.35:
				self.joint_trajectory.points[0].effort[0] = 0.15
		elif (active_segment>0 and active_segment<self.ssm.nb_states-1) or \
			(self.args.action=='handshake' and self.crossed) or \
			(self.args.action=='rocket' and active_segment==0 and self.alpha_t[active_segment]<0.6) or \
			(active_segment==self.ssm.nb_states-1 and self.alpha_t[active_segment]<0.6):
			if (self.args.action=='handshake' and active_segment==0 and self.alpha_t[active_segment]<transition_state_prob):
				print('Transition state')
			if self.args.ik:
				self.ik_result = self.pepper_chain.active_to_full(self.joint_trajectory.points[0].positions[0:4], [0] * len(self.pepper_chain.links))
				super().step(nui_skeleton, hand_pose, optimizer="scalar",  regularization_parameter=0.01)
			if self.args.action=='handshake': # Handshake Stiffness control
				print(active_segment, self.still_reaching, np.linalg.norm(self.joint_readings[:4] - self.joint_trajectory.points[0].positions[0:4]), np.linalg.norm(self.joint_readings[0] - self.joint_trajectory.points[0].positions[0]))
				if self.still_reaching:
					if np.linalg.norm(self.joint_readings[:4] - self.joint_trajectory.points[0].positions[0:4]) < 0.1:
						self.still_reaching = False
						self.joint_trajectory.points[0].effort[0] = 0.15
				else:
					self.joint_trajectory.points[0].effort[0] = 0.15
		else:# self.args.action=='handshake':
			if self.joint_trajectory.points[0].effort[0] < 0.5:
				self.joint_trajectory.points[0].effort[0] += 0.01

		self.predicted_segment.append(active_segment)

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